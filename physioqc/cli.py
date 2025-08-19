#!/usr/bin/env python3
"""
PhysioQC - Headless CLI for real-time Signal Quality + auto-event logging.

Commands:
  - simulate  : create a synthetic multichannel signal (.npy or .csv)
  - ingest    : compute SQIs in sliding windows; write Parquet/JSONL/JSON
  - annotate  : keyboard-driven annotation system (append to events.jsonl)

Features comprehensive SQI computation, auto-event detection, and provenance tracking.
Dependencies: typer, numpy, scipy, pandas, pyarrow, readchar.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import typer
from numpy.typing import NDArray
from scipy.signal import welch

try:
    import readchar  # for annotate
except ImportError:  # graceful fallback for optional dependency
    readchar = None

app = typer.Typer(add_completion=False, help="PhysioQC headless CLI for signal quality assessment")

# -------------------------
# Config / Provenance types
# -------------------------
@dataclass
class Config:
    fs: int = 250
    line_freq: int = 60
    window_s: float = 1.0
    hop_s: float = 0.25
    bands_lf: Tuple[float, float] = (1.0, 30.0)
    bands_hf: Tuple[float, float] = (30.0, 90.0)
    # quality thresholds
    th_lnr: float = 0.6
    th_burst_z: float = 6.0
    th_flat_std: float = 1e-3
    th_clip_pct: float = 0.05


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# -------------------------
# IO helpers
# -------------------------

def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_provenance(out_dir: Path, session_id: str, cmd: str, cfg: Config, channels: List[str]):
    prov = {
        "physioq_version": "0.1.0",
        "session_id": session_id,
        "commit": os.environ.get("PHYSIOQ_COMMIT", "dev"),
        "cmd": cmd,
        "started_at": utc_now_iso(),
        "host": {
            "name": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "os": sys.platform,
            "python": sys.version.split(" ")[0],
        },
        "params": {
            "fs": cfg.fs,
            "line_freq": cfg.line_freq,
            "window_s": cfg.window_s,
            "hop_s": cfg.hop_s,
            "bands": {"lf": cfg.bands_lf, "hf": cfg.bands_hf},
            "thresholds": {
                "lnr": cfg.th_lnr,
                "burst_z": cfg.th_burst_z,
                "flat_std": cfg.th_flat_std,
                "clip_pct": cfg.th_clip_pct,
            },
        },
        "channels": channels,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))


# -------------------------
# Simulation
# -------------------------

def simulate_signal(
    secs: float,
    fs: int,
    n_ch: int,
    line_freq: int = 60,
    bursts: bool = True,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Create multichannel signal with baseline noise, optional line noise + bursts."""
    rng = np.random.default_rng(seed)
    n = int(secs * fs)
    t = np.arange(n) / fs
    x = 0.05 * rng.standard_normal((n_ch, n)).astype(np.float32)
    # line noise
    x += 0.02 * np.sin(2 * np.pi * line_freq * t)[None, :]
    if bursts:
        for ch in range(n_ch):
            for _ in range(rng.integers(1, 3)):
                start = int(rng.uniform(0.2, 0.8) * n)
                length = int(0.05 * n)
                x[ch, start : start + length] += rng.normal(0.5, 0.1)
    return x


@app.command()
def simulate(
    out: Path = typer.Option(Path("demo.npy"), help="Output file (.npy or .csv)"),
    fs: int = typer.Option(250, help="Sampling rate (Hz)"),
    secs: float = typer.Option(10.0, help="Duration (s)"),
    n_ch: int = typer.Option(4, help="Number of channels"),
    line_freq: int = typer.Option(60, help="Mains frequency (Hz)"),
    seed: int | None = typer.Option(7, help="Random seed"),
):
    """Generate synthetic multichannel signal for demos/validation."""
    x = simulate_signal(secs=secs, fs=fs, n_ch=n_ch, line_freq=line_freq, seed=seed)
    if out.suffix.lower() == ".csv":
        df = pd.DataFrame(x.T, columns=[f"ch{i}" for i in range(n_ch)])
        df.to_csv(out, index=False)
    else:
        np.save(out, x)
    typer.echo(f"Wrote {out}")


# -------------------------
# Windowing
# -------------------------

def sliding_windows(n_samples: int, win: int, hop: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += hop


# -------------------------
# SQIs
# -------------------------

def composite_sqi(sqi_dict: dict) -> float:
    """Compute composite SQI score (0-100, where 100 is perfect quality)."""
    # Normalize individual SQIs to 0-1 scale
    rms_score = min(1.0, max(0.0, 1.0 - sqi_dict["rms"] / 0.1))  # Lower RMS is better
    dc_score = min(1.0, max(0.0, 1.0 - abs(sqi_dict["dc_offset"]) / 0.05))  # Lower DC offset is better
    lnr_score = min(1.0, max(0.0, 1.0 - sqi_dict["line_noise_ratio"]))  # Lower line noise is better
    hf_lf_score = min(1.0, max(0.0, sqi_dict["hf_lf_ratio"] / 2.0))  # Moderate HF/LF ratio is good
    clip_score = min(1.0, max(0.0, 1.0 - sqi_dict["clip_pct"]))  # Lower clipping is better
    flat_score = min(1.0, max(0.0, 1.0 - sqi_dict["flatline_ratio"]))  # Lower flatline is better
    dropout_score = min(1.0, max(0.0, 1.0 - sqi_dict["drop_ct"] / 10.0))  # Lower dropout count is better
    
    # Weighted combination (adjust weights based on importance)
    weights = {
        "rms": 0.25,
        "dc_offset": 0.15,
        "line_noise_ratio": 0.25,
        "hf_lf_ratio": 0.10,
        "clip_pct": 0.15,
        "flatline_ratio": 0.05,
        "drop_ct": 0.05
    }
    
    composite = (
        weights["rms"] * rms_score +
        weights["dc_offset"] * dc_score +
        weights["line_noise_ratio"] * lnr_score +
        weights["hf_lf_ratio"] * hf_lf_score +
        weights["clip_pct"] * clip_score +
        weights["flatline_ratio"] * flat_score +
        weights["drop_ct"] * dropout_score
    )
    
    return round(composite * 100, 1)  # Return as percentage

def bandpower_welch(signal: NDArray[np.float32], fs: int, band: Tuple[float, float]) -> float:
    f, pxx = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    lo, hi = band
    idx = np.logical_and(f >= lo, f <= hi)
    return float(np.trapz(pxx[idx], f[idx]))


def rms(x: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(x * x)))


def dc_offset(x: NDArray[np.float32]) -> float:
    return float(np.mean(x))


def line_noise_ratio(x: NDArray[np.float32], fs: int, f0: int, bw: float = 1.0) -> float:
    total = bandpower_welch(x, fs, (0.5, fs / 2 - 1))
    ln = bandpower_welch(x, fs, (f0 - bw, f0 + bw))
    return float(ln / max(total, 1e-12))


def hf_lf_ratio(x: NDArray[np.float32], fs: int, lf: Tuple[float, float], hf: Tuple[float, float]) -> float:
    low = bandpower_welch(x, fs, lf)
    high = bandpower_welch(x, fs, hf)
    return float(high / max(low, 1e-12))


def clip_pct(x: NDArray[np.float32], eps: float | None = None) -> float:
    if eps is None:
        # define rails as per-window min/max; near-rail within 1% of range
        x_min, x_max = float(x.min()), float(x.max())
        rng = max(x_max - x_min, 1e-12)
        eps = 0.01 * rng
        near = (np.abs(x - x_min) <= eps) | (np.abs(x - x_max) <= eps)
    else:
        near = (x <= eps) | (x >= eps)
    return float(np.mean(near))


def flatline_ratio(x: NDArray[np.float32], std_th: float) -> float:
    s = float(np.std(x))
    return 1.0 if s < std_th else 0.0


def dropout_count(x: NDArray[np.float32], zero_th: float = 1e-9, min_len: int = 3) -> int:
    # Count contiguous sequences of exact zeros or NaNs of length >= min_len
    z = (np.abs(x) <= zero_th) | np.isnan(x)
    if not z.any():
        return 0
    count, run = 0, 0
    for b in z:
        if b:
            run += 1
            if run == min_len:
                count += 1
        else:
            run = 0
    return count


# -------------------------
# Events writer
# -------------------------

def append_event(out_path: Path, event: dict):
    with out_path.open("a") as f:
        f.write(json.dumps(event) + "\n")


# -------------------------
# Decoder Integration
# -------------------------

@dataclass
class DecoderConfig:
    name: str
    type: str  # "python", "subprocess", "socket"
    path: str | None = None
    args: List[str] | None = None
    timeout: float = 5.0
    buffer_size: int = 1000

class DecoderInterface:
    """Interface for external decoders."""
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.process = None
        self.decoder_output = []
        
    def start(self):
        """Start the decoder process."""
        if self.config.type == "subprocess" and self.config.path:
            import subprocess
            cmd = [self.config.path] + (self.config.args or [])
            self.process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
        elif self.config.type == "python":
            # Python module import
            import importlib
            try:
                module = importlib.import_module(self.config.path)
                self.decoder = module
                if hasattr(module, 'initialize'):
                    module.initialize()
            except ImportError:
                raise ValueError(f"Could not import decoder module: {self.config.path}")
    
    def process_data(self, data: NDArray[np.float32], timestamp: int) -> dict | None:
        """Process data through decoder and return results."""
        if self.config.type == "subprocess" and self.process:
            try:
                # Send data to subprocess
                data_str = ",".join(map(str, data.flatten()))
                input_line = f"{timestamp},{data_str}\n"
                self.process.stdin.write(input_line)
                self.process.stdin.flush()
                
                # Read output with timeout
                import select
                if select.select([self.process.stdout], [], [], self.config.timeout)[0]:
                    output = self.process.stdout.readline().strip()
                    if output:
                        return self._parse_decoder_output(output, timestamp)
            except Exception as e:
                typer.echo(f"Decoder error: {e}")
                
        elif self.config.type == "python" and hasattr(self, 'decoder'):
            try:
                if hasattr(self.decoder, 'process'):
                    result = self.decoder.process(data, timestamp)
                    return result
            except Exception as e:
                typer.echo(f"Python decoder error: {e}")
        
        return None
    
    def _parse_decoder_output(self, output: str, timestamp: int) -> dict:
        """Parse decoder output into structured format."""
        try:
            # Simple CSV parsing - can be extended for different formats
            parts = output.split(',')
            if len(parts) >= 2:
                return {
                    "timestamp": timestamp,
                    "decoder": self.config.name,
                    "prediction": parts[0],
                    "confidence": float(parts[1]) if len(parts) > 1 else 1.0,
                    "raw_output": output
                }
        except:
            pass
        
        return {
            "timestamp": timestamp,
            "decoder": self.config.name,
            "prediction": "unknown",
            "confidence": 0.0,
            "raw_output": output
        }
    
    def stop(self):
        """Stop the decoder process."""
        if self.process:
            self.process.terminate()
            self.process.wait()

def compute_performance_metrics(predictions: List[dict], ground_truth: List[str] | None = None) -> dict:
    """Compute decoder performance metrics."""
    if not predictions:
        return {"accuracy": 0.0, "latency_ms": 0.0, "throughput_hz": 0.0}
    
    # Basic metrics
    total_predictions = len(predictions)
    avg_confidence = sum(p.get("confidence", 0.0) for p in predictions) / total_predictions
    
    # Latency calculation (if timestamps available)
    latencies = []
    for i in range(1, len(predictions)):
        if "timestamp" in predictions[i] and "timestamp" in predictions[i-1]:
            latency = (predictions[i]["timestamp"] - predictions[i-1]["timestamp"]) / 1e6  # to ms
            latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    # Accuracy (if ground truth provided)
    accuracy = 0.0
    if ground_truth and len(ground_truth) == len(predictions):
        correct = sum(1 for p, gt in zip(predictions, ground_truth) 
                     if p.get("prediction") == gt)
        accuracy = correct / len(predictions)
    
    return {
        "accuracy": round(accuracy, 3),
        "avg_confidence": round(avg_confidence, 3),
        "latency_ms": round(avg_latency, 2),
        "throughput_hz": round(throughput, 1),
        "total_predictions": total_predictions
    }

# -------------------------
# Multi-Dataset Support
# -------------------------

@dataclass
class DataSource:
    type: str  # "file", "lsl", "simulated", "bci_competition"
    path: str | None = None
    stream_name: str | None = None
    duration: float | None = None
    channels: List[str] | None = None

def load_data_source(source: str, **kwargs) -> Tuple[NDArray[np.float32], List[str]]:
    """Load data from various sources."""
    if source.startswith("file:"):
        # File-based sources
        file_path = source[5:]
        return load_signal(Path(file_path))
    
    elif source.startswith("lsl:"):
        # LabStreamingLayer streams (OpenBCI, etc.)
        stream_name = source[4:]
        return load_lsl_stream(stream_name, **kwargs)
    
    elif source.startswith("simulated:"):
        # Simulated data with parameters
        params = source[10:].split(",")
        return load_simulated_data(params, **kwargs)
    
    elif source.startswith("bci_competition:"):
        # BCI Competition datasets
        dataset_info = source[15:].split(",")
        return load_bci_competition_data(dataset_info, **kwargs)
    
    else:
        # Default to file
        return load_signal(Path(source))

def load_lsl_stream(stream_name: str, duration: float = 10.0) -> Tuple[NDArray[np.float32], List[str]]:
    """Load data from LSL stream (OpenBCI, etc.)."""
    try:
        import pylsl
        
        # Find the stream
        streams = pylsl.resolve_streams()
        target_stream = None
        for stream in streams:
            if stream_name.lower() in stream.name().lower():
                target_stream = stream
                break
        
        if not target_stream:
            raise ValueError(f"LSL stream '{stream_name}' not found")
        
        # Create inlet and collect data
        inlet = pylsl.StreamInlet(target_stream)
        inlet.open_stream()
        
        typer.echo(f"Collecting {duration}s from LSL stream: {target_stream.name()}")
        
        # Collect data
        data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            sample, timestamp = inlet.pull_sample()
            data.append(sample)
            time.sleep(0.001)  # 1ms
        
        inlet.close_stream()
        
        if not data:
            raise ValueError("No data collected from LSL stream")
        
        # Convert to numpy array
        data_array = np.array(data, dtype=np.float32).T  # (n_channels, n_samples)
        channels = [f"ch{i}" for i in range(data_array.shape[0])]
        
        typer.echo(f"Collected {data_array.shape[1]} samples from {len(channels)} channels")
        return data_array, channels
        
    except ImportError:
        raise ImportError("pylsl not installed. Install with: pip install pylsl")
    except Exception as e:
        raise ValueError(f"Failed to load LSL stream: {e}")

def load_simulated_data(params: List[str], **kwargs) -> Tuple[NDArray[np.float32], List[str]]:
    """Load simulated data with specified parameters."""
    # Parse parameters: simulated:fs=250,secs=10,n_ch=4,line_freq=60
    config = {}
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)
            try:
                if "." in value:
                    config[key] = float(value)
                else:
                    config[key] = int(value)
            except ValueError:
                config[key] = value
    
    # Default values
    fs = config.get("fs", 250)
    secs = config.get("secs", 10.0)
    n_ch = config.get("n_ch", 4)
    line_freq = config.get("line_freq", 60)
    seed = config.get("seed", None)
    
    # Generate simulated data
    data = simulate_signal(secs=secs, fs=fs, n_ch=n_ch, line_freq=line_freq, seed=seed)
    channels = [f"ch{i}" for i in range(n_ch)]
    
    typer.echo(f"Generated simulated data: {n_ch} channels, {secs}s, {fs}Hz")
    return data, channels

def load_bci_competition_data(dataset_info: List[str], **kwargs) -> Tuple[NDArray[np.float32], List[str]]:
    """Load BCI Competition datasets."""
    # This is a placeholder for BCI Competition dataset loading
    # In a real implementation, you'd download and load specific datasets
    
    dataset_name = dataset_info[0] if dataset_info else "bci_iv_2a"
    
    typer.echo(f"BCI Competition dataset '{dataset_name}' loading not yet implemented")
    typer.echo("This would download and load the specified BCI Competition dataset")
    
    # For now, return simulated data that mimics BCI Competition format
    # 22 channels, 250Hz, 4 classes
    data = simulate_signal(secs=10.0, fs=250, n_ch=22, line_freq=50, seed=42)
    channels = [f"C{i+1:02d}" for i in range(22)]  # C01, C02, ..., C22
    
    return data, channels

# -------------------------
# Ingest/QC
# -------------------------

def load_signal(path: Path) -> Tuple[NDArray[np.float32], List[str]]:
    if path.suffix.lower() == ".npy":
        x = np.load(path)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] < x.shape[1]:
            # assume (n_ch, n)
            n_ch = x.shape[0]
            channels = [f"ch{i}" for i in range(n_ch)]
            return x.astype(np.float32), channels
        else:
            # assume (n, n_ch)
            x = x.T
            n_ch = x.shape[0]
            channels = [f"ch{i}" for i in range(n_ch)]
            return x.astype(np.float32), channels
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        channels = list(df.columns)
        x = df.values.T.astype(np.float32)
        return x, channels
    else:
        raise ValueError("Unsupported file type. Use .npy or .csv")


@app.command()
def ingest(
    source: str = typer.Option(..., help="Data source: file:path, lsl:stream_name, simulated:params, or bci_competition:dataset"),
    out_dir: Path = typer.Option(Path("runs/demo1"), help="Output dir"),
    fs: int = typer.Option(250, help="Sampling rate (Hz)"),
    win: str = typer.Option("1s", help="Window (e.g., 1s, 500ms)"),
    hop: str = typer.Option("250ms", help="Hop (e.g., 250ms)"),
    line_freq: int = typer.Option(60, help="Mains frequency"),
    th_lnr: float = typer.Option(0.6, help="Line-noise ratio threshold"),
    th_burst_z: float = typer.Option(6.0, help="Burst RMS z threshold"),
    th_flat_std: float = typer.Option(1e-3, help="Flatline std threshold"),
    th_clip_pct: float = typer.Option(0.05, help="Clip percentage threshold"),
    decode: str | None = typer.Option(None, help="Decoder to use (e.g., 'python:my_decoder' or 'subprocess:/path/to/decoder')"),
    decode_args: str | None = typer.Option(None, help="Additional arguments for decoder"),
    duration: float = typer.Option(None, help="Duration for LSL streams (seconds)"),
):
    """Compute SQIs + auto-events over sliding windows; write Parquet/JSONL/JSON."""
    # parse durations
    def parse_duration(s: str) -> float:
        s = s.strip().lower()
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1])
        return float(s)

    cfg = Config(
        fs=fs,
        line_freq=line_freq,
        window_s=parse_duration(win),
        hop_s=parse_duration(hop),
        th_lnr=float(th_lnr),
        th_burst_z=float(th_burst_z),
        th_flat_std=float(th_flat_std),
        th_clip_pct=float(th_clip_pct),
    )

    cmd = "physioq " + " ".join(sys.argv[1:])
    ensure_out_dir(out_dir)
    session_id = str(uuid.uuid4())

    # Load data from specified source
    try:
        if duration:
            x, channels = load_data_source(source, duration=duration)
        else:
            x, channels = load_data_source(source)
    except Exception as e:
        typer.echo(f"Failed to load data source '{source}': {e}")
        raise typer.Exit(code=1)
    
    n_ch, n = x.shape

    # Initialize decoder if specified
    decoder = None
    decode_rows = []
    if decode:
        try:
            decode_type, decode_path = decode.split(":", 1)
            decoder_config = DecoderConfig(
                name=decode_path.split("/")[-1] if "/" in decode_path else decode_path,
                type=decode_type,
                path=decode_path,
                args=decode_args.split() if decode_args else None
            )
            decoder = DecoderInterface(decoder_config)
            decoder.start()
            typer.echo(f"Started decoder: {decoder_config.name}")
        except Exception as e:
            typer.echo(f"Failed to start decoder: {e}")
            decoder = None

    # Prepare outputs
    save_provenance(out_dir, session_id, cmd, cfg, channels)
    events_path = out_dir / "events.jsonl"
    sqi_rows = []

    # Precompute RMS per channel for z-score baseline
    base_rms = np.array([rms(x[ch]) for ch in range(n_ch)], dtype=np.float64)
    base_mean = float(base_rms.mean())
    base_std = float(base_rms.std() + 1e-9)

    win_samp = int(round(cfg.window_s * fs))
    hop_samp = int(round(cfg.hop_s * fs))
    epoch_ns = lambda secs: int(secs * 1e9)

    start_ns = time.time_ns()  # reference timestamp for processing

    for i0, i1 in sliding_windows(n, win_samp, hop_samp):
        w = x[:, i0:i1]
        t0_ns = start_ns + epoch_ns(i0 / fs)
        t1_ns = start_ns + epoch_ns(i1 / fs)

        for ch_idx, ch_name in enumerate(channels):
            seg = w[ch_idx]
            sqi = {
                "session_id": session_id,
                "t_start": t0_ns,
                "t_end": t1_ns,
                "ch": ch_name,
                "rms": rms(seg),
                "dc_offset": dc_offset(seg),
                "line_noise_ratio": line_noise_ratio(seg, fs, cfg.line_freq, bw=1.0),
                "hf_lf_ratio": hf_lf_ratio(seg, fs, cfg.bands_lf, cfg.bands_hf),
                "clip_pct": clip_pct(seg),
                "flatline_ratio": flatline_ratio(seg, cfg.th_flat_std),
                "drop_ct": dropout_count(seg),
                "composite_sqi": 0.0,  # Will be computed below
                "flags": 0,
            }
            sqi["composite_sqi"] = composite_sqi(sqi)
            sqi_rows.append(sqi)

            # Process through decoder if available
            if decoder:
                try:
                    # Use the current window data for decoding
                    decode_result = decoder.process_data(w, t1_ns)
                    if decode_result:
                        decode_result["session_id"] = session_id
                        decode_result["window_start"] = t0_ns
                        decode_result["window_end"] = t1_ns
                        decode_rows.append(decode_result)
                except Exception as e:
                    typer.echo(f"Decoder processing error: {e}")

            # Automated event detection
            if sqi["line_noise_ratio"] > cfg.th_lnr:
                append_event(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "type": "auto",
                        "label": "line_noise",
                        "ch": [ch_name],
                        "level": round(sqi["line_noise_ratio"], 4),
                        "threshold": cfg.th_lnr,
                        "detector": "lnr_v1",
                        "context": {"win": cfg.window_s, "hop": cfg.hop_s},
                    },
                )

            z = (sqi["rms"] - base_mean) / base_std
            if z > cfg.th_burst_z:
                append_event(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "type": "auto",
                        "label": "burst",
                        "ch": [ch_name],
                        "level": round(float(z), 3),
                        "threshold": cfg.th_burst_z,
                        "detector": "rms_z_v1",
                        "context": {"win": cfg.window_s, "hop": cfg.hop_s},
                    },
                )

            if sqi["flatline_ratio"] >= 1.0:
                append_event(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "type": "auto",
                        "label": "flatline",
                        "ch": [ch_name],
                        "level": 1.0,
                        "threshold": cfg.th_flat_std,
                        "detector": "flat_v1",
                        "context": {"win": cfg.window_s, "hop": cfg.hop_s},
                    },
                )

            if sqi["clip_pct"] > cfg.th_clip_pct:
                append_event(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "type": "auto",
                        "label": "clipping",
                        "ch": [ch_name],
                        "level": round(sqi["clip_pct"], 3),
                        "threshold": cfg.th_clip_pct,
                        "detector": "clip_v1",
                        "context": {"win": cfg.window_s, "hop": cfg.hop_s},
                    },
                )

    # Write SQIs
    df = pd.DataFrame.from_records(sqi_rows)
    df.to_parquet(out_dir / "sqi.parquet", index=False)
    
    # Write decoder outputs and compute performance metrics
    if decoder and decode_rows:
        decode_df = pd.DataFrame.from_records(decode_rows)
        decode_df.to_parquet(out_dir / "decode.parquet", index=False)
        
        # Compute performance metrics
        metrics = compute_performance_metrics(decode_rows)
        metrics["session_id"] = session_id
        metrics["decoder"] = decoder.config.name
        
        # Save metrics
        with open(out_dir / "decode_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        typer.echo(f"Decoder performance: {metrics['accuracy']:.1%} accuracy, "
                  f"{metrics['latency_ms']:.1f}ms latency, "
                  f"{metrics['throughput_hz']:.1f}Hz throughput")
        
        # Stop decoder
        decoder.stop()
    
    typer.echo(f"Wrote {out_dir / 'sqi.parquet'} and {events_path}")
    if decoder and decode_rows:
        typer.echo(f"Wrote {out_dir / 'decode.parquet'} and {out_dir / 'decode_metrics.json'}")


# -------------------------
# Phase 5: Advanced Visualization & Testing
# -------------------------

def create_web_dashboard():
    """Create a lightweight web dashboard for live visualization."""
    try:
        from fastapi import FastAPI, WebSocket
        from fastapi.responses import HTMLResponse
        import asyncio
        
        app = FastAPI(title="PhysioQC Live Dashboard")
        
        # Store for live data
        live_data = {"sqi": [], "predictions": []}
        
        @app.get("/")
        async def dashboard():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>PhysioQC Live Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #1e1e1e; color: white; }
                    .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .panel { background: #2d2d2d; padding: 20px; border-radius: 8px; }
                    h1 { text-align: center; color: #4CAF50; }
                    .metric { display: inline-block; margin: 10px; padding: 10px; background: #3d3d3d; border-radius: 4px; }
                </style>
            </head>
            <body>
                <h1>PhysioQC Live Dashboard</h1>
                <div class="container">
                    <div class="panel">
                        <h2>Signal Quality Heatmap</h2>
                        <div id="sqi-heatmap" style="height: 400px;"></div>
                    </div>
                    <div class="panel">
                        <h2>Decoder Output</h2>
                        <div id="decoder-plot" style="height: 400px;"></div>
                    </div>
                </div>
                <div class="panel">
                    <h2>Live Metrics</h2>
                    <div id="metrics"></div>
                </div>
                
                <script>
                    const ws = new WebSocket("ws://localhost:8000/ws");
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    function updateDashboard(data) {
                        if (data.type === 'sqi') {
                            updateSQIHeatmap(data.payload);
                        } else if (data.type === 'decoder') {
                            updateDecoderPlot(data.payload);
                        }
                    }
                    
                    function updateSQIHeatmap(sqi_data) {
                        if (!sqi_data.length) return;
                        const channels = sqi_data.map(d => d.channel);
                        const scores = sqi_data.map(d => d.composite_sqi);
                        
                        const trace = {
                            z: [scores],
                            x: channels,
                            type: 'heatmap',
                            colorscale: [[0, 'red'], [0.5, 'yellow'], [1, 'green']],
                            zmin: 0,
                            zmax: 100
                        };
                        
                        Plotly.newPlot('sqi-heatmap', [trace], {
                            title: 'Channel Quality (0-100)',
                            paper_bgcolor: '#2d2d2d',
                            plot_bgcolor: '#2d2d2d',
                            font: {color: 'white'}
                        });
                    }
                    
                    function updateDecoderPlot(decoder_data) {
                        if (!decoder_data.length) return;
                        const times = decoder_data.map((d, i) => i);
                        const confidences = decoder_data.map(d => d.confidence);
                        
                        const trace = {
                            x: times,
                            y: confidences,
                            mode: 'lines+markers',
                            type: 'scatter',
                            name: 'Confidence',
                            line: {color: '#4CAF50'}
                        };
                        
                        Plotly.newPlot('decoder-plot', [trace], {
                            title: 'Decoder Confidence Over Time',
                            paper_bgcolor: '#2d2d2d',
                            plot_bgcolor: '#2d2d2d',
                            font: {color: 'white'},
                            yaxis: {range: [0, 1]}
                        });
                    }
                </script>
            </body>
            </html>
            """)
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    if live_data["sqi"]:
                        await websocket.send_json({
                            "type": "sqi",
                            "payload": live_data["sqi"][-10:]
                        })
                    
                    if live_data["predictions"]:
                        await websocket.send_json({
                            "type": "decoder", 
                            "payload": live_data["predictions"][-20:]
                        })
                    
                    await asyncio.sleep(2)
            except:
                pass
        
        return app, live_data
        
    except ImportError:
        raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")

def inject_fault(data: NDArray[np.float32], fault_type: str, **kwargs) -> NDArray[np.float32]:
    """Inject specific types of faults into signal data for testing."""
    corrupted = data.copy()
    n_ch, n_samples = corrupted.shape
    
    if fault_type == "line_burst":
        freq = kwargs.get("freq", 60)
        channels = kwargs.get("channels", [0])
        
        for ch in channels:
            if ch < n_ch:
                t = np.arange(n_samples) / kwargs.get("fs", 250)
                burst_start = int(0.3 * n_samples)
                burst_end = int(0.7 * n_samples)
                line_signal = 0.5 * np.sin(2 * np.pi * freq * t)
                corrupted[ch, burst_start:burst_end] += line_signal[burst_start:burst_end]
    
    elif fault_type == "dropout":
        channels = kwargs.get("channels", [0, 1])
        dropout_len = kwargs.get("length", 100)
        
        for ch in channels:
            if ch < n_ch:
                start = np.random.randint(0, n_samples - dropout_len)
                corrupted[ch, start:start + dropout_len] = 0
    
    elif fault_type == "clipping":
        channels = kwargs.get("channels", [0])
        clip_level = kwargs.get("level", 0.1)
        
        for ch in channels:
            if ch < n_ch:
                corrupted[ch] = np.clip(corrupted[ch], -clip_level, clip_level)
    
    return corrupted

# -------------------------
# Export Helpers
# -------------------------

def export_to_npz(session_dir: Path, output_path: Path | None = None):
    """Export session data to NumPy .npz format."""
    if output_path is None:
        output_path = session_dir / "export.npz"
    
    # Load SQI data
    sqi_path = session_dir / "sqi.parquet"
    if not sqi_path.exists():
        raise ValueError(f"No SQI data found in {session_dir}")
    
    df = pd.read_parquet(sqi_path)
    
    # Organize data by channel
    channels = df['ch'].unique()
    data = {}
    
    for ch in channels:
        ch_data = df[df['ch'] == ch]
        data[f'{ch}_rms'] = ch_data['rms'].values
        data[f'{ch}_composite_sqi'] = ch_data['composite_sqi'].values
        data[f'{ch}_line_noise'] = ch_data['line_noise_ratio'].values
        data[f'{ch}_timestamps'] = ch_data['t_start'].values
    
    # Add metadata
    data['channels'] = channels
    data['sampling_rate'] = 250  # Default, could be extracted from provenance
    
    np.savez_compressed(output_path, **data)
    return output_path

def export_to_mat(session_dir: Path, output_path: Path | None = None):
    """Export session data to MATLAB .mat format."""
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy not installed. Install with: pip install scipy")
    
    if output_path is None:
        output_path = session_dir / "export.mat"
    
    # Load SQI data
    sqi_path = session_dir / "sqi.parquet"
    if not sqi_path.exists():
        raise ValueError(f"No SQI data found in {session_dir}")
    
    df = pd.read_parquet(sqi_path)
    
    # Organize data for MATLAB
    channels = df['ch'].unique()
    mat_data = {}
    
    for ch in channels:
        ch_data = df[df['ch'] == ch]
        mat_data[f'{ch}_rms'] = ch_data['rms'].values
        mat_data[f'{ch}_composite_sqi'] = ch_data['composite_sqi'].values
        mat_data[f'{ch}_line_noise'] = ch_data['line_noise_ratio'].values
        mat_data[f'{ch}_timestamps'] = ch_data['t_start'].values
    
    # Add metadata
    mat_data['channels'] = channels
    mat_data['sampling_rate'] = 250
    
    sio.savemat(output_path, mat_data)
    return output_path

def run_benchmark(source: str, decoder: str | None = None, **kwargs) -> dict:
    """Run comprehensive benchmark of SQI and decoder performance."""
    import tempfile
    from pathlib import Path
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run ingest with all features
        typer.echo("Running benchmark...")
        
        # Load data to get basic info
        x, channels = load_data_source(source, **kwargs)
        n_ch, n_samples = x.shape
        duration = n_samples / kwargs.get('fs', 250)
        
        # Run QC analysis with proper parameter names
        ingest_params = {
            'source': source,
            'out_dir': temp_path,
            'decode': decoder,
            'fs': kwargs.get('fs', 250),
            'win': kwargs.get('win', '1s'),
            'hop': kwargs.get('hop', '250ms'),
            'line_freq': 60,
            'th_lnr': 0.6,
            'th_burst_z': 6.0,
            'th_flat_std': 1e-3,
            'th_clip_pct': 0.05
        }
        ingest(**ingest_params)
        
        # Load results
        sqi_df = pd.read_parquet(temp_path / "sqi.parquet")
        
        # Compute SQI statistics
        sqi_stats = {
            'mean_composite_sqi': float(sqi_df['composite_sqi'].mean()),
            'std_composite_sqi': float(sqi_df['composite_sqi'].std()),
            'min_composite_sqi': float(sqi_df['composite_sqi'].min()),
            'max_composite_sqi': float(sqi_df['composite_sqi'].max()),
            'channels_with_issues': int((sqi_df.groupby('ch')['composite_sqi'].mean() < 70).sum()),
            'total_windows': len(sqi_df),
            'data_duration_s': duration,
            'channels': len(channels),
            'sample_rate': kwargs.get('fs', 250)
        }
        
        # Load decoder metrics if available
        decode_metrics = {}
        if decoder and (temp_path / "decode_metrics.json").exists():
            with open(temp_path / "decode_metrics.json") as f:
                decode_metrics = json.load(f)
        
        # Combine results
        benchmark_results = {
            'timestamp': utc_now_iso(),
            'source': source,
            'decoder': decoder,
            'sqi_performance': sqi_stats,
            'decoder_performance': decode_metrics,
            'processing_time_s': 0.0  # Could be measured
        }
        
        return benchmark_results

# -------------------------
# Advanced Visualization Commands
# -------------------------

@app.command()
def dashboard(
    session: Path = typer.Option(None, exists=True, help="Session directory to monitor (optional)"),
    port: int = typer.Option(8000, help="Port for web dashboard"),
    host: str = typer.Option("localhost", help="Host for web dashboard"),
):
    """Launch live web dashboard for real-time signal quality and decoder monitoring."""
    try:
        import uvicorn
        
        app_instance, live_data = create_web_dashboard()
        
        # If session provided, load initial data
        if session:
            sqi_path = session / "sqi.parquet"
            if sqi_path.exists():
                df = pd.read_parquet(sqi_path)
                latest_sqi = df.groupby('ch').tail(1)
                live_data["sqi"] = [
                    {
                        "channel": row['ch'],
                        "composite_sqi": row['composite_sqi'],
                        "timestamp": row['t_end']
                    }
                    for _, row in latest_sqi.iterrows()
                ]
            
            decode_path = session / "decode.parquet"
            if decode_path.exists():
                decode_df = pd.read_parquet(decode_path)
                live_data["predictions"] = [
                    {
                        "timestamp": row['timestamp'],
                        "prediction": row['prediction'],
                        "confidence": row['confidence']
                    }
                    for _, row in decode_df.tail(20).iterrows()
                ]
        
        typer.echo(f"Starting PhysioQC monitoring dashboard at http://{host}:{port}")
        typer.echo("Press Ctrl+C to stop")
        
        uvicorn.run(app_instance, host=host, port=port, log_level="warning")
        
    except ImportError:
        typer.echo("FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Dashboard failed: {e}")
        raise typer.Exit(code=1)

@app.command()
def hotkeys(
    session: Path = typer.Argument(..., exists=True, help="Session directory"),
    keymap: str = typer.Option(
        "a:artifact,n:noise,s:stim,m:movement,b:burst,f:flatline,space:bookmark",
        help="Key mappings (key:label,key:label,...)"
    ),
):
    """Interactive hotkey annotation mode with live feedback."""
    from rich.console import Console
    from rich.table import Table
    
    if readchar is None:
        typer.echo("readchar not installed. Install with: pip install readchar")
        raise typer.Exit(code=1)
    
    # Parse keymap
    km = {}
    for pair in keymap.split(","):
        key, label = pair.split(":", 1)
        km[key.strip()] = label.strip()
    
    console = Console()
    events_path = session / "events.jsonl"
    
    # Create help table
    help_table = Table(title="PhysioQC Hotkey Annotations")
    help_table.add_column("Key", style="cyan")
    help_table.add_column("Action", style="green")
    
    for key, label in km.items():
        help_table.add_row(key, label)
    help_table.add_row("q", "Quit annotation mode")
    
    console.print(help_table)
    console.print("\n[green]Annotation mode active. Press keys to annotate events.[/green]")
    
    annotation_count = 0
    
    while True:
        try:
            key = readchar.readkey()
            
            if key == 'q':
                break
            
            label = km.get(key)
            if label:
                event = {
                    "ts": utc_now_iso(),
                    "type": "hotkey",
                    "label": label,
                    "key": key,
                    "user": os.getenv("USER", "user"),
                    "annotation_id": str(uuid.uuid4())
                }
                
                append_event(events_path, event)
                annotation_count += 1
                
                console.print(f"[yellow]+ {label}[/yellow] (#{annotation_count})")
            
        except KeyboardInterrupt:
            break
    
    console.print(f"\n[green]Annotation complete. Added {annotation_count} annotations.[/green]")

@app.command()
def inject_noise(
    source: str = typer.Option(..., help="Data source"),
    out_dir: Path = typer.Option(Path("runs/fault_test"), help="Output directory"),
    pattern: str = typer.Option("line_burst", help="Fault pattern: line_burst, dropout, clipping"),
    freq: int = typer.Option(60, help="Frequency for line burst (Hz)"),
    channels: str = typer.Option("0,1", help="Channels to affect (comma-separated)"),
    fs: int = typer.Option(250, help="Sampling rate (Hz)"),
):
    """Inject specific noise patterns for signal quality assessment validation."""
    try:
        # Load clean data
        x_clean, channel_names = load_data_source(source)
        
        # Parse affected channels
        affected_channels = [int(ch.strip()) for ch in channels.split(",")]
        
        # Inject fault
        x_corrupted = inject_fault(
            x_clean,
            pattern,
            freq=freq,
            channels=affected_channels,
            fs=fs
        )
        
        # Save corrupted data
        ensure_out_dir(out_dir)
        corrupted_path = out_dir / f"corrupted_{pattern}.npy"
        np.save(corrupted_path, x_corrupted)
        
        # Run SQI analysis on both clean and corrupted
        typer.echo(f"Applied {pattern} pattern to channels {affected_channels}")
        
        # Analyze clean data
        clean_dir = out_dir / "clean"
        ensure_out_dir(clean_dir)
        clean_path = clean_dir / "clean.npy"
        np.save(clean_path, x_clean)
        
        typer.echo("Analyzing clean data...")
        ingest(f"file:{clean_path}", clean_dir, fs=fs, win="1s", hop="250ms", line_freq=60, 
               th_lnr=0.6, th_burst_z=6.0, th_flat_std=1e-3, th_clip_pct=0.05)
        
        typer.echo("Analyzing corrupted data...")
        ingest(f"file:{corrupted_path}", out_dir, fs=fs, win="1s", hop="250ms", line_freq=60,
               th_lnr=0.6, th_burst_z=6.0, th_flat_std=1e-3, th_clip_pct=0.05)
        
        # Compare results
        clean_sqi = pd.read_parquet(clean_dir / "sqi.parquet")
        corrupt_sqi = pd.read_parquet(out_dir / "sqi.parquet")
        
        # Show impact
        typer.echo(f"\nSignal validation results:")
        typer.echo(f"Clean SQI:     {clean_sqi['composite_sqi'].mean():.1f} ± {clean_sqi['composite_sqi'].std():.1f}")
        typer.echo(f"Corrupted SQI: {corrupt_sqi['composite_sqi'].mean():.1f} ± {corrupt_sqi['composite_sqi'].std():.1f}")
        
        # Show per-channel impact
        clean_by_ch = clean_sqi.groupby('ch')['composite_sqi'].mean()
        corrupt_by_ch = corrupt_sqi.groupby('ch')['composite_sqi'].mean()
        
        typer.echo(f"\nPer-channel quality impact:")
        for ch in channel_names:
            if ch in clean_by_ch.index and ch in corrupt_by_ch.index:
                impact = clean_by_ch[ch] - corrupt_by_ch[ch]
                marker = "🔴" if ch in [f"ch{i}" for i in affected_channels] else "⚪"
                typer.echo(f"{marker} {ch}: {impact:+.1f} quality points")
        
        typer.echo(f"\nFiles saved: {corrupted_path}, {out_dir}/sqi.parquet")
        
    except Exception as e:
        typer.echo(f"Signal validation failed: {e}")
        raise typer.Exit(code=1)

# -------------------------
# Export Commands
# -------------------------

@app.command()
def to_npz(
    session: Path = typer.Argument(..., exists=True, help="Session directory"),
    output: Path = typer.Option(None, help="Output .npz file path"),
):
    """Export session data to NumPy .npz format for analysis."""
    try:
        output_path = export_to_npz(session, output)
        typer.echo(f"Exported to: {output_path}")
        typer.echo("Contains: RMS, composite SQI, line noise, timestamps per channel")
    except Exception as e:
        typer.echo(f"Export failed: {e}")
        raise typer.Exit(code=1)

@app.command()
def to_mat(
    session: Path = typer.Argument(..., exists=True, help="Session directory"),
    output: Path = typer.Option(None, help="Output .mat file path"),
):
    """Export session data to MATLAB .mat format for analysis."""
    try:
        output_path = export_to_mat(session, output)
        typer.echo(f"Exported to: {output_path}")
        typer.echo("Contains: RMS, composite SQI, line noise, timestamps per channel")
    except Exception as e:
        typer.echo(f"Export failed: {e}")
        raise typer.Exit(code=1)

@app.command()
def benchmark(
    source: str = typer.Option(..., help="Data source for benchmarking"),
    decoder: str | None = typer.Option(None, help="Decoder to test"),
    out_file: Path = typer.Option(Path("benchmark_results.json"), help="Output file for results"),
    fs: int = typer.Option(250, help="Sampling rate (Hz)"),
    win: str = typer.Option("1s", help="Window size"),
    hop: str = typer.Option("250ms", help="Hop size"),
):
    """Run comprehensive benchmark of SQI and decoder performance."""
    try:
        # Parse durations
        def parse_duration(s: str) -> float:
            s = s.strip().lower()
            if s.endswith("ms"):
                return float(s[:-2]) / 1000.0
            if s.endswith("s"):
                return float(s[:-1])
            return float(s)
        
        results = run_benchmark(
            source=source,
            decoder=decoder,
            fs=fs,
            win=win,
            hop=hop
        )
        
        # Save results
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        typer.echo(f"Benchmark completed: {out_file}")
        typer.echo(f"Data: {results['sqi_performance']['channels']} channels, "
                  f"{results['sqi_performance']['data_duration_s']:.1f}s")
        typer.echo(f"SQI: {results['sqi_performance']['mean_composite_sqi']:.1f} ± "
                  f"{results['sqi_performance']['std_composite_sqi']:.1f}")
        if decoder and results['decoder_performance']:
            typer.echo(f"Decoder: {results['decoder_performance'].get('accuracy', 0):.1%} accuracy, "
                      f"{results['decoder_performance'].get('latency_ms', 0):.1f}ms latency")
        
    except Exception as e:
        typer.echo(f"Benchmark failed: {e}")
        raise typer.Exit(code=1)

@app.command()
def dry_run(
    source: str = typer.Option(..., help="Data source to test"),
    duration: float = typer.Option(5.0, help="Test duration (seconds)"),
):
    """Test data source connectivity and basic processing without saving results."""
    try:
        typer.echo(f"Testing data source: {source}")
        typer.echo(f"Duration: {duration}s")
        
        # Test data loading
        if source.startswith("lsl:"):
            x, channels = load_data_source(source, duration=duration)
        else:
            x, channels = load_data_source(source)
        
        typer.echo(f"✅ Data loaded: {len(channels)} channels, {x.shape[1]} samples")
        
        # Test basic SQI computation
        if x.shape[1] > 100:  # Enough samples for testing
            test_segment = x[:, :100]
            rms_val = rms(test_segment[0])
            composite = composite_sqi({
                'rms': rms_val,
                'dc_offset': dc_offset(test_segment[0]),
                'line_noise_ratio': line_noise_ratio(test_segment[0], 250, 60),
                'hf_lf_ratio': hf_lf_ratio(test_segment[0], 250, (1.0, 30.0), (30.0, 90.0)),
                'clip_pct': clip_pct(test_segment[0]),
                'flatline_ratio': flatline_ratio(test_segment[0], 1e-3),
                'drop_ct': dropout_count(test_segment[0])
            })
            typer.echo(f"✅ SQI test: RMS={rms_val:.4f}, Composite={composite:.1f}")
        
        typer.echo("✅ Dry run completed successfully!")
        typer.echo(f"Ready to run: physioq ingest --source '{source}' --out-dir runs/my_session")
        
    except Exception as e:
        typer.echo(f"❌ Dry run failed: {e}")
        raise typer.Exit(code=1)

# -------------------------
# Sources command
# -------------------------
@app.command()
def sources(
    list_lsl: bool = typer.Option(False, help="List available LSL streams"),
    list_bci: bool = typer.Option(False, help="List available BCI Competition datasets"),
):
    """List available data sources and streams."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    if list_lsl:
        try:
            import pylsl
            streams = pylsl.resolve_streams()
            
            if not streams:
                console.print("[yellow]No LSL streams found. Make sure your device is connected and streaming.[/yellow]")
                return
            
            table = Table(title="Available LSL Streams")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Channels", style="green")
            table.add_column("Sample Rate", style="yellow")
            table.add_column("Source ID", style="blue")
            
            for stream in streams:
                table.add_row(
                    stream.name(),
                    stream.type(),
                    str(stream.channel_count()),
                    f"{stream.nominal_srate():.0f} Hz" if stream.nominal_srate() > 0 else "Variable",
                    stream.source_id()
                )
            
            console.print(table)
            console.print(f"\n[green]Use: physioq ingest --source lsl:STREAM_NAME[/green]")
            
        except ImportError:
            console.print("[red]pylsl not installed. Install with: pip install pylsl[/red]")
        except Exception as e:
            console.print(f"[red]Error listing LSL streams: {e}[/red]")
    
    elif list_bci:
        table = Table(title="Available BCI Competition Datasets")
        table.add_column("Dataset", style="cyan")
        table.add_column("Description", style="magenta")
        table.add_column("Channels", style="green")
        table.add_column("Classes", style="yellow")
        table.add_column("Usage", style="blue")
        
        datasets = [
            ("bci_iv_2a", "BCI Competition IV Dataset 2a", "22 EEG", "4 classes", "bci_competition:bci_iv_2a"),
            ("bci_iv_2b", "BCI Competition IV Dataset 2b", "3 EEG", "2 classes", "bci_competition:bci_iv_2b"),
            ("neural_latents", "Neural Latents Benchmark", "Variable", "Variable", "bci_competition:neural_latents"),
        ]
        
        for dataset in datasets:
            table.add_row(*dataset)
        
        console.print(table)
        console.print(f"\n[green]Use: physioq ingest --source bci_competition:DATASET_NAME[/green]")
    
    else:
        # Show general help
        console.print("[bold]PhysioQC Data Sources[/bold]")
        console.print("=" * 50)
        
        help_table = Table()
        help_table.add_column("Source Type", style="cyan")
        help_table.add_column("Format", style="magenta")
        help_table.add_column("Example", style="green")
        
        help_table.add_row(
            "File",
            "file:path/to/data.npy",
            "file:demo.npy"
        )
        help_table.add_row(
            "LSL Stream",
            "lsl:stream_name",
            "lsl:OpenBCI"
        )
        help_table.add_row(
            "Simulated",
            "simulated:fs=250,secs=10,n_ch=4",
            "simulated:fs=500,secs=5,n_ch=8"
        )
        help_table.add_row(
            "BCI Competition",
            "bci_competition:dataset_name",
            "bci_competition:bci_iv_2a"
        )
        
        console.print(help_table)
        console.print(f"\n[green]Commands:[/green]")
        console.print(f"  physioq sources --list-lsl    # List available LSL streams")
        console.print(f"  physioq sources --list-bci    # List BCI Competition datasets")


# -------------------------
# Decode Status command
# -------------------------
@app.command()
def decode_status(
    session: Path = typer.Argument(..., exists=True, help="Session dir with decode.parquet"),
):
    """Show decoder performance metrics and recent predictions."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    # Load decoder metrics
    metrics_path = session / "decode_metrics.json"
    decode_path = session / "decode.parquet"
    
    if not metrics_path.exists():
        console.print("[red]No decoder metrics found. Run ingest with --decode to generate metrics.[/red]")
        return
    
    # Load metrics
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Create metrics table
    metrics_table = Table(title="Decoder Performance Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Decoder", metrics.get("decoder", "unknown"))
    metrics_table.add_row("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    metrics_table.add_row("Avg Confidence", f"{metrics.get('avg_confidence', 0):.3f}")
    metrics_table.add_row("Latency", f"{metrics.get('latency_ms', 0):.1f} ms")
    metrics_table.add_row("Throughput", f"{metrics.get('throughput_hz', 0):.1f} Hz")
    metrics_table.add_row("Total Predictions", str(metrics.get("total_predictions", 0)))
    
    console.print(metrics_table)
    
    # Show recent predictions if available
    if decode_path.exists():
        try:
            df = pd.read_parquet(decode_path)
            if not df.empty:
                recent_table = Table(title="Recent Predictions (Last 10)")
                recent_table.add_column("Timestamp", style="cyan")
                recent_table.add_column("Prediction", style="magenta")
                recent_table.add_column("Confidence", style="yellow")
                
                for _, row in df.tail(10).iterrows():
                    timestamp = datetime.fromtimestamp(row['timestamp'] / 1e9).strftime("%H:%M:%S.%f")[:-3]
                    recent_table.add_row(
                        timestamp,
                        row.get('prediction', 'unknown'),
                        f"{row.get('confidence', 0):.3f}"
                    )
                
                console.print(recent_table)
        except Exception as e:
            console.print(f"[red]Error loading predictions: {e}[/red]")


# -------------------------
# Status command
# -------------------------
@app.command()
def status(
    session: Path = typer.Argument(..., exists=True, help="Session dir with sqi.parquet"),
    top_n: int = typer.Option(10, help="Number of worst channels to show"),
    refresh: float = typer.Option(2.0, help="Refresh interval in seconds"),
):
    """Show live status of worst channels based on composite SQI scores."""
    import time
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    
    console = Console()
    
    def generate_table():
        table = Table(title=f"PhysioQC Status - Top {top_n} Worst Channels")
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Channel", style="magenta")
        table.add_column("Composite SQI", style="red")
        table.add_column("RMS", style="yellow")
        table.add_column("Line Noise", style="yellow")
        table.add_column("Clipping %", style="yellow")
        table.add_column("Last Update", style="green")
        
        sqi_path = session / "sqi.parquet"
        if not sqi_path.exists():
            table.add_row("N/A", "No data", "N/A", "N/A", "N/A", "N/A", "N/A")
            return table
        
        try:
            df = pd.read_parquet(sqi_path)
            if df.empty:
                table.add_row("N/A", "No data", "N/A", "N/A", "N/A", "N/A", "N/A")
                return table
            
            # Get latest SQI for each channel
            latest_sqi = df.groupby('ch').tail(1).copy()
            latest_sqi = latest_sqi.sort_values('composite_sqi').head(top_n)
            
            for i, (_, row) in enumerate(latest_sqi.iterrows(), 1):
                # Color coding based on SQI score
                sqi_color = "red" if row['composite_sqi'] < 50 else "yellow" if row['composite_sqi'] < 80 else "green"
                
                table.add_row(
                    str(i),
                    row['ch'],
                    f"[{sqi_color}]{row['composite_sqi']:.1f}[/{sqi_color}]",
                    f"{row['rms']:.4f}",
                    f"{row['line_noise_ratio']:.3f}",
                    f"{row['clip_pct']:.1%}",
                    datetime.fromtimestamp(row['t_end'] / 1e9).strftime("%H:%M:%S")
                )
                
        except Exception as e:
            table.add_row("ERROR", str(e), "N/A", "N/A", "N/A", "N/A", "N/A")
        
        return table
    
    try:
        with Live(generate_table(), refresh_per_second=1/refresh, console=console) as live:
            while True:
                live.update(generate_table())
                time.sleep(refresh)
    except KeyboardInterrupt:
        console.print("\n[yellow]Status monitoring stopped.[/yellow]")


# -------------------------
# Annotate
# -------------------------
@app.command()
def annotate(
    session: Path = typer.Option(..., exists=True, help="Session dir with events.jsonl"),
    keymap: str = typer.Option(
        "b:burst,n:noise,m:motion,x:artifact,space:bookmark",
        help="Comma list of key:label",
    ),
    user: str = typer.Option(os.getenv("USER", "user"), help="Annotator name"),
    ch: str | None = typer.Option(None, help="Channel to include (optional)"),
):
    """Append keypress annotations to events.jsonl (press 'q' to quit)."""
    events_path = session / "events.jsonl"
    if not events_path.exists():
        events_path.touch()

    if readchar is None:
        typer.secho("readchar not installed: pip install readchar", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    km = {}
    for pair in keymap.split(","):
        k, v = pair.split(":", 1)
        km[k] = v

    typer.secho("Annotation started — press 'q' to quit", fg=typer.colors.GREEN)
    while True:
        key = readchar.readkey()
        if key == "q":
            break
        label = km.get("space" if key == " " else key)
        if not label:
            continue
        event = {
            "ts": utc_now_iso(),
            "type": "key",
            "label": label,
            "ch": [ch] if ch else [],
            "key": "space" if key == " " else key,
            "user": user,
        }
        append_event(events_path, event)
        typer.echo(f"+ {label}")


# -------------------------
# Info command
# -------------------------
@app.command()
def info():
    """Show PhysioQC information and version."""
    from . import __version__
    
    typer.echo("PhysioQC Information")
    typer.echo("=" * 50)
    typer.echo(f"Version: {__version__}")
    typer.echo("Description: Headless CLI for real-time Signal Quality + auto-event logging")
    typer.echo("Tagline: Stream in → SQIs & auto‑events out")
    typer.echo()
    typer.echo("Available Commands:")
    typer.echo("  simulate      - Generate synthetic multichannel signal")
    typer.echo("  ingest        - Compute SQIs + auto-events over sliding windows")
    typer.echo("  sources       - List available data sources and streams")
    typer.echo("  status        - Show live status of worst channels")
    typer.echo("  decode-status - Show decoder performance metrics")
    typer.echo("  dashboard     - Launch live web dashboard for real-time monitoring")
    typer.echo("  hotkeys       - Interactive hotkey annotation mode")
    typer.echo("  inject-noise  - Apply noise patterns for quality validation")
    typer.echo("  to-npz        - Export session data to NumPy .npz format")
    typer.echo("  to-mat        - Export session data to MATLAB .mat format")
    typer.echo("  benchmark     - Run comprehensive SQI and decoder benchmarks")
    typer.echo("  dry-run       - Test data source connectivity without saving")
    typer.echo("  annotate      - Append keypress annotations to events.jsonl")
    typer.echo("  info          - Show this information")


# -------------------------
# Demo command
# -------------------------
@app.command()
def demo():
    """Run the 60-second demo workflow."""
    typer.echo("Running PhysioQC 60-second demo...")
    
    # Step 1: Create demo signal
    typer.echo("\nStep 1: Creating demo signal...")
    simulate(Path("demo.npy"), fs=250, secs=10.0, n_ch=4, line_freq=60, seed=7)
    
    # Step 2: Run QC with file source
    typer.echo("\nStep 2: Running QC analysis (file source)...")
    ingest("file:demo.npy", Path("runs/demo1"), fs=250, win="1s", hop="250ms", line_freq=60, 
           th_lnr=0.6, th_burst_z=6.0, th_flat_std=1e-3, th_clip_pct=0.05)
    
    # Step 2.5: Run QC with simulated source
    typer.echo("\nStep 2.5: Running QC analysis (simulated source)...")
    ingest("simulated:fs=250,secs=5,n_ch=8,line_freq=60", Path("runs/demo2"), fs=250, win="1s", hop="250ms", line_freq=60,
           th_lnr=0.6, th_burst_z=6.0, th_flat_std=1e-3, th_clip_pct=0.05)
    
    # Step 2.6: Run with decoder (if example decoder exists)
    if Path("example_decoder.py").exists():
        typer.echo("\nStep 2.6: Running QC with decoder integration...")
        ingest("file:demo.npy", Path("runs/demo1_decoded"), fs=250, win="1s", hop="250ms", line_freq=60,
               th_lnr=0.6, th_burst_z=6.0, th_flat_std=1e-3, th_clip_pct=0.05,
               decode="python:example_decoder")
    
    # Step 3: Export and benchmark
    typer.echo("\nStep 3: Exporting data for analysis...")
    try:
        to_npz(Path("runs/demo1"))
        to_mat(Path("runs/demo1"))
    except Exception as e:
        typer.echo(f"Export failed: {e}")
    
    typer.echo("\nStep 3.5: Running benchmark...")
    try:
        benchmark("simulated:fs=250,secs=3,n_ch=4", decoder="python:example_decoder" if Path("example_decoder.py").exists() else None)
    except Exception as e:
        typer.echo(f"Benchmark failed: {e}")
    
    # Step 4: Show results
    typer.echo("\nStep 4: Results summary:")
    if Path("runs/demo1").exists():
        for file in Path("runs/demo1").iterdir():
            typer.echo(f"  ✓ {file.name}")
    
    typer.echo("\nDemo complete!")
    typer.echo("Next: Run 'physioq annotate runs/demo1' to add annotations")


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    app()
