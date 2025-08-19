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
    path: Path = typer.Option(..., exists=True, help="Input .npy or .csv (shape: n_ch×n or n×n_ch)"),
    out_dir: Path = typer.Option(Path("runs/demo1"), help="Output dir"),
    fs: int = typer.Option(250, help="Sampling rate (Hz)"),
    win: str = typer.Option("1s", help="Window (e.g., 1s, 500ms)"),
    hop: str = typer.Option("250ms", help="Hop (e.g., 250ms)"),
    line_freq: int = typer.Option(60, help="Mains frequency"),
    th_lnr: float = typer.Option(0.6, help="Line-noise ratio threshold"),
    th_burst_z: float = typer.Option(6.0, help="Burst RMS z threshold"),
    th_flat_std: float = typer.Option(1e-3, help="Flatline std threshold"),
    th_clip_pct: float = typer.Option(0.05, help="Clip percentage threshold"),
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

    x, channels = load_signal(path)
    n_ch, n = x.shape

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
    typer.echo(f"Wrote {out_dir / 'sqi.parquet'} and {events_path}")


# -------------------------
# Status command
# -------------------------
@app.command()
def status(
    session: Path = typer.Option(..., exists=True, help="Session dir with sqi.parquet"),
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
    typer.echo("  simulate - Generate synthetic multichannel signal")
    typer.echo("  ingest   - Compute SQIs + auto-events over sliding windows")
    typer.echo("  status   - Show live status of worst channels")
    typer.echo("  annotate - Append keypress annotations to events.jsonl")
    typer.echo("  info     - Show this information")


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
    
    # Step 2: Run QC
    typer.echo("\nStep 2: Running QC analysis...")
    ingest(Path("demo.npy"), Path("runs/demo1"), fs=250, win="1s", hop="250ms", line_freq=60, 
           th_lnr=0.6, th_burst_z=6.0, th_flat_std=1e-3, th_clip_pct=0.05)
    
    # Step 3: Show results
    typer.echo("\nStep 3: Results summary:")
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
