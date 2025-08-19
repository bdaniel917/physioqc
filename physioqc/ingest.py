"""Process signals and compute SQIs in real-time."""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from .sqi import compute_all_sqis


def run(path: str, fs: int, win: float = 1.0, hop: float = 0.25, 
        line_freq: int = 60, out_dir: str = "runs/out") -> None:
    """Process a signal file and compute SQIs.
    
    Args:
        path: Path to input signal file (.npy)
        fs: Sampling frequency in Hz
        win: Window size in seconds
        hop: Hop size in seconds
        line_freq: Line frequency for noise detection
        out_dir: Output directory for results
    """
    # Load signal
    signal = np.load(path)
    print(f"Loaded signal: {signal.shape}")
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Calculate window parameters
    win_samples = int(win * fs)
    hop_samples = int(hop * fs)
    
    # Initialize results
    sqi_results = []
    events = []
    
    # Process signal in windows
    for start_idx in range(0, signal.shape[0] - win_samples + 1, hop_samples):
        end_idx = start_idx + win_samples
        window = signal[start_idx:end_idx, :]
        
        # Calculate timestamp
        timestamp = start_idx / fs
        
        # Compute SQIs for each channel
        for ch in range(window.shape[1]):
            sqi_dict = compute_all_sqis(window[:, ch], fs, line_freq)
            sqi_dict.update({
                "timestamp": timestamp,
                "channel": ch,
                "window_start": start_idx,
                "window_end": end_idx,
            })
            sqi_results.append(sqi_dict)
            
            # Detect events based on SQI thresholds
            events.extend(detect_events(sqi_dict, timestamp, ch))
    
    # Save SQIs to Parquet
    sqi_df = pd.DataFrame(sqi_results)
    sqi_path = os.path.join(out_dir, "sqi.parquet")
    sqi_df.to_parquet(sqi_path, index=False)
    
    # Save events to JSONL
    events_path = os.path.join(out_dir, "events.jsonl")
    with open(events_path, 'w') as f:
        for event in events:
            f.write(json.dumps(event) + '\n')
    
    # Save provenance
    provenance = {
        "input_file": path,
        "sampling_frequency": fs,
        "window_size": win,
        "hop_size": hop,
        "line_frequency": line_freq,
        "signal_shape": list(signal.shape),
        "processing_time": datetime.now().isoformat(),
        "total_windows": len(sqi_results) // signal.shape[1],
        "total_events": len(events),
    }
    
    provenance_path = os.path.join(out_dir, "provenance.json")
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"Processing complete!")
    print(f"  SQIs saved to: {sqi_path}")
    print(f"  Events saved to: {events_path}")
    print(f"  Provenance saved to: {provenance_path}")
    print(f"  Total windows processed: {len(sqi_results) // signal.shape[1]}")
    print(f"  Total events detected: {len(events)}")


def detect_events(sqi_dict: Dict[str, Any], timestamp: float, channel: int) -> List[Dict[str, Any]]:
    """Detect events based on SQI thresholds.
    
    Args:
        sqi_dict: Dictionary containing SQI values
        timestamp: Current timestamp
        channel: Channel number
        
    Returns:
        List of detected events
    """
    events = []
    
    # Define thresholds (these could be configurable)
    thresholds = {
        "line_noise_ratio": 0.1,
        "clip_pct": 0.05,
        "flatline_ratio": 0.8,
        "hf_lf_ratio": 10.0,
    }
    
    # Check for line noise
    if sqi_dict["line_noise_ratio"] > thresholds["line_noise_ratio"]:
        events.append({
            "timestamp": timestamp,
            "channel": channel,
            "event_type": "line_noise",
            "severity": "high" if sqi_dict["line_noise_ratio"] > 0.2 else "medium",
            "value": sqi_dict["line_noise_ratio"],
            "threshold": thresholds["line_noise_ratio"],
        })
    
    # Check for clipping
    if sqi_dict["clip_pct"] > thresholds["clip_pct"]:
        events.append({
            "timestamp": timestamp,
            "channel": channel,
            "event_type": "clipping",
            "severity": "high" if sqi_dict["clip_pct"] > 0.1 else "medium",
            "value": sqi_dict["clip_pct"],
            "threshold": thresholds["clip_pct"],
        })
    
    # Check for flatline
    if sqi_dict["flatline_ratio"] > thresholds["flatline_ratio"]:
        events.append({
            "timestamp": timestamp,
            "channel": channel,
            "event_type": "flatline",
            "severity": "high" if sqi_dict["flatline_ratio"] > 0.95 else "medium",
            "value": sqi_dict["flatline_ratio"],
            "threshold": thresholds["flatline_ratio"],
        })
    
    # Check for excessive high frequency content
    if sqi_dict["hf_lf_ratio"] > thresholds["hf_lf_ratio"]:
        events.append({
            "timestamp": timestamp,
            "channel": channel,
            "event_type": "high_frequency_noise",
            "severity": "high" if sqi_dict["hf_lf_ratio"] > 20.0 else "medium",
            "value": sqi_dict["hf_lf_ratio"],
            "threshold": thresholds["hf_lf_ratio"],
        })
    
    return events
