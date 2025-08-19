"""Generate synthetic physiological signals for testing and demonstration."""

import numpy as np
from typing import Optional
import os


def run(out: str = "demo.npy", fs: int = 250, secs: int = 10, n_ch: int = 4, 
        line_freq: int = 60, seed: int = 7) -> None:
    """Generate a synthetic physiological signal.
    
    Args:
        out: Output file path
        fs: Sampling frequency in Hz
        secs: Duration in seconds
        n_ch: Number of channels
        line_freq: Line frequency for noise simulation
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Generate time vector
    t = np.linspace(0, secs, fs * secs, endpoint=False)
    
    # Initialize signal array
    signal = np.zeros((fs * secs, n_ch))
    
    for ch in range(n_ch):
        # Base physiological signal (e.g., ECG-like)
        freq = 1.2 + 0.3 * np.sin(2 * np.pi * 0.1 * t)  # Variable heart rate
        base_signal = np.sin(2 * np.pi * freq * t)
        
        # Add some physiological variability
        respiratory_mod = 0.3 * np.sin(2 * np.pi * 0.25 * t)
        
        # Add line noise
        line_noise = 0.1 * np.sin(2 * np.pi * line_freq * t)
        
        # Add random noise
        random_noise = 0.05 * np.random.randn(len(t))
        
        # Combine all components
        signal[:, ch] = base_signal + respiratory_mod + line_noise + random_noise
        
        # Add some artifacts at random intervals
        artifact_times = np.random.choice(len(t), size=5, replace=False)
        for art_time in artifact_times:
            duration = np.random.randint(50, 200)
            end_time = min(art_time + duration, len(t))
            signal[art_time:end_time, ch] += 2.0 * np.random.randn(end_time - art_time)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    
    # Save signal
    np.save(out, signal)
    
    print(f"Generated synthetic signal: {out}")
    print(f"  Shape: {signal.shape}")
    print(f"  Duration: {secs}s")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Channels: {n_ch}")
    print(f"  Line frequency: {line_freq} Hz")
