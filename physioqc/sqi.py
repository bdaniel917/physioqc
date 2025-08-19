"""Signal Quality Indices (SQIs) for physiological signal analysis."""

import numpy as np
from scipy.signal import welch
from typing import Tuple, Optional


def rms(x: np.ndarray) -> float:
    """Calculate Root Mean Square amplitude.
    
    Args:
        x: Input signal array
        
    Returns:
        RMS value as float
    """
    return float(np.sqrt(np.mean(x**2)))


def dc_offset(x: np.ndarray) -> float:
    """Calculate DC offset (mean signal level).
    
    Args:
        x: Input signal array
        
    Returns:
        DC offset as float
    """
    return float(np.mean(x))


def line_noise_ratio(x: np.ndarray, fs: int, line_freq: int = 60, bw: int = 1) -> float:
    """Calculate line noise ratio at specified frequency.
    
    Args:
        x: Input signal array
        fs: Sampling frequency in Hz
        line_freq: Line frequency (50 or 60 Hz)
        bw: Bandwidth around line frequency in Hz
        
    Returns:
        Line noise ratio as float
    """
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    total_power = np.trapz(Pxx, f)
    mask = (f >= line_freq - bw) & (f <= line_freq + bw)
    line_power = np.trapz(Pxx[mask], f[mask])
    return float(line_power / (total_power + 1e-12))


def hf_lf_ratio(x: np.ndarray, fs: int, lf: Tuple[int, int] = (1, 30), hf: Tuple[int, int] = (30, 90)) -> float:
    """Calculate high frequency to low frequency power ratio.
    
    Args:
        x: Input signal array
        fs: Sampling frequency in Hz
        lf: Low frequency band (min, max) in Hz
        hf: High frequency band (min, max) in Hz
        
    Returns:
        HF/LF ratio as float
    """
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    
    def band_power(band: Tuple[int, int]) -> float:
        mask = (f >= band[0]) & (f <= band[1])
        return np.trapz(Pxx[mask], f[mask])
    
    lf_pow, hf_pow = band_power(lf), band_power(hf)
    return float(hf_pow / (lf_pow + 1e-12))


def clip_pct(x: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None, eps: float = 1e-6) -> float:
    """Calculate percentage of clipped samples.
    
    Args:
        x: Input signal array
        min_val: Minimum value threshold (if None, uses signal min)
        max_val: Maximum value threshold (if None, uses signal max)
        eps: Epsilon for clipping detection
        
    Returns:
        Clipping percentage as float
    """
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    
    clipped = (np.abs(x - min_val) < eps) | (np.abs(x - max_val) < eps)
    return float(np.mean(clipped))


def flatline_ratio(x: np.ndarray, eps: float = 1e-3) -> float:
    """Calculate percentage of flat/constant samples.
    
    Args:
        x: Input signal array
        eps: Epsilon for flatline detection
        
    Returns:
        Flatline ratio as float
    """
    median_val = np.median(x)
    flat_samples = np.abs(x - median_val) < eps
    return float(np.mean(flat_samples))


def compute_all_sqis(x: np.ndarray, fs: int, line_freq: int = 60) -> dict:
    """Compute all SQIs for a given signal.
    
    Args:
        x: Input signal array
        fs: Sampling frequency in Hz
        line_freq: Line frequency for noise detection
        
    Returns:
        Dictionary containing all SQI values
    """
    return {
        "rms": rms(x),
        "dc_offset": dc_offset(x),
        "line_noise_ratio": line_noise_ratio(x, fs, line_freq),
        "hf_lf_ratio": hf_lf_ratio(x, fs),
        "clip_pct": clip_pct(x),
        "flatline_ratio": flatline_ratio(x),
    }
