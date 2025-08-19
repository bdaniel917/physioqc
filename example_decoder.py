#!/usr/bin/env python3
"""
Example decoder for testing PhysioQC decoder integration.

This decoder simulates a simple classifier that:
1. Computes RMS of input data
2. Classifies based on RMS threshold
3. Returns prediction with confidence
"""

import numpy as np
import sys
import time

def initialize():
    """Initialize the decoder."""
    print("Example decoder initialized", file=sys.stderr)

def process(data, timestamp):
    """Process data and return prediction."""
    # Compute RMS of the data
    rms = np.sqrt(np.mean(data**2))
    
    # Simple classification based on RMS
    if rms < 0.05:
        prediction = "low_activity"
        confidence = 0.8
    elif rms < 0.1:
        prediction = "medium_activity"
        confidence = 0.9
    else:
        prediction = "high_activity"
        confidence = 0.7
    
    return {
        "timestamp": timestamp,
        "prediction": prediction,
        "confidence": confidence,
        "rms": float(rms),
        "raw_output": f"{prediction},{confidence:.3f}"
    }

if __name__ == "__main__":
    # Command-line interface for subprocess mode
    print("Example decoder started", file=sys.stderr)
    
    while True:
        try:
            line = input().strip()
            if not line:
                continue
                
            # Parse input: timestamp,data1,data2,...
            parts = line.split(',')
            timestamp = int(parts[0])
            data = np.array([float(x) for x in parts[1:]]).reshape(-1, 1)
            
            # Process and output result
            result = process(data, timestamp)
            print(f"{result['prediction']},{result['confidence']:.3f}")
            
        except EOFError:
            break
        except Exception as e:
            print(f"error,0.0", file=sys.stderr)
            print(f"error,0.0")
