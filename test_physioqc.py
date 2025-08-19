#!/usr/bin/env python3
"""Simple test script for PhysioQC."""

import numpy as np
import sys
import os

# Import from installed package

def test_sqi():
    """Test SQI computation."""
    print("Testing SQI computation...")
    
    from physioqc.sqi import compute_all_sqis
    
    # Create a simple test signal
    fs = 250
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    
    # Compute SQIs
    sqis = compute_all_sqis(signal, fs, line_freq=60)
    
    print("SQI Results:")
    for key, value in sqis.items():
        print(f"  {key}: {value:.6f}")
    
    return True

def test_simulate():
    """Test signal simulation."""
    print("\nTesting signal simulation...")
    
    from physioqc.simulate import run
    
    # Generate a test signal
    run("test_signal.npy", fs=250, secs=2, n_ch=2, line_freq=60, seed=42)
    
    # Check if file was created
    if os.path.exists("test_signal.npy"):
        signal = np.load("test_signal.npy")
        print(f"Generated signal shape: {signal.shape}")
        os.remove("test_signal.npy")  # Clean up
        return True
    else:
        print("Error: Signal file not created")
        return False

def test_ingest():
    """Test signal ingestion."""
    print("\nTesting signal ingestion...")
    
    from physioqc.simulate import run as simulate_run
    from physioqc.ingest import run as ingest_run
    
    # First create a test signal
    simulate_run("test_ingest.npy", fs=250, secs=3, n_ch=2, line_freq=60, seed=42)
    
    # Then process it
    ingest_run("test_ingest.npy", fs=250, win=0.5, hop=0.25, line_freq=60, out_dir="test_run")
    
    # Check if output files were created
    expected_files = ["sqi.parquet", "events.jsonl", "provenance.json"]
    all_exist = True
    
    for file in expected_files:
        filepath = os.path.join("test_run", file)
        if os.path.exists(filepath):
            print(f"  ✓ {file}")
        else:
            print(f"  X {file} (missing)")
            all_exist = False
    
    # Clean up
    if os.path.exists("test_ingest.npy"):
        os.remove("test_ingest.npy")
    if os.path.exists("test_run"):
        import shutil
        shutil.rmtree("test_run")
    
    return all_exist

def main():
    """Run all tests."""
    print("PhysioQC Test Suite")
    print("=" * 50)
    
    tests = [
        ("SQI Computation", test_sqi),
        ("Signal Simulation", test_simulate),
        ("Signal Ingestion", test_ingest),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"X {test_name} FAILED")
        except Exception as e:
            print(f"X {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! PhysioQC is working correctly.")
        return 0
    else:
        print("X Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
