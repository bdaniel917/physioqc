# PhysioQC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/PyPI-physioqc-green.svg)](https://pypi.org/project/physioqc/)

**Professional-grade CLI for real-time Signal Quality Assessment (SQI) and auto-event logging on physiological signals.**

> Stream in → SQIs & auto‑events out

## Overview

PhysioQC is a comprehensive tool designed for researchers, clinicians, and engineers working with neural data. It provides real-time signal quality assessment, automated event detection, decoder integration, and professional visualization capabilities.

### Key Features

- **Real-time SQI computation** with composite quality scoring (0-100)
- **Multi-source data support** (files, LSL streams, simulated data, BCI datasets)
- **External decoder integration** with performance metrics
- **Live web dashboard** with real-time visualization
- **Interactive annotation system** with hotkey support
- **Signal validation tools** for robustness testing
- **Export capabilities** for MATLAB and Python analysis
- **Professional CLI interface** with comprehensive help

## Quick Start

### Installation

```bash
pip install physioqc
```

### Basic Usage

```bash
# Generate demo data
physioq simulate --out demo.npy

# Run quality assessment
physioq ingest --source "file:demo.npy" --out-dir runs/demo

# View live status
physioq status runs/demo

# Launch web dashboard
physioq dashboard --session runs/demo
```

## Core Capabilities

### Signal Quality Assessment

PhysioQC computes comprehensive signal quality metrics:

- **RMS**: Root Mean Square amplitude
- **Line Noise Ratio**: Power at mains frequency (50/60Hz)
- **DC Offset**: Mean signal level
- **HF/LF Ratio**: High-frequency to low-frequency power ratio
- **Clipping Percentage**: Signal saturation detection
- **Flatline Ratio**: Signal variation assessment
- **Dropout Count**: Data loss detection
- **Composite SQI**: Weighted quality score (0-100)

### Multi-Source Data Support

```bash
# File-based data
physioq ingest --source "file:data.npy" --out-dir runs/file_analysis

# LSL streams (OpenBCI, g.tec, etc.)
physioq ingest --source "lsl:OpenBCI" --duration 30 --out-dir runs/lsl_analysis

# Simulated data with parameters
physioq ingest --source "simulated:fs=500,secs=10,n_ch=16" --out-dir runs/sim_analysis

# BCI Competition datasets
physioq ingest --source "bci_competition:bci_iv_2a" --out-dir runs/bci_analysis
```

### Decoder Integration

```bash
# With Python decoder
physioq ingest --source "file:data.npy" --decode "python:my_decoder" --out-dir runs/decoded

# With subprocess decoder
physioq ingest --source "file:data.npy" --decode "subprocess:/path/to/decoder" --out-dir runs/decoded

# View decoder performance
physioq decode-status runs/decoded
```

### Live Monitoring

```bash
# Real-time status table
physioq status runs/session --top-n 5 --refresh 1

# Web dashboard
physioq dashboard --session runs/session --port 8000
```

### Interactive Annotation

```bash
# Hotkey annotation mode
physioq hotkeys runs/session --keymap "a:artifact,n:noise,s:stim,m:movement"

# Traditional annotation
physioq annotate --session runs/session --user researcher
```

### Signal Validation

```bash
# Test robustness with noise injection
physioq inject-noise --source "simulated:fs=250,secs=5,n_ch=4" \
                     --pattern "line_burst" --channels "0,2" \
                     --out-dir runs/validation_test
```

### Export & Analysis

```bash
# Export for Python analysis
physioq to-npz runs/session --output analysis_data.npz

# Export for MATLAB analysis
physioq to-mat runs/session --output analysis_data.mat

# Run comprehensive benchmarks
physioq benchmark --source "simulated:fs=250,secs=10,n_ch=8" \
                  --decoder "python:my_decoder" \
                  --out-file benchmark_results.json
```

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install from PyPI

```bash
pip install physioqc
```

### Install from Source

```bash
git clone https://github.com/yourusername/physioqc.git
cd physioqc
pip install -e .
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For LSL stream support
pip install pylsl

# For web dashboard
pip install fastapi uvicorn

# For interactive features
pip install readchar

# For rich terminal output
pip install rich
```

## Documentation

### Command Reference

```bash
# Get help for any command
physioq --help
physioq ingest --help
physioq dashboard --help

# List available data sources
physioq sources
physioq sources --list-lsl
physioq sources --list-bci
```

### Configuration

PhysioQC uses sensible defaults but allows customization:

```bash
# Custom SQI thresholds
physioq ingest --source "file:data.npy" \
               --th-lnr 0.5 \
               --th-burst-z 5.0 \
               --th-flat-std 1e-4 \
               --th-clip-pct 0.03

# Custom window parameters
physioq ingest --source "file:data.npy" \
               --win "2s" \
               --hop "500ms" \
               --fs 1000
```

### Output Structure

```
runs/session_name/
├── sqi.parquet          # Signal quality metrics
├── events.jsonl         # Automated and manual events
├── decode.parquet       # Decoder outputs (if applicable)
├── decode_metrics.json  # Performance metrics
├── provenance.json      # Processing metadata
├── export.npz          # NumPy export
└── export.mat          # MATLAB export
```

## Use Cases

### Research Applications

- **BCI Research**: Real-time quality monitoring during experiments
- **Clinical Studies**: Signal validation for medical applications
- **Algorithm Development**: Performance benchmarking
- **Data Collection**: Quality assurance during recording

### Clinical Applications

- **EEG Monitoring**: Real-time quality assessment
- **Neuromodulation**: Signal validation for therapeutic devices
- **Diagnostic Tools**: Quality metrics for clinical decision support
- **Research Protocols**: Standardized assessment across studies

### Engineering Applications

- **Hardware Testing**: Device validation and characterization
- **Algorithm Validation**: Performance testing and comparison
- **System Integration**: Quality monitoring in production systems
- **Development Workflows**: Continuous testing and validation

## Development

### Project Structure

```
physioqc/
├── physioqc/
│   ├── __init__.py
│   ├── cli.py              # Main CLI implementation
│   ├── sqi.py              # Signal quality algorithms
│   ├── simulate.py         # Data simulation
│   ├── ingest.py           # Data ingestion
│   └── annotate.py         # Annotation system
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Example scripts
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=physioqc
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Performance

### Benchmarks

- **Processing Speed**: Real-time capable with sub-second processing
- **Scalability**: Handles 100+ channels efficiently
- **Memory Usage**: Minimal footprint with streaming processing
- **Accuracy**: Validated against peer-reviewed SQI algorithms

### System Requirements

- **CPU**: Multi-core recommended for real-time processing
- **Memory**: 4GB+ RAM for large datasets
- **Storage**: SSD recommended for high-throughput applications
- **Network**: Required for LSL stream processing


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**PhysioQC** - Making physiological signal quality assessment accessible, reliable, and professional.

*Built for the neuroscience and BCI communities.*
