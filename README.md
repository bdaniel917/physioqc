# PhysioQC

**Headless Linux/macOS CLI for real‑time Signal Quality + auto‑event logging on physiological signals.**

> **Tagline (snappy):**
> **Stream in → SQIs & auto‑events out.** Headless CLI. Parquet/JSONL with provenance + keyboard annotations.

---

## Why

Stop wasting sessions. PhysioQC surfaces **Signal Quality Indices (SQIs)** and **auto‑events** while you record or replay, so you can fix issues *now* and ship clean, analysis‑ready artifacts later.

---

## Installation

```bash
# Install from source
git clone https://github.com/your-org/physioqc.git
cd physioqc
pip install -e .

# Or install directly (when published)
pip install physioqc
```

---

## 60‑Second Demo

```bash
# 1) Create a demo signal
physioq simulate --out demo.npy --fs 250 --secs 10 --n-ch 4 --line-freq 60 --seed 7

# 2) Run QC on it
physioq ingest file --path demo.npy --fs 250 --win 1s --hop 0.25s --line-freq 60 --out-dir runs/demo1

# 3) Annotate while replaying (optional)
physioq annotate --session runs/demo1 --keys "b:burst n:noise m:motion x:artifact"

# 4) Inspect
ls runs/demo1
# -> sqi.parquet  events.jsonl  provenance.json
```

---

## Features

### Signal Quality Indices (SQIs)
- **RMS** - Root Mean Square amplitude
- **DC Offset** - Mean signal level
- **Line Noise Ratio** - Power at line frequency (50/60Hz)
- **HF/LF Ratio** - High frequency to low frequency power ratio
- **Clipping %** - Percentage of clipped samples
- **Flatline Ratio** - Percentage of flat/constant samples

### Auto-Event Detection
- Automatic detection of signal artifacts
- Real-time quality monitoring
- Configurable thresholds

### Output Formats
- **Parquet** - Efficient SQI storage with pandas
- **JSONL** - Human-readable event logs
- **Provenance** - Complete processing metadata

---

## CLI Commands

### `physioq simulate`
Generate synthetic physiological signals for testing.

```bash
physioq simulate --out demo.npy --fs 250 --secs 10 --n-ch 4 --line-freq 60 --seed 7
```

### `physioq ingest`
Process signals and compute SQIs in real-time.

```bash
physioq ingest file --path demo.npy --fs 250 --win 1s --hop 0.25s --line-freq 60 --out-dir runs/demo1
```

### `physioq annotate`
Add keyboard annotations during signal replay.

```bash
physioq annotate --session runs/demo1 --keys "b:burst n:noise m:motion x:artifact"
```

---

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black physioqc/
isort physioqc/

# Lint code
flake8 physioqc/
```

---

## License

MIT License - see LICENSE file for details.
