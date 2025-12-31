# Boat Race AI Prediction System

An AI system for predicting exacta/trifecta outcomes in Japanese boat racing (Kyotei).
Aims to improve ROI using an expected value-based betting strategy.

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Training** | Python + LightGBM | Model training, ONNX export |
| **Operations** | Rust + ONNX Runtime | CLI, API, prediction, backtesting |

## Project Structure

```
boatrace-ai/
├── config/settings.py           # Configuration
├── data/
│   ├── raw/                     # Raw data (LZH, TXT)
│   ├── processed/               # Processed CSV files
│   └── odds/                    # Scraped odds (JSON)
├── models/
│   ├── boatrace_model.pkl       # LightGBM model
│   └── onnx/                    # ONNX models for Rust
├── src/                         # Python (training only)
│   ├── data_collection/         # Download & extract data
│   ├── preprocessing/           # Parse raw data
│   └── models/                  # Train & export models
├── rust-api/                    # Rust (all operations)
│   └── src/
│       ├── bin/cli.rs           # CLI binary
│       ├── main.rs              # API server
│       ├── predictor.rs         # ONNX inference
│       ├── backtesting/         # Backtest simulator
│       └── scraper/             # Odds scraping
└── notebooks/                   # Jupyter exploration
```

## Quick Start

### 1. Initial Setup (Python)

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Download & parse data
uv run python src/data_collection/downloader.py
uv run python src/data_collection/extractor.py
uv run python src/preprocessing/parser.py

# Train model & export to ONNX
uv run python src/models/train.py --historical
uv run python src/models/export_onnx.py --verify
```

### 2. Build Rust CLI

```bash
cd rust-api
cargo build --release --features full

# Optional: add alias
echo "alias boat='$(pwd)/target/release/boatrace-cli'" >> ~/.zshrc
```

### 3. Daily Operations (Rust CLI)

```bash
# Predict today's races (auto-scrapes odds)
boat today

# Specific stadiums
boat today -s 23,12

# With trifecta, high EV only
boat today --trifecta --threshold 1.1
```

## Rust CLI Commands

| Command | Description |
|---------|-------------|
| `today` | Today's predictions with live odds |
| `predict` | Predict a specific race |
| `list` | List races for a date |
| `backtest` | Run backtesting simulation |
| `scrape` | Scrape odds from boatrace.jp |
| `parse` | Parse raw data files |

See [rust-api/README.md](rust-api/README.md) for detailed CLI documentation.

## Supported Bet Types

| Type | Japanese | Combinations | Description |
|------|----------|--------------|-------------|
| Exacta | 2連単 | 30 | 1st & 2nd in order |
| Trifecta | 3連単 | 120 | 1st, 2nd & 3rd in order |

## Strategy

```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

## Data Sources

- Official Data: https://www.boatrace.jp/owpc/pc/extra/data/download.html
- Race Results: https://www1.mbrace.or.jp/od2/K/
- Race Programs: https://www1.mbrace.or.jp/od2/B/

## Development Status

- [x] Data Collection & Parsing
- [x] Model Training (LightGBM)
- [x] ONNX Export
- [x] Rust CLI & API
- [x] Backtesting
- [x] Odds Scraping
- [ ] Model Improvements (weather, deep learning)
- [ ] Production Features (notifications, tracking)
