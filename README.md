# Boat Race AI Prediction System

An AI system for predicting exacta (2-consecutive) outcomes in Japanese boat racing (Kyotei).
Aims to improve ROI using an expected value-based betting strategy.

## Project Structure

```
boatrace-ai/
├── config/
│   └── settings.py          # Configuration file
├── data/
│   ├── raw/                  # Raw data (LZH, TXT)
│   │   ├── results/          # Race results
│   │   └── programs/         # Race programs
│   └── processed/            # Processed data (CSV)
├── src/
│   ├── data_collection/      # Data collection
│   │   ├── downloader.py     # Downloader
│   │   └── extractor.py      # LZH extraction
│   ├── preprocessing/        # Preprocessing
│   │   └── parser.py         # Parser
│   ├── models/               # Prediction models (Phase 2)
│   ├── backtesting/          # Backtesting (Phase 3)
│   └── api/                  # Inference API (deprecated)
├── rust-api/                 # Rust inference API (Phase 4)
├── notebooks/                # Jupyter exploration
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Phase 1: Data Collection

```bash
# 1. Download data (2023-2024, about 2 years)
uv run python src/data_collection/downloader.py

# 2. Extract LZH files
uv run python src/data_collection/extractor.py

# 3. Convert to CSV
uv run python src/preprocessing/parser.py
```

### Phase 2: Model Training

```bash
# Train with historical features
uv run python src/models/train.py --historical

# Export to ONNX
uv run python src/models/export_onnx.py --verify
```

### Phase 3: Backtesting

```bash
# Run backtest with EV > 1.0 strategy
uv run python -m src.backtesting.simulator
```

### Phase 4: Rust API

```bash
cd rust-api && cargo run
# Endpoints: GET /health, POST /predict, POST /predict/exacta
```

## Data Sources

- Race Results: https://www1.mbrace.or.jp/od2/K/
- Race Programs: https://www1.mbrace.or.jp/od2/B/
- Official Data: https://www.boatrace.jp/owpc/pc/extra/data/download.html

## Development Phases

- [x] Phase 1: Data Collection & Exploration
- [x] Phase 2: Model Building
- [x] Phase 3: Backtesting
- [x] Phase 4: Inference API

## Strategy

**Expected Value Based**
```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

## Future Extensions

- Additional bet types (trifecta, trio)
- Target profit calculation
- Kelly criterion bet sizing optimization
