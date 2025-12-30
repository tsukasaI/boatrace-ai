# Boat Race AI Prediction System

An AI system for predicting exacta/trifecta outcomes in Japanese boat racing (Kyotei).
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
│   ├── processed/            # Processed data (CSV)
│   └── odds/                 # Scraped real-time odds (JSON)
├── src/
│   ├── data_collection/      # Data collection
│   │   ├── downloader.py     # Download historical data
│   │   ├── extractor.py      # LZH extraction
│   │   ├── odds_scraper.py   # Real-time odds scraper
│   │   └── collect_daily.py  # Daily odds collection
│   ├── preprocessing/        # Preprocessing
│   │   └── parser.py         # Parser
│   ├── models/               # Prediction models
│   ├── backtesting/          # Backtesting simulator
│   └── cli/                  # CLI prediction tool
├── rust-api/                 # Rust inference API
├── models/                   # Saved models (pkl, onnx)
├── tests/                    # Test suite
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

# Use real scraped odds (with fallback to payout CSV)
uv run python -m src.backtesting.simulator --use-real-odds

# Use synthetic odds (for testing without data leakage)
uv run python -m src.backtesting.simulator --synthetic-odds
```

### Phase 4: Rust API

```bash
cd rust-api && cargo run
# Endpoints: GET /health, POST /predict, POST /predict/exacta
```

### Phase 5: Odds Collection

```bash
# Scrape single race exacta odds
uv run python -m src.data_collection.odds_scraper -d 20251230 -s 23 -r 1

# Scrape trifecta odds
uv run python -m src.data_collection.odds_scraper -d 20251230 -s 23 -r 1 --trifecta

# Collect all stadiums for a date
uv run python -m src.data_collection.collect_daily --date 20251230

# Collect trifecta odds for all stadiums
uv run python -m src.data_collection.collect_daily --date 20251230 --trifecta

# List stadium codes
uv run python -m src.data_collection.odds_scraper --list-stadiums
```

## Data Sources

- Race Results: https://www1.mbrace.or.jp/od2/K/
- Race Programs: https://www1.mbrace.or.jp/od2/B/
- Official Data: https://www.boatrace.jp/owpc/pc/extra/data/download.html

## Development Phases

- [x] Phase 1: Data Collection & Exploration
- [x] Phase 2: Model Building
- [x] Phase 3: Backtesting
- [x] Phase 4: Inference API (Rust)
- [x] Phase 5: Real-time Odds Scraping

## Supported Bet Types

| Type | Japanese | Combinations | Description |
|------|----------|--------------|-------------|
| Exacta | 2連単 | 30 | 1st & 2nd in order |
| Trifecta | 3連単 | 120 | 1st, 2nd & 3rd in order |

## Strategy

**Expected Value Based**
```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

## Future Extensions

- Quinella (unordered exacta) support
- Web dashboard for daily predictions
- Deep learning models (Transformer)
- Docker containerization
