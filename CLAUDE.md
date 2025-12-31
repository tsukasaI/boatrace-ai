# Claude Code Instructions

## Project Overview

This is a Japanese boat racing (Kyotei) AI prediction system. The goal is to predict exacta (2-consecutive) race outcomes and maximize ROI using expected value-based betting strategy.

## Tech Stack & Responsibility Split

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Training** | Python + LightGBM | Model training, hyperparameter tuning, ONNX export |
| **Operations** | Rust + ONNX Runtime | CLI, API, prediction, backtesting, odds scraping |

### Python (Training Only)
- `src/data_collection/` - Download & extract raw data files
- `src/preprocessing/` - Parse raw TXT to CSV (initial data prep)
- `src/models/` - Train LightGBM, export to ONNX

### Rust (All Operations)
- `rust-api/` - CLI, REST API, ONNX inference, backtesting, scraping

**Rule: After initial setup, use Rust CLI for all daily operations.**

## Project Structure

```
boatrace-ai/
├── config/settings.py           # Configuration (dates, URLs, stadium codes)
├── data/
│   ├── raw/                     # Raw data (LZH -> TXT files)
│   ├── processed/               # Processed CSV files
│   └── odds/                    # Scraped real-time odds (JSON)
├── models/
│   ├── boatrace_model.pkl       # LightGBM model (Python)
│   └── onnx/                    # ONNX models (for Rust)
│       ├── position_1-6.onnx    # 6 position prediction models
│       └── metadata.json        # Feature names & model info
├── src/                         # Python (training only)
│   ├── data_collection/
│   │   ├── downloader.py        # Download LZH files
│   │   └── extractor.py         # Extract LZH -> TXT
│   ├── preprocessing/
│   │   └── parser.py            # Parse TXT -> CSV
│   └── models/
│       ├── features.py          # Feature engineering
│       ├── dataset.py           # Dataset builder
│       ├── train.py             # LightGBM training
│       ├── evaluate.py          # Model evaluation
│       └── export_onnx.py       # Export to ONNX
├── rust-api/                    # Rust (all operations)
│   ├── src/
│   │   ├── main.rs              # API server
│   │   ├── bin/cli.rs           # CLI binary
│   │   ├── predictor.rs         # ONNX inference
│   │   ├── core/kelly.rs        # Kelly criterion
│   │   ├── data/                # CSV & odds loading
│   │   ├── scraper/             # Odds scraping
│   │   └── backtesting/         # Backtest simulator
│   └── Cargo.toml
└── notebooks/                   # Jupyter exploration
```

## Quick Start

### Initial Setup (One-time)

```bash
# 1. Download & parse data (Python)
cd boatrace-ai
uv run python src/data_collection/downloader.py
uv run python src/data_collection/extractor.py
uv run python src/preprocessing/parser.py

# 2. Train model & export ONNX (Python)
uv run python src/models/train.py --historical
uv run python src/models/export_onnx.py --verify

# 3. Build Rust CLI
cd rust-api
cargo build --release --features full
```

### Daily Operations (Rust CLI)

```bash
cd rust-api

# Today's predictions (auto-scrape odds + predict)
./target/release/boatrace-cli today

# Specific stadium
./target/release/boatrace-cli today -s 23,12

# With trifecta
./target/release/boatrace-cli today --trifecta

# High EV only
./target/release/boatrace-cli today --threshold 1.1
```

## Rust CLI Commands

```bash
cd rust-api

# Build
cargo build --release --features full

# Alias (add to ~/.bashrc or ~/.zshrc)
alias boat='./target/release/boatrace-cli'
```

### Predict Today's Races
```bash
# All stadiums, active races
boat today

# Specific stadiums (23=Karatsu, 12=Suminoe)
boat today -s 23,12

# Include trifecta, EV > 1.1
boat today --trifecta --threshold 1.1

# Skip scraping (use cached odds)
boat today --no-scrape
```

### Predict Specific Race
```bash
boat predict -d 20240115 -s 23 -r 1

# With betting recommendations
boat predict -d 20240115 -s 23 -r 1 --bankroll 50000 --kelly 0.25
```

### List Races
```bash
boat list -d 20240115
```

### Backtest
```bash
# With ONNX model + synthetic odds
boat backtest --all-data --model-dir ../models/onnx --synthetic-odds

# Custom EV threshold
boat backtest --all-data --threshold 1.1
```

### Scrape Odds
```bash
# Single race
boat scrape -d 20240115 -s 23 -r 1

# All races at stadium
boat scrape -d 20240115 -s 23

# Trifecta odds
boat scrape -d 20240115 -s 23 --trifecta
```

### Parse Raw Data
```bash
boat parse -i ../data/raw/programs -o ../data/processed -t programs
boat parse -i ../data/raw/results -o ../data/processed -t results
boat parse -i ../data/raw/results -o ../data/processed -t payouts
```

## Stadium Codes

| Code | Stadium | Code | Stadium |
|------|---------|------|---------|
| 1 | Kiryu | 13 | Amagasaki |
| 2 | Toda | 14 | Naruto |
| 3 | Edogawa | 15 | Marugame |
| 4 | Heiwajima | 16 | Kojima |
| 5 | Tamagawa | 17 | Miyajima |
| 6 | Hamanako | 18 | Tokuyama |
| 7 | Gamagori | 19 | Shimonoseki |
| 8 | Tokoname | 20 | Wakamatsu |
| 9 | Tsu | 21 | Ashiya |
| 10 | Mikuni | 22 | Fukuoka |
| 11 | Biwako | 23 | Karatsu |
| 12 | Suminoe | 24 | Omura |

## Python Commands (Training Only)

### Download Data
```bash
uv run python src/data_collection/downloader.py
uv run python src/data_collection/extractor.py
```

### Parse Data
```bash
uv run python src/preprocessing/parser.py
```

### Train Model
```bash
# Basic training
uv run python src/models/train.py --historical

# With hyperparameter optimization
uv run python src/models/train.py --historical --optimize --n-trials 50

# Evaluate
uv run python src/models/evaluate.py --historical

# Export to ONNX
uv run python src/models/export_onnx.py --verify --compare
```

## Key Concepts

### Expected Value Strategy
```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

### Bet Types
- **Exacta (2連単)**: 1st + 2nd in order (30 combinations)
- **Trifecta (3連単)**: 1st + 2nd + 3rd in order (120 combinations)

### Model Output
```python
# 6 boats × 6 positions probability matrix
position_probs[boat_idx, position_idx]  # P(boat finishes in position)

# Exacta probability
P(boat_i=1st, boat_j=2nd) ≈ P(boat_i=1st) × P(boat_j=2nd) / (1 - P(boat_j=1st))
```

## Important Notes

1. **Request Interval**: 2+ seconds between requests
2. **Encoding**: Raw files use CP932 (Shift-JIS)
3. **Data Split**: 2023 train / 2024-H1 val / 2024-H2 test
4. **Historical Odds**: boatrace.jp only keeps ~1 week of odds

## Backtest Results (Rust + ONNX + Synthetic Odds)

| Metric | Value |
|--------|-------|
| Total bets | 109,647 |
| Winning bets | 813 |
| ROI | **+0.8%** |
| Hit rate | 0.7% |
| Profit factor | 1.01 |

## References

- Official data: https://www.boatrace.jp/owpc/pc/extra/data/download.html
- Results index: https://www1.mbrace.or.jp/od2/K/dindex.html
- Programs index: https://www1.mbrace.or.jp/od2/B/dindex.html
