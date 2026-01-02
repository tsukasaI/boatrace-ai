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
└── rust-api/                    # Rust (all operations)
    ├── src/
    │   ├── main.rs              # API server
    │   ├── bin/cli.rs           # CLI binary
    │   ├── predictor.rs         # ONNX inference
    │   ├── core/kelly.rs        # Kelly criterion
    │   ├── data/                # CSV & odds loading
    │   ├── scraper/             # Odds scraping
    │   └── backtesting/         # Backtest simulator
    └── Cargo.toml
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

# With real odds (recommended: use --max-odds or --by-prob)
boat backtest --all-data --model-dir ../models/onnx --max-odds 30

# Probability-based betting (best for real odds)
boat backtest --all-data --model-dir ../models/onnx --by-prob

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

## Betting Strategy

### Synthetic vs Real Odds

The model behaves differently with synthetic odds (generated from probabilities) vs real market odds:

| Strategy | Synthetic Odds | Real Odds |
|----------|---------------|-----------|
| EV (default) | +14.9% ROI | -96.4% ROI |
| EV + max-odds 30 | N/A | +35.8% ROI |
| Probability (--by-prob) | N/A | +42.4% ROI |

**Why EV fails with real odds**: The model exhibits "favorite-longshot bias" - it overestimates probabilities for longshots (high odds). Real odds already reflect market efficiency, so high-EV bets tend to be on extreme longshots that rarely win.

### Recommended Settings

**For backtesting with synthetic odds:**
```bash
boat backtest --all-data --model-dir ../models/onnx --synthetic-odds
```

**For backtesting/betting with real odds:**
```bash
# Option 1: Probability-based (highest total profit)
boat backtest --all-data --model-dir ../models/onnx --by-prob

# Option 2: EV with odds cap (more selective)
boat backtest --all-data --model-dir ../models/onnx --max-odds 30
```

### Strategy Comparison (Real Odds)

| Strategy | Bets | Wins | Hit Rate | ROI | Avg Odds |
|----------|------|------|----------|-----|----------|
| EV (no cap) | 2,448 | 1 | 0.04% | -96.4% | 461.9 |
| EV + max-odds 30 | 453 | 24 | 5.3% | +35.8% | 26.3 |
| Probability | 2,448 | 467 | 19.1% | +42.4% | 10.7 |

## Backtest Results (Rust + ONNX + Synthetic Odds)

| Metric | Value |
|--------|-------|
| Total bets | 109,647 |
| Winning bets | 813 |
| ROI | **+0.8%** |
| Hit rate | 0.7% |
| Profit factor | 1.01 |

## Known Issues & Limitations

### Favorite-Longshot Bias

The model exhibits probability overestimation for high-odds (longshot) combinations:
- Real market odds already incorporate efficient pricing
- EV > 1.0 bets tend to be on extreme longshots that rarely win
- This is why EV strategy fails with real odds (-96.4% ROI)

**Workarounds:**
- Use `--by-prob` for probability-based betting (+42.4% ROI)
- Use `--max-odds 30` to filter extreme longshots (+35.8% ROI)

### ~~Regression vs Classification Mismatch~~ (Fixed)

~~Current training uses MSE regression (`objective: "regression"`) but position prediction is fundamentally a classification/ranking problem.~~ **Fixed**: Now uses `objective: "binary"` with `metric: "binary_logloss"` for proper probability optimization.

### ~~Probability Calibration Gap~~ (Fixed)

~~Platt scaling calibrators are trained in Python but NOT exported to ONNX.~~ **Fixed**: Calibrator coefficients are now exported to `metadata.json` and applied in Rust inference before softmax normalization.

## Future Improvements

### High Priority (ROI Impact)

| ID | Improvement | File | Impact | Status |
|----|-------------|------|--------|--------|
| H1 | Change objective from `regression` to `binary` | train.py:49 | Better probability estimates | ✅ Done |
| H2 | Export Platt scaling to ONNX metadata | export_onnx.py | Calibrated predictions in Rust | ✅ Done |
| H3 | Joint position model (Plackett-Luce) | train.py | Enforce one-boat-per-position | |
| H4 | Add weather features | features.py, parser.py | Weather significantly affects outcomes | |

### Medium Priority (Quality of Life)

| ID | Improvement | Impact |
|----|-------------|--------|
| M1 | Add `--output-format json\|csv` | Enable automation |
| M2 | Add date validation | Prevent invalid date errors |
| M3 | Per-stadium course advantage | Stadium-specific predictions |
| M4 | Use `tabled` crate | Better table formatting |
| M5 | TOML config file | Persistent settings |

### Low Priority

| ID | Improvement |
|----|-------------|
| L1 | Expand interactive mode |
| L2 | ASCII charts for trends |
| L3 | Terminal width detection |

## Troubleshooting

### Common Issues

**"No entries found for this race"**
- Check data directory path (`--data-dir`)
- Verify CSV files exist in data/processed/

**ONNX model loading fails**
- Ensure models exist in models/onnx/
- Run `uv run python src/models/export_onnx.py --verify`

**Backtest shows -96% ROI with EV strategy**
- This is expected with real odds (favorite-longshot bias)
- Use `--by-prob` or `--max-odds 30` instead

**Scraping fails**
- Check network connectivity
- Respect 2+ second request interval
- boatrace.jp may be temporarily unavailable

**Feature mismatch error**
- Ensure Python and Rust feature counts match (43 features)
- Re-export ONNX after training: `uv run python src/models/export_onnx.py --verify`

## API Reference

### Start Server
```bash
cargo run --features api
```

### Endpoints

**POST /api/predict**
```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{"date": 20240115, "stadium": 23, "race": 1}'
```

**Response**
```json
{
  "predictions": [
    {"boat_no": 1, "win_prob": 0.45, "in2_prob": 0.62},
    {"boat_no": 2, "win_prob": 0.18, "in2_prob": 0.35}
  ],
  "exacta": [
    {"combination": "1-2", "probability": 0.18, "ev": 1.24}
  ]
}
```

## Development Guide

### Architecture

**Data Flow:**
1. Raw TXT (CP932) → Python parser → CSV
2. CSV → Rust loader → Feature extraction (43 features)
3. Features → ONNX model → Position probabilities
4. Probabilities + Odds → Expected Value → Betting decision

**Feature Categories (43 total):**
- Stadium code (1)
- Base features (10): national/local win rates, age, weight, class, branch, motor/boat rates
- Historical features (16): recent performance, course-specific stats, start timing
- Relative features (5): rankings within race
- Exhibition features (3): time, rank, diff from average
- Context features (2): race_grade, is_final
- Interaction features (6): class×course, motor×exhibition, equipment scores

### Adding New Features

1. Add extraction in `src/models/features.py`
2. Update `get_feature_columns()` list
3. Retrain: `uv run python src/models/train.py --historical`
4. Export: `uv run python src/models/export_onnx.py --verify`
5. Update Rust `predictor.rs`:
   - Update `NUM_FEATURES` constant
   - Add feature extraction in `extract_features_full()`

### Testing

```bash
# Python tests
uv run pytest tests/

# Rust tests
cd rust-api && cargo test

# Verify feature parity
uv run python scripts/compare_features.py
```

## Data Format Specification

### Raw Data Format
- **Encoding**: CP932 (Shift-JIS)
- **Format**: Fixed-width text
- **Programs**: B*.TXT files (race entries)
- **Results**: K*.TXT files (race outcomes)

### Processed CSV Schema

**programs.csv**
| Column | Type | Description |
|--------|------|-------------|
| date | int | YYYYMMDD format |
| stadium_code | int | 1-24 (see Stadium Codes) |
| race_no | int | 1-12 |
| boat_no | int | 1-6 (lane number) |
| racer_id | int | 4-digit racer ID |
| racer_name | str | Racer name (Japanese) |
| racer_class | str | A1/A2/B1/B2 |
| national_win_rate | float | Win rate (%) |
| national_in2_rate | float | Top-2 rate (%) |
| local_win_rate | float | Stadium-specific win rate |
| local_in2_rate | float | Stadium-specific top-2 rate |
| motor_no | int | Motor number |
| motor_in2_rate | float | Motor top-2 rate (%) |
| boat_in2_rate | float | Boat top-2 rate (%) |

**results.csv**
| Column | Type | Description |
|--------|------|-------------|
| date | int | YYYYMMDD |
| stadium_code | int | 1-24 |
| race_no | int | 1-12 |
| boat_no | int | 1-6 |
| rank | int | Finishing position (1-6) |
| course | int | Actual start course |
| start_timing | float | Start timing (seconds) |

### Odds JSON Schema

```json
{
  "date": 20240115,
  "stadium": 23,
  "race": 1,
  "exacta": {
    "1-2": 5.6,
    "1-3": 12.4,
    "2-1": 8.2
  },
  "trifecta": {
    "1-2-3": 15.2,
    "1-2-4": 28.5
  }
}
```

## References

- Official data: https://www.boatrace.jp/owpc/pc/extra/data/download.html
- Results index: https://www1.mbrace.or.jp/od2/K/dindex.html
- Programs index: https://www1.mbrace.or.jp/od2/B/dindex.html
