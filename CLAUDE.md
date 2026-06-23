# Claude Code Instructions

## Project Overview

This is a Japanese boat racing (Kyotei) AI prediction system. The goal is to predict exacta (2-consecutive) race outcomes and maximize ROI using expected value-based betting strategy.

> ⚠️ **How to read the ROI numbers in this document (2026-06 audit)**
>
> The headline ROI figures below are **not** validated live-tradeable returns. Two
> structural issues inflate them; treat the numbers as engineering signals, not profit forecasts:
>
> 1. **Synthetic-odds ROI (+178%, +127%, etc.) is circular.** Synthetic odds
>    (`backtesting/synthetic.rs`) are a *fixed course-position prior* (same 30 odds for
>    every race) with a flat 25% takeout — they carry no race-specific market information.
>    A model only has to beat a naive lane-number baseline to show positive ROI. These
>    numbers are valid **only for relative comparisons** (old model vs new model), never as
>    a real-money forecast.
> 2. **Real-odds ROI (+45.7%, +32.5%, etc.) has lookahead bias.** The backtest decides
>    bets using the **final settled pari-mutuel odds**, which are unknown until after betting
>    closes (`odds_loader` ignores `scraped_at`; only one post-close snapshot exists per race).
>    A lookahead-free result requires pre-deadline odds snapshots, which are not yet collected.
> 3. **Train/serve skew.** `exhibition_time` and weather features are trained from the
>    *results* file but the live `today` path can't supply them (falls back to defaults), so
>    backtest accuracy overstates live accuracy.
>
> A defensible "can this actually win?" answer requires the phased work tracked below
> (pre-deadline odds collection → lookahead-free backtest → calibration-first evaluation).

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
│   ├── boatrace_model.pkl       # LightGBM binary classifiers (Python)
│   ├── boatrace_ranker.pkl      # LightGBM LambdaRank model (Python)
│   └── onnx/                    # ONNX models (for Rust)
│       ├── position_1-6.onnx    # 6 position prediction models (binary)
│       ├── ranker.onnx          # Single ranking model (LambdaRank)
│       └── metadata.json        # Feature names, model type & calibrators
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
# NOTE: --model-dir is required from the rust-api dir, else it falls back to the
# heuristic predictor (default is models/onnx relative to CWD = rust-api/models/onnx).
./target/release/boatrace-cli --model-dir ../models/onnx today

# Specific stadium
./target/release/boatrace-cli --model-dir ../models/onnx today -s 23,12

# With trifecta
./target/release/boatrace-cli --model-dir ../models/onnx today --trifecta

# High EV only
./target/release/boatrace-cli --model-dir ../models/onnx today --threshold 1.1
```

**`predict` and `today` use the trained LambdaRank ONNX model** (the same predictor
as `backtest`), loaded from `--model-dir` (global flag, default `models/onnx`). If the
model can't be loaded, both commands print a warning to stderr and fall back to the
heuristic `FallbackPredictor` (uniform-ish, course-bias-only output). Watch for
`Using LambdaRank predictor ...` vs `Failed to load ... using fallback` on stderr.

- `predict` (a past date) builds the **full 50-feature set** (historical, exhibition,
  weather, race context) from processed CSVs, so its probabilities match `backtest`.
- `today` (a future date) has no results/weather yet, so it uses racer **history +
  base features only**; exhibition/weather/context fall back to model defaults, and
  history is bounded by the latest processed results (currently Dec 2025, so it may be
  stale). Live ROI will therefore differ from the synthetic-odds backtest figures below.

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
# --model-dir and --data-dir are global flags (before the subcommand). From the
# rust-api dir they must point up one level, like backtest.
boat --model-dir ../models/onnx --data-dir ../data/processed predict -d 20240115 -s 23 -r 1

# With betting recommendations
boat --model-dir ../models/onnx --data-dir ../data/processed \
  predict -d 20240115 -s 23 -r 1 --bankroll 50000 --kelly 0.25
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

# Output formats (table, json, csv)
boat backtest --all-data --output-format json -o results.json
boat backtest --all-data --output-format csv -o summary.csv
boat backtest --all-data --output-format csv --detailed -o bets.csv
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
# Binary classification model (6 models, one per position)
uv run python src/models/train.py --historical

# LambdaRank ranking model (single model with Plackett-Luce)
uv run python src/models/train.py --historical --ranking

# With hyperparameter optimization
uv run python src/models/train.py --historical --optimize --n-trials 50

# Evaluate
uv run python src/models/evaluate.py --historical

# Export to ONNX (binary classifiers)
uv run python src/models/export_onnx.py --verify --compare

# Export to ONNX (LambdaRank ranker)
uv run python src/models/export_onnx.py --ranking
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

### Model Types

**LambdaRank Ranker (recommended)**
- Single learning-to-rank model
- Outputs ranking scores, converted to probabilities via Plackett-Luce
- Enforces one-boat-per-position constraint naturally
- Files: `ranker.onnx`
- Training: NDCG@1=0.855, NDCG@2=0.861, NDCG@3=0.889
- **+178% ROI** with synthetic odds (vs +9% for binary)

**Binary Classifier (legacy)**
- 6 independent models, one per finishing position
- Each model predicts P(boat finishes in position k)
- Platt scaling calibration for probability estimates
- Files: `position_1.onnx` ... `position_6.onnx`

### Model Output
```python
# 6 boats × 6 positions probability matrix
position_probs[boat_idx, position_idx]  # P(boat finishes in position)

# Exacta probability
P(boat_i=1st, boat_j=2nd) ≈ P(boat_i=1st) × P(boat_j=2nd) / (1 - P(boat_j=1st))
```

## Dec-2025 Holdout Validation & Walk-Forward Retrain (2026-05)

The production ranker (`ranker.onnx`) had been trained on **2023 only**. With 2024 (full) and
Dec 2025 data now available, Dec 2025 is a genuine forward holdout neither the old nor the new
model trained on. All figures use `--synthetic-odds` (real odds end 2024-12-29) on Dec 2025
(4,608 races). The old 2023 model is preserved at `models/onnx_2023baseline/` for rollback.

**Old (2023) vs New (≤2024-H1) on the Dec-2025 holdout — measured, not estimated:**

| Strategy | Model | Bets | Hit rate | ROI | Profit factor | Avg odds |
|----------|-------|------|----------|-----|---------------|----------|
| by-prob | Old (2023) | 13,731 | 19.93% | +126.5% | 2.58 | 14.6 |
| by-prob | **New (≤2024-H1)** | 13,731 | 19.93% | **+127.4%** | 2.59 | 14.6 |
| EV | Old (2023) | 13,728 | 7.58% | +160.0% | 2.73 | 64.9 |
| EV | **New (≤2024-H1)** | 13,729 | 7.58% | **+163.0%** | 2.76 | 65.2 |

**Conclusion**: The 2023-only model did **not** decay — it held +126% ROI on unseen Dec-2025
data. Adding 2024 to training improved ROI by <1pp (within noise; the by-prob runs differ by a
single winning bet). The model is temporally stable; frequent retraining is not required. New
model val NDCG@1=0.852/@2=0.857/@3=0.885 ≈ baseline (0.855/0.861/0.889).

**Caveats**: (1) Holdout is one month (Dec 2025), not a full year — limited seasonal coverage.
(2) Synthetic odds inflate ROI vs real odds (see Betting Strategy table below); use these
numbers for *relative* old-vs-new comparison, not as a real-odds ROI forecast. (3) Stadium-course
features must stay **off** (50-feature model): a 53-feature retrain early-stopped at iteration 14
with NDCG@1=0.843, reproducing the documented M3 degradation.

## Important Notes

1. **Request Interval**: 2+ seconds between requests
2. **Encoding**: Raw files use CP932 (Shift-JIS)
3. **Data Split** (walk-forward, default since 2026-05): train ≤ 2024-06-30 (2023 + 2024-H1) / val ≤ 2024-12-31 (2024-H2, early stopping) / test > 2024-12-31. Override with `train.py --train-end-date --val-end-date`. Original split was 2023 / 2024-H1 / 2024-H2.
4. **Data coverage**: processed CSVs span 2023-01 → 2024-12 (full) plus **2025-12 only** (Dec 2025, ~4,600 races). There is no Jan–Nov 2025 data; "test" in the current split resolves to Dec 2025.
5. **Historical Odds**: boatrace.jp only keeps ~1 week of odds. Scraped real odds in `data/odds/` end at 2024-12-29, so any backtest on 2025 data must use `--synthetic-odds`.

## Betting Strategy

### LambdaRank vs Binary Classifier (Real Odds)

> ⚠️ The real-odds ROI in this section has **lookahead bias** (bets decided on post-close
> final odds). Use for relative strategy comparison only. See the audit note at the top.

LambdaRank fixes the favorite-longshot bias that plagued the binary classifier:

| Strategy | Binary ROI | LambdaRank ROI | Improvement |
|----------|------------|----------------|-------------|
| EV (default) | -96.4% | **+12.8%** | +109% |
| EV + max-odds 30 | +35.8% | +32.5% | similar |
| Probability (--by-prob) | +42.4% | **+45.7%** | +3% |

### Recommended Settings

**For backtesting with synthetic odds:**
```bash
boat backtest --all-data --model-dir ../models/onnx --synthetic-odds
```

**For backtesting/betting with real odds:**
```bash
# Option 1: Probability-based (best ROI + lowest drawdown)
boat backtest --all-data --model-dir ../models/onnx --by-prob

# Option 2: EV strategy (now works with LambdaRank!)
boat backtest --all-data --model-dir ../models/onnx

# Option 3: EV with odds cap (more selective)
boat backtest --all-data --model-dir ../models/onnx --max-odds 30
```

### LambdaRank Strategy Comparison (Real Odds)

| Strategy | Bets | Wins | Hit Rate | ROI | Max Drawdown |
|----------|------|------|----------|-----|--------------|
| EV (default) | 2,431 | 50 | 2.1% | +12.8% | 30,620 |
| EV + max-odds 30 | 1,933 | 181 | 9.4% | +32.5% | 17,070 |
| **Probability** | 2,448 | 494 | 20.2% | **+45.7%** | 3,952 |

**Probability strategy recommended**: Highest ROI (+45.7%) with lowest drawdown (¥3,952).

## Backtest Results (Rust + ONNX + Synthetic Odds)

> ⚠️ Synthetic-odds ROI is **circular** (model vs a fixed lane-number prior, not vs a real
> market). The +178.4% is a relative model-quality signal, **not** a real-money forecast.
> See the audit note at the top of this document.

### Model Comparison

| Metric | Binary Classifier | LambdaRank |
|--------|-------------------|------------|
| **ROI** | +9.2% | **+178.4%** |
| Total bets | 345,960 | 345,867 |
| Winning bets | 2,859 | 28,245 |
| Hit rate | 0.8% | 8.2% |
| Avg probability | 3.0% | 9.3% |
| Avg odds | 146.3 | 60.6 |
| Profit factor | 1.09 | 2.94 |

**Why LambdaRank performs better:**
- **Better calibration**: Plackett-Luce probabilities (9.3%) closely match actual hit rate (8.2%)
- **Reasonable odds**: Bets on 60x avg odds vs 146x for binary classifier
- **10× higher hit rate**: Model learned ranking relationships, not just independent probabilities

### Training Commands

```bash
# Train LambdaRank (recommended)
uv run python src/models/train.py --historical --ranking
uv run python src/models/export_onnx.py --ranking

# Train Binary classifier
uv run python src/models/train.py --historical
uv run python src/models/export_onnx.py --verify
```

The simulator auto-detects model type from `metadata.json` and uses the appropriate predictor.

## Known Issues & Limitations

### ~~Favorite-Longshot Bias~~ (Fixed with LambdaRank)

~~The binary classifier exhibits probability overestimation for high-odds (longshot) combinations, causing EV strategy to fail with real odds (-96.4% ROI).~~

**Fixed**: LambdaRank model has better probability calibration:
- EV strategy now works with real odds (+12.8% ROI)
- Probability strategy achieves +45.7% ROI
- Plackett-Luce probabilities match actual outcomes

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
| H3 | LambdaRank ranking model | train.py, predictor.rs | Single model with Plackett-Luce probabilities | ✅ Done |
| H4 | Add weather features | features.py, parser.py | Weather significantly affects outcomes | ✅ Done |
| H5 | Wire `predict`/`today` to the ONNX model | predictor.rs, cli.rs | Daily commands used the heuristic `FallbackPredictor`, not the trained LambdaRank model — only `backtest` used ONNX | ✅ Done |

### H5 Notes

`predict` and `today` previously instantiated `FallbackPredictor::new()` directly,
so the trained model never ran in daily operation (its ROI figures came only from
`backtest`). Both now use the shared `UnifiedPredictor` loaded from `--model-dir`
(default `models/onnx`), with fail-soft fallback + stderr warning. `predict` builds
the full feature set via `RaceFeatureContext` (matches `backtest` to 5 decimals,
verified on 20241229/24/1); `today` uses racer history + base features only
(no results/weather for a future race). The refactor was verified non-regressive:
`backtest` detailed output is byte-identical before/after (modulo HashMap row order).

### Medium Priority (Quality of Life)

| ID | Improvement | Impact | Status |
|----|-------------|--------|--------|
| M1 | Add `--output-format json\|csv` | Enable automation | ✅ Done |
| M2 | Add date validation | Prevent invalid date errors | ✅ Done |
| M3 | Per-stadium course advantage | Stadium-specific predictions | ❌ Abandoned (hurt ROI, see below) |
| M4 | Use `tabled` crate | Better table formatting | |
| M5 | TOML config file | Persistent settings | |

### M3 Investigation Results

Stadium course features were tested but **hurt model performance**:

| Model | Features | Best Iteration | NDCG@1 | ROI |
|-------|----------|----------------|--------|-----|
| **Baseline** | 50 | 137 | 0.855 | **+178%** |
| +5 stadium course | 55 | 2 | 0.845 | +10% |
| +3 stadium course (reduced) | 53 | 13 | 0.846 | +50% |

**Root cause**: `stadium_course_win_rate` has 0.98 correlation with existing `course_advantage` feature. The new features were mostly redundant, causing early stopping (iteration 2-13 vs 137).

**Conclusion**: The existing `course_advantage` feature already captures stadium-specific patterns via learned model weights. Explicit stadium course features add noise without signal.

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
- Ensure Python and Rust feature counts match (50 features)
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
2. CSV → Rust loader → Feature extraction (50 features)
3. Features → ONNX model → Position probabilities
4. Probabilities + Odds → Expected Value → Betting decision

**Feature Categories (50 total):**
- Stadium code (1)
- Base features (10): national/local win rates, age, weight, class, branch, motor/boat rates
- Historical features (16): recent performance, course-specific stats, start timing
- Relative features (5): rankings within race
- Exhibition features (3): time, rank, diff from average
- Context features (2): race_grade, is_final
- Interaction features (6): class×course, motor×exhibition, equipment scores
- Weather features (7): weather, wind speed/direction, wave height, interactions

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

### Backtest Output Formats

The backtest command supports three output formats via `--output-format`:

**JSON Output** (`--output-format json`)
```json
{
  "config": { "ev_threshold": 1.0, "stake": 100, ... },
  "summary": { "total_bets": 345867, "roi": 1.78, ... },
  "metrics": { "hit_rate": 0.082, "profit_factor": 2.94, ... },
  "analysis": { "by_stadium": [...], "by_odds_range": [...] },
  "bets": [...]  // only with --detailed
}
```

**CSV Summary** (`--output-format csv`)
```
total_races,races_with_bets,total_bets,winning_bets,hit_rate,total_stake,total_payout,total_profit,roi,avg_ev,avg_odds,avg_probability,profit_factor,max_drawdown
116664,115317,345867,28245,0.0817,34586700,96304150,61717450,1.7844,3.6980,60.57,0.0932,2.9431,13990
```

**CSV Detailed** (`--output-format csv --detailed`)
```
date,stadium_code,race_no,combination,probability,odds,expected_value,stake,actual_first,actual_second,won,profit
20240715,23,5,1-3,0.080000,12.5,1.0000,100,1,3,true,1150
```

## References

- Official data: https://www.boatrace.jp/owpc/pc/extra/data/download.html
- Results index: https://www1.mbrace.or.jp/od2/K/dindex.html
- Programs index: https://www1.mbrace.or.jp/od2/B/dindex.html
