# Claude Code Instructions

## Project Overview

This is a Japanese boat racing (競艇/Kyotei) AI prediction system. The goal is to predict 2-consecutive (2連単/Exacta) race outcomes and maximize ROI using expected value-based betting strategy.

## Tech Stack

- **Language**: Python 3.11+ (use `uv` for package management)
- **ML Framework**: LightGBM (gradient boosting for tabular data)
- **Hyperparameter Tuning**: Optuna
- **Data Processing**: pandas, numpy
- **Inference API**: Go or Rust (Phase 4 - not yet implemented)
- **Data Source**: Official boat race website (boatrace.jp)

## Project Structure

```
boatrace-ai/
├── config/settings.py           # Configuration (dates, URLs, stadium codes)
├── data/
│   ├── raw/                     # Raw data (LZH → TXT files)
│   │   ├── results/             # Race results (着順, タイム)
│   │   └── programs/            # Race programs (出走表, 選手情報)
│   └── processed/               # Processed CSV files
├── models/                      # Saved model files (.pkl)
├── results/                     # Evaluation results & plots
├── src/
│   ├── data_collection/
│   │   ├── downloader.py        # Download LZH files from official site
│   │   └── extractor.py         # Extract LZH → TXT
│   ├── preprocessing/
│   │   └── parser.py            # Parse TXT → CSV (programs & results)
│   ├── models/
│   │   ├── features.py          # Feature engineering
│   │   ├── dataset.py           # Dataset builder & train/val/test split
│   │   ├── train.py             # LightGBM training pipeline
│   │   ├── predictor.py         # Exacta probability & EV calculation
│   │   └── evaluate.py          # Model evaluation metrics
│   └── api/                     # Inference API (Phase 4)
└── notebooks/                   # Jupyter exploration
```

## Development Phases

### Phase 1: Data Collection & Exploration ✅ COMPLETE
1. ✅ Download race data from official site (2023-2024, 2 years)
2. ✅ Extract LZH files to TXT
3. ✅ Parse TXT to CSV
4. ⬚ Explore data in Jupyter to understand features

### Phase 2: Model Building ✅ IMPLEMENTED
1. ✅ Feature engineering (base, historical, relative features)
2. ✅ LightGBM multi-output model for position probabilities
3. ✅ Exacta probability calculation
4. ✅ Expected value calculation
5. ⬚ Train and evaluate on full dataset

### Phase 3: Backtesting 🔲 TODO
1. Validate strategy on historical data
2. Calculate ROI for "expected value > 1.0" betting
3. Analyze profitability by stadium, race type, odds range

### Phase 4: Inference API & UI 🔲 TODO
1. Build REST API in Go or Rust
2. Simple dashboard for daily predictions
3. Real-time odds integration

## Commands

### Setup
```bash
cd boatrace-ai
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Phase 1: Data Collection
```bash
# 1. Download data (2 years, ~4-5 hours with 2-sec delays)
uv run python src/data_collection/downloader.py

# 2. Extract LZH files
uv run python src/data_collection/extractor.py

# 3. Parse to CSV
uv run python src/preprocessing/parser.py
```

### Phase 2: Model Training
```bash
# Train simple model (no historical features, faster)
uv run python src/models/train.py

# Train with historical features (recommended)
uv run python src/models/train.py --historical

# Train with hyperparameter optimization
uv run python src/models/train.py --historical --optimize --n-trials 50

# Evaluate model
uv run python src/models/evaluate.py --historical
```

## Data URLs

- Race Results: `https://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh`
- Race Programs: `https://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh`

## Key Concepts

### Expected Value Strategy
```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

### Bet Type: 2連単 (Exacta)
- Predict 1st and 2nd place in exact order
- 30 combinations (6 boats × 5 remaining)

### Model Output Design
The model outputs **probability distribution for each boat's finishing position**:

```python
# Output per boat (6 boats × 6 positions)
position_probs[boat_idx, position_idx]  # P(boat finishes in position)

# Exacta probability calculation
P(boat_i=1st, boat_j=2nd) ≈ P(boat_i=1st) × P(boat_j=2nd) / (1 - P(boat_j=1st))
```

### Feature Categories
1. **Base features**: National/local win rates, age, weight, class, motor/boat stats
2. **Historical features**: Recent N races performance, course preference, avg ST timing
3. **Relative features**: Rank within race, difference from race average

## Important Notes

1. **Request Interval**: 2+ seconds between requests (server load)
2. **Encoding**: Raw files use CP932 (Shift-JIS) encoding
3. **File Format**: Fixed-width text, requires careful parsing
4. **Stadium Codes**: 1-24 for 24 race venues across Japan
5. **Data Split**: 2023 train / 2024-H1 val / 2024-H2 test (time-based)

## TODO

### Immediate
- [ ] Wait for data download to complete (~5 hours)
- [ ] Run extractor and parser on full dataset
- [ ] Train model and evaluate baseline accuracy

### Short-term
- [ ] Add weather/water condition features (requires scraping)
- [ ] Implement backtesting simulation with actual odds
- [ ] Create Jupyter notebook for data exploration

### Long-term
- [ ] Build inference API (Go/Rust)
- [ ] Add support for 3連単, 3連複
- [ ] Implement Kelly criterion for bet sizing
- [ ] Create web dashboard

## References

- Official data download: https://www.boatrace.jp/owpc/pc/extra/data/download.html
- Race results index: https://www1.mbrace.or.jp/od2/K/dindex.html
- Race programs index: https://www1.mbrace.or.jp/od2/B/dindex.html
