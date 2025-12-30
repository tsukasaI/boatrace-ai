# Claude Code Instructions

## Project Overview

This is a Japanese boat racing (Kyotei) AI prediction system. The goal is to predict exacta (2-consecutive) race outcomes and maximize ROI using expected value-based betting strategy.

## Tech Stack

- **Language**: Python 3.11+ (use `uv` for package management)
- **ML Framework**: LightGBM (gradient boosting for tabular data)
- **Hyperparameter Tuning**: Optuna
- **Data Processing**: pandas, numpy
- **Inference API**: Rust + actix-web + ONNX Runtime
- **Data Source**: Official boat race website (boatrace.jp)

## Project Structure

```
boatrace-ai/
├── config/settings.py           # Configuration (dates, URLs, stadium codes)
├── data/
│   ├── raw/                     # Raw data (LZH -> TXT files)
│   │   ├── results/             # Race results (rankings, times)
│   │   └── programs/            # Race programs (entries, racer info)
│   ├── processed/               # Processed CSV files
│   └── odds/                    # Scraped real-time odds (JSON)
├── models/                      # Saved model files
│   ├── boatrace_model.pkl       # LightGBM model (Python)
│   └── onnx/                    # ONNX models (for Rust API)
│       ├── position_1-6.onnx    # 6 position prediction models
│       └── metadata.json        # Feature names & model info
├── results/                     # Evaluation results & plots
├── src/
│   ├── data_collection/
│   │   ├── downloader.py        # Download LZH files from official site
│   │   ├── extractor.py         # Extract LZH -> TXT
│   │   ├── odds_scraper.py      # Real-time odds scraper (exacta/trifecta)
│   │   └── collect_daily.py     # Daily odds collection orchestrator
│   ├── preprocessing/
│   │   └── parser.py            # Parse TXT -> CSV (programs & results)
│   ├── models/
│   │   ├── features.py          # Feature engineering
│   │   ├── dataset.py           # Dataset builder & train/val/test split
│   │   ├── train.py             # LightGBM training pipeline
│   │   ├── predictor.py         # Exacta probability & EV calculation
│   │   ├── evaluate.py          # Model evaluation metrics
│   │   └── export_onnx.py       # Export LightGBM to ONNX format
│   ├── backtesting/
│   │   ├── simulator.py         # Backtest simulator with EV strategy
│   │   ├── metrics.py           # ROI, hit rate, drawdown calculations
│   │   ├── report.py            # CSV/text report generation
│   │   └── synthetic_odds.py    # Synthetic odds generator
│   └── api/                     # (deprecated - using rust-api)
├── rust-api/                    # Rust inference API
│   ├── src/
│   │   ├── main.rs              # HTTP server (actix-web)
│   │   ├── models.rs            # Request/response types
│   │   ├── predictor.rs         # ONNX inference + fallback predictor
│   │   └── handlers/            # Route handlers (health, predict)
│   └── Cargo.toml               # Rust dependencies
├── tests/                       # Python test suite
└── notebooks/                   # Jupyter exploration
```

## Development Phases

### Phase 1: Data Collection & Exploration ✅ COMPLETE
1. ✅ Download race data from official site (2023-2024, 2 years)
2. ✅ Extract LZH files to TXT
3. ✅ Parse TXT to CSV (programs, results, payouts)
4. ✅ Basic data exploration

### Phase 2: Model Building ✅ COMPLETE
1. ✅ Feature engineering (base, historical, relative features)
2. ✅ LightGBM multi-output model for position probabilities
3. ✅ Exacta probability calculation
4. ✅ Expected value calculation
5. ✅ ONNX export for production inference

### Phase 3: Backtesting ✅ COMPLETE
1. ✅ PayoutParser extracts odds from raw result files
2. ✅ BacktestSimulator with EV > 1.0 betting strategy
3. ✅ Metrics: ROI, hit rate, profit factor, max drawdown
4. ✅ Analysis by stadium, race type, odds range
5. ✅ Synthetic odds generation (market simulation with noise)
6. ✅ Kelly criterion bet sizing
7. ✅ Trifecta (3-consecutive) support

### Phase 4: Inference API ✅ COMPLETE
1. ✅ Rust REST API with actix-web
2. ✅ Endpoints: /health, /predict, /predict/exacta
3. ✅ ONNX Runtime inference in Rust
4. ✅ Fallback predictor when models unavailable
5. ✅ Proper error handling with graceful fallback

### Phase 4.5: CLI Tool ✅ COMPLETE
1. ✅ Interactive prediction mode
2. ✅ Single race prediction with --date/--stadium/--race
3. ✅ Kelly criterion bet sizing recommendations
4. ✅ Trifecta prediction support
5. ✅ List available races by date

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

# Export to ONNX (for Rust API)
uv run python src/models/export_onnx.py --verify --compare
```

### Phase 3: Backtesting
```bash
# Run backtest with default settings (EV > 1.0)
uv run python -m src.backtesting.simulator

# Custom EV threshold
uv run python -m src.backtesting.simulator --threshold 1.1

# More bets per race
uv run python -m src.backtesting.simulator --max-bets 5

# Use all data (not just test set)
uv run python -m src.backtesting.simulator --all-data

# Use synthetic odds (avoids data leakage from real payouts)
uv run python -m src.backtesting.simulator --synthetic-odds

# Use real scraped odds (JSON) with fallback to payout CSV
uv run python -m src.backtesting.simulator --use-real-odds
```

### Phase 5: Odds Collection
```bash
# Scrape single race exacta odds
uv run python -m src.data_collection.odds_scraper -d 20251230 -s 23 -r 1

# Scrape single race trifecta odds
uv run python -m src.data_collection.odds_scraper -d 20251230 -s 23 -r 1 --trifecta

# Scrape all races at a stadium
uv run python -m src.data_collection.odds_scraper -d 20251230 -s 23

# Daily collection: all 24 stadiums × 12 races
uv run python -m src.data_collection.collect_daily --date 20251230

# Collect trifecta odds for all stadiums
uv run python -m src.data_collection.collect_daily --date 20251230 --trifecta

# Collect specific stadiums only
uv run python -m src.data_collection.collect_daily --stadiums 23 24

# List all stadium codes
uv run python -m src.data_collection.odds_scraper --list-stadiums
```

### CLI Tool
```bash
# Interactive mode
uv run python -m src.cli.predict --interactive

# Single race prediction
uv run python -m src.cli.predict --date 20240115 --stadium 23 --race 1

# With trifecta predictions
uv run python -m src.cli.predict -d 20240115 -s 23 -r 1 --trifecta

# Custom Kelly sizing
uv run python -m src.cli.predict -d 20240115 -s 23 -r 1 --bankroll 50000 --kelly 0.25

# List available races for a date
uv run python -m src.cli.predict --list 20240115
```

### Phase 4: Rust API
```bash
# Build and check
cd rust-api && cargo check

# Run tests
cd rust-api && cargo test

# Start server (default: http://127.0.0.1:8080)
cd rust-api && cargo run

# Custom host/port
HOST=0.0.0.0 PORT=3000 cargo run

# With ONNX models (default: ../models/onnx)
MODEL_DIR=/path/to/models/onnx cargo run
```

#### API Endpoints
- `GET /health` - Health check (returns model_loaded status)
- `POST /predict` - Full prediction (position probs + exacta + value bets)
- `POST /predict/exacta` - Exacta predictions only (top 10)

## Data URLs

- Race Results: `https://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh`
- Race Programs: `https://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh`

## Key Concepts

### Expected Value Strategy
```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

### Bet Type: Exacta (2-consecutive)
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

## Backtest Results (Synthetic Odds)

Using synthetic odds with 70% market efficiency and 25% takeout:

| EV Threshold | Bets | Wins | ROI | Hit Rate | Avg EV |
|--------------|------|------|------|----------|--------|
| 1.0 | 51,540 | 2,781 | -8.7% | 5.4% | 1.07 |
| 1.1 | 14,488 | 735 | -7.4% | 5.1% | 1.16 |
| 1.2 | 3,152 | 154 | -4.8% | 4.9% | 1.26 |
| 1.5 | 22 | 3 | +193% | 13.6% | 1.55 |

**Key Insights:**
- Higher EV thresholds reduce volume but improve ROI
- EV > 1.5 shows profitability but with very few opportunities
- Real odds data needed for accurate validation

**Note:** These results use synthetic odds (simulated market). For production validation,
real-time odds scraping from boatrace.jp is required.

## Future Plans

### Phase 5: Real-time Odds Scraping ✅ COMPLETE
- [x] Scrape exacta (2連単) odds from boatrace.jp
- [x] Scrape trifecta (3連単) odds from boatrace.jp
- [x] Daily collection script for all 24 stadiums
- [x] Integrate real odds with backtester (--use-real-odds flag)
- [x] JSON storage with date/stadium/race indexing

### Phase 6: Production Deployment
- [ ] Docker containerization for Rust API
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Health monitoring and alerting
- [ ] Rate limiting and authentication

### Phase 7: Enhanced Features
- [ ] Weather/water condition features (scrape from boatrace.jp)
- [ ] ~~Support for trifecta and trio bet types~~ ✅ Done
- [ ] ~~Kelly criterion for optimal bet sizing~~ ✅ Done
- [ ] Quinella (unordered exacta) support

### Phase 8: User Interface
- [ ] Web dashboard for daily predictions
- [ ] Mobile-friendly responsive design
- [ ] Historical performance tracking
- [ ] Customizable betting strategies

### Phase 9: Advanced ML
- [ ] Deep learning models (Transformer, LSTM)
- [ ] Ensemble methods
- [ ] Online learning for model updates
- [ ] Feature importance analysis dashboard

## References

- Official data download: https://www.boatrace.jp/owpc/pc/extra/data/download.html
- Race results index: https://www1.mbrace.or.jp/od2/K/dindex.html
- Race programs index: https://www1.mbrace.or.jp/od2/B/dindex.html
