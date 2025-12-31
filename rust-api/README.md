# Boatrace Rust API & CLI

Rust implementation of the boatrace prediction system with REST API and CLI interfaces.

## Features

- **ONNX Runtime** inference for ML predictions
- **Kelly criterion** bet sizing
- **Exacta & Trifecta** probability calculations
- **Backtesting** with historical data
- **Odds scraping** from boatrace.jp
- **Raw data parsing** (CP932 fixed-width text)

## Build

```bash
# Default (API + CLI)
cargo build --release

# Full features (includes scraper)
cargo build --release --features full

# CLI only
cargo build --release --features cli
```

### Feature Flags

| Feature | Description | Binaries |
|---------|-------------|----------|
| `api` | REST API server | `boatrace-api` |
| `cli` | Command-line interface | `boatrace-cli` |
| `scraper` | Odds scraping from boatrace.jp | Adds `scrape` command |
| `full` | All features | All |

## CLI Usage

### Global Options

```bash
boatrace-cli [OPTIONS] <COMMAND>

Options:
  --data-dir <PATH>   Path to processed CSV data (default: data/processed)
  --odds-dir <PATH>   Path to odds JSON files (default: data/odds)
  -i, --interactive   Run in interactive mode
  -h, --help          Print help
  -V, --version       Print version
```

### Commands

#### `today` - Today's Race Predictions (requires `scraper` feature)

```bash
# Predict all active races (auto-scrapes odds)
boatrace-cli today

# Specific stadiums (23=Karatsu, 12=Suminoe)
boatrace-cli today -s 23,12

# With trifecta, high EV threshold
boatrace-cli today --trifecta --threshold 1.1

# Skip scraping (use cached odds)
boatrace-cli today --no-scrape

# Only show active races (currently selling tickets)
boatrace-cli today --active-only
```

Options:
- `-s, --stadiums <LIST>` - Comma-separated stadium codes
- `-r, --races <LIST>` - Comma-separated race numbers
- `--trifecta` - Include trifecta predictions
- `--threshold <FLOAT>` - EV threshold (default: 1.0)
- `--top <INT>` - Predictions per race (default: 5)
- `--no-scrape` - Use cached odds only
- `--active-only` - Only active races

#### `predict` - Single Race Prediction

```bash
# Basic prediction
boatrace-cli predict -d 20240115 -s 23 -r 1

# With trifecta predictions
boatrace-cli predict -d 20240115 -s 23 -r 1 --trifecta

# Custom Kelly sizing
boatrace-cli predict -d 20240115 -s 23 -r 1 --bankroll 50000 --kelly 0.25

# Filter by EV threshold
boatrace-cli predict -d 20240115 -s 23 -r 1 --threshold 1.1 --top 5
```

Options:
- `-d, --date <YYYYMMDD>` - Race date
- `-s, --stadium <1-24>` - Stadium code
- `-r, --race <1-12>` - Race number
- `--trifecta` - Include trifecta predictions
- `--bankroll <INT>` - Bankroll for Kelly sizing (default: 100000)
- `--kelly <FLOAT>` - Kelly multiplier (default: 0.25 = quarter Kelly)
- `--threshold <FLOAT>` - EV threshold (default: 1.0)
- `--top <INT>` - Number of predictions to show (default: 10)

#### `list` - List Available Races

```bash
# List all races for a date
boatrace-cli list -d 20240115
```

#### `backtest` - Run Backtesting

```bash
# Default backtest (test period: 2024-07-01 onwards)
boatrace-cli backtest

# Custom EV threshold
boatrace-cli backtest --threshold 1.1

# Use all historical data
boatrace-cli backtest --all-data

# With ONNX model (better predictions)
boatrace-cli backtest --all-data --model-dir ../models/onnx

# With synthetic odds (when real odds unavailable)
boatrace-cli backtest --all-data --synthetic-odds

# Full backtest: ONNX model + synthetic odds
boatrace-cli backtest --all-data --model-dir ../models/onnx --synthetic-odds

# Custom parameters
boatrace-cli backtest --threshold 1.2 --stake 1000 --max-bets 5
```

Options:
- `--threshold <FLOAT>` - EV threshold for betting (default: 1.0)
- `--stake <INT>` - Stake per bet in yen (default: 100)
- `--max-bets <INT>` - Maximum bets per race (default: 3)
- `--test-start <YYYYMMDD>` - Test period start date (default: 20240701)
- `--all-data` - Use all data, ignore test_start filter
- `--model-dir <PATH>` - Path to ONNX models (uses fallback if not specified)
- `--synthetic-odds` - Use synthetic odds when real odds unavailable

#### `scrape` - Scrape Odds (requires `scraper` feature)

```bash
# Build with scraper feature
cargo build --release --features full

# Scrape single race exacta odds
boatrace-cli scrape -d 20240115 -s 23 -r 1

# Scrape all 12 races at a stadium
boatrace-cli scrape -d 20240115 -s 23

# Scrape trifecta odds
boatrace-cli scrape -d 20240115 -s 23 --trifecta

# List stadium codes
boatrace-cli scrape --list-stadiums
```

Options:
- `-d, --date <YYYYMMDD>` - Race date
- `-s, --stadium <1-24>` - Stadium code
- `-r, --race <1-12>` - Race number (optional, scrapes all if omitted)
- `--trifecta` - Scrape trifecta instead of exacta
- `--delay <MS>` - Delay between requests (default: 2000)
- `--list-stadiums` - List all stadium codes

#### `parse` - Parse Raw Data Files

```bash
# Parse program files
boatrace-cli parse -i data/raw/programs -o data/processed -t programs

# Parse result files
boatrace-cli parse -i data/raw/results -o data/processed -t results

# Parse payout data
boatrace-cli parse -i data/raw/results -o data/processed -t payouts
```

Options:
- `-i, --input <PATH>` - Input directory with raw .txt files
- `-o, --output <PATH>` - Output directory for CSV files
- `-t, --data-type <TYPE>` - Data type: `programs`, `results`, or `payouts`

### Interactive Mode

```bash
boatrace-cli -i
# or
boatrace-cli --interactive
```

Interactive mode provides a guided menu for:
1. Predict a race
2. List races for a date
3. Run backtest
4. Exit

## API Server

```bash
# Start the API server (default port: 8080)
cargo run --bin boatrace-api

# Or with release build
./target/release/boatrace-api
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Full prediction (exacta + trifecta) |
| POST | `/predict/exacta` | Exacta prediction only |

## Stadium Codes

| Code | Name | Code | Name | Code | Name |
|------|------|------|------|------|------|
| 1 | 桐生 | 9 | 津 | 17 | 宮島 |
| 2 | 戸田 | 10 | 三国 | 18 | 徳山 |
| 3 | 江戸川 | 11 | びわこ | 19 | 下関 |
| 4 | 平和島 | 12 | 住之江 | 20 | 若松 |
| 5 | 多摩川 | 13 | 尼崎 | 21 | 芦屋 |
| 6 | 浜名湖 | 14 | 鳴門 | 22 | 福岡 |
| 7 | 蒲郡 | 15 | 丸亀 | 23 | 唐津 |
| 8 | 常滑 | 16 | 児島 | 24 | 大村 |

## Data Directory Structure

```
data/
├── processed/           # CSV files (from parse command)
│   ├── programs_races.csv
│   ├── programs_entries.csv
│   ├── results_races.csv
│   ├── results_entries.csv
│   └── payouts.csv
├── odds/                # Scraped odds JSON files
│   ├── exacta_20240115_23_01.json
│   └── trifecta_20240115_23_01.json
└── raw/                 # Raw text files (CP932 encoded)
    ├── programs/
    └── results/
```

## Tests

```bash
# Run all tests
cargo test --features full

# Run specific module tests
cargo test --features full backtesting
cargo test --features full scraper
```
