---
name: data-pipeline
description: >
  End-to-end boatrace-ai data pipeline: official race data download → LZH
  extraction → CP932 fixed-width parsing → CSV → model training → ONNX
  export → Rust inference build. Use when: データ再生成 (regenerating
  data/models after a fresh clone, since data/ and models/*.pkl|*.onnx are
  gitignored), fresh cloneのセットアップ, parse/downloadエラー調査, 新しい
  期間のデータ追加, モデル再訓練の準備. NOT for: ROI評価・backtestの妥当性
  検証 (real vs synthetic odds, lookahead bias, walk-forward split sanity) —
  see sibling skill `validation-and-qa`.
---

# data-pipeline

Map of how boatrace-ai turns official race data into a runnable Rust
predictor. Everything under `data/` and all `models/*.pkl|*.onnx` are
gitignored (confirmed in `.gitignore`) — a fresh clone has none of it and
must regenerate locally by running the stages below in order.

## When NOT to use

- Judging whether a backtest ROI number is trustworthy (lookahead bias,
  real vs. synthetic odds, walk-forward split correctness) — that's
  `validation-and-qa`, a sibling skill in this repo.
- Day-to-day Rust CLI operation (`boat today`, `boat predict`, `boat
  backtest`, `boat scrape`/`snapshot`) once models and data already exist —
  out of scope for this skill, which stops at "build the binary." (`boat` is
  a shell alias from `~/.zshrc`, not a binary on PATH — see the "Build Rust
  inference binary" stage below; it won't exist in a non-interactive
  session, so use `./rust-api/target/release/boatrace-cli` from repo root
  instead.)

## Pipeline stages (run in order)

All Python commands via `uv run python ...` from repo root; no
`pyproject.toml` — `uv venv && uv pip install -r requirements.txt` is the
setup (confirmed: no pyproject.toml file in repo, requirements.txt present).

1. **Download** — `uv run python src/data_collection/downloader.py`
   No CLI args; date range and URLs come from `config/settings.py`
   (`DATA_CONFIG`). Hits `https://www1.mbrace.or.jp/od2/{K,B}/{YYYYMM}/{k,b}{YYMMDD}.lzh`
   directly (K=results, B=programs) — **not** the boatrace.jp download page;
   that page is just the human-facing catalog, the code is the source of
   truth. Output: `data/raw/{results,programs}/*.lzh`.
2. **Extract** — `uv run python src/data_collection/extractor.py`
   No CLI args. Uses the pure-Python `lhafile` library (see Gotchas), not an
   external `lha`/`lhasa` binary. Output: `data/raw/{results,programs}/*.txt`
   (CP932/Shift-JIS fixed-width text).
3. **Parse** — `uv run python src/preprocessing/parser.py`
   No CLI args. Reads the `.txt` files with `encoding="cp932"` (falls back
   to utf-8/ignore on decode error), regex-parses fixed-width lines. Output:
   `data/processed/programs_races.csv`, `programs_entries.csv`,
   `results_races.csv`, `results_entries.csv`, `payouts.csv` (payouts derived
   from *result* files' 単勝/２連単/３連単 sections, not from live odds).
   Note: `data/odds/*.json` is a **separate artifact**, produced later by the
   Rust scraper (`boat scrape`/`snapshot`, `rust-api/src/scraper/`,
   `data/odds_loader.rs`) — not by this Python stage.
4. **Train** — `uv run python src/models/train.py --historical [--ranking]
   [--optimize --n-trials N] [--no-calibrate] [--no-stadium-course]
   [--train-end-date YYYYMMDD] [--val-end-date YYYYMMDD]`
   (flags confirmed via `argparse` in `src/models/train.py`). `--ranking`
   trains a single LambdaRank model; default is 6 binary classifiers.
   Output: `models/boatrace_model.pkl` (binary) or `models/boatrace_ranker.pkl`
   (ranking).
5. **Export ONNX** — `uv run python src/models/export_onnx.py --verify
   [--ranking] [--compare] [--model PATH] [--output DIR] [--features N]`
   (flags confirmed via `argparse`). Output: `models/onnx/*.onnx` +
   `models/onnx/metadata.json`.
6. **Build Rust inference binary** — `cd rust-api && cargo build --release
   --features full`
   Feature flags confirmed in `rust-api/Cargo.toml`: `default = ["api",
   "cli"]`, `scraper = [...]`, `full = ["api", "cli", "scraper"]`. `full` is
   required to get the scraper subcommands (`boat scrape`/`snapshot`) that
   later populate `data/odds/`. Output: `rust-api/target/release/boatrace-cli`
   (or crate name per `Cargo.toml`).

## Gotchas

- **LZH extraction has no external dependency to install.** `extractor.py`
  imports `lhafile` (pure Python, `requirements.txt: lhafile>=0.3.0`) guarded
  by `try/except ImportError` (`HAS_LHAFILE`); it does not shell out to `lha`
  or `lhasa`. If `lhafile` is missing, `extract_file` silently returns
  `None` per file and `main()` exits with an error only when nothing is
  installed at all.
- **CP932 encoding.** All three parsers (`ProgramParser`, `ResultParser`,
  `PayoutParser` in `src/preprocessing/parser.py`) default to
  `encoding="cp932"` and fall back to `utf-8, errors="ignore"` on
  `UnicodeDecodeError` — a silent fallback that can corrupt fullwidth
  Japanese text without raising. If parsed CSVs show garbled racer
  names/branches, suspect this fallback firing, not a parser bug.
- **Rate limit.** `config/settings.py` sets `DATA_CONFIG["request_interval"]
  = 2` (seconds); `downloader.py` sleeps this long after every date/data-type
  fetch. Do not lower this — it exists to keep load on the official server
  low; the boatrace-ai `CLAUDE.md` also documents it as "2+ seconds between
  requests."
- **`data/odds/` is the one thing a fresh clone can NEVER fully regenerate.**
  boatrace.jp only keeps ~1 week of live odds, so whatever window is already
  captured in `data/odds/` can't be re-scraped once gone. This repo's own
  `CLAUDE.md` (lines ~350, ~378-379) claims that window "end[s] at
  2024-12-29" — that's **stale**, not independently re-derived when this
  skill was first written, and now directly contradicted: `ls data/odds/ |
  cut -c1-4 | sort | uniq -c`, observed 2026-07-07, shows 600 files from
  2024, 432 from 2025 (a full Dec-2025 exacta+trifecta batch included), and
  24 already in 2026, with the newest file `20260531_11_12.json`. Whether a
  given backtest needs `--synthetic-odds` for 2025/2026 dates is a
  `validation-and-qa` concern (see that skill's corrected coverage note);
  don't assume "just re-run the scraper" fixes a coverage gap without
  re-checking `ls data/odds/` first, since the actual end date moves as
  scraping continues.
- **Data coverage gap (per repo `CLAUDE.md`, same citation as above):**
  processed CSVs span 2023-01 → 2024-12 in full, plus **Dec 2025 only** — no
  Jan–Nov 2025. `config/settings.py DATA_CONFIG` sets `start_date=2023-01-01,
  end_date=2025-12-30`, consistent with this.
- **`data/`, `*.pkl`, `*.onnx` are gitignored; `.claude/` is NOT.** Verified
  in `.gitignore`: it has `!.claude/`, overriding the user-global `.claude/`
  ignore so project-scoped config (skills, etc.) is trackable, with only
  `.claude/plans/` and `.claude/settings.local.json` re-ignored underneath
  that override. `git status` showing `data/`/model files as untracked/absent
  is expected, not a sign of a broken clone — but `.claude/` (this file
  included) is tracked and should appear in `git status`/`git diff` like any
  other repo file.

## Re-verify before relying on this map

Flags, paths, and feature names drift faster than prose. Before following a
stage above verbatim, re-check:
- `rg -n "add_argument" src/models/train.py src/models/export_onnx.py` —
  confirm flags haven't changed.
- `rg -n "\[features\]" -A6 rust-api/Cargo.toml` — confirm feature names.
- `cat .gitignore` — confirm `data/`, `*.pkl`, `*.onnx` are still ignored.
- `cat config/settings.py` — confirm `DATA_CONFIG` URLs, date range, and
  `request_interval` haven't moved.

All commands and file contents above were read from source/config files in
this repo (未実行・ファイル根拠で検証) — none of the download, extraction,
training, or build commands were actually executed while writing this skill.
