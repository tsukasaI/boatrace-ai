---
name: validation-and-qa
description: >
  The discipline for making or checking ROI and model-quality claims in boatrace-ai.
  Use when: 精度/ROIの評価・検証, backtestの実行や結果の解釈, モデル再訓練の判断,
  "この戦略は儲かるか" / "このROIは信じていいか", feature追加の可否判断 (does it help
  or silently degrade the model). NOT for: data downloading/parsing/scraping — see
  sibling skill `data-pipeline`; live daily prediction runs (`today`/`predict`), which
  are operational, not validation.
allowed-tools: Bash, Read, Grep, Glob
---

# validation-and-qa

## Purpose

Every naive evaluation path in this repo is misleading in a specific, non-obvious way:
synthetic-odds ROI is circular, real-odds backtest ROI has lookahead bias, and the
50-feature model boundary is a landmine the training CLI does not enforce by default.
This skill is the checklist for not shipping (or believing) a wrong number.

## When NOT to use

- Downloading/extracting/parsing raw data → `data-pipeline` sibling skill.
- Running `today`/`predict` for actual bets → operational, not a validation question.
- One-off curiosity about a single race's features → just read the CSV/predict output.

## The validity hierarchy

Trust ROI numbers in this order — never report a lower tier as if it were a higher one:

1. **Lookahead-free, real pre-deadline snapshots** (`backtest --lookahead-free`) — the
   only tier that reflects money decided with information actually available before the
   deadline. **Currently unusable**: no historical race has a qualifying snapshot (A1,
   `snapshot-day`, has not been run across live days yet), so every race is excluded
   (`races_excluded_no_predeadline_odds == total_races`, 0 bets).
2. **Real-odds backtest** (default `backtest`, no flags) — uses final settled odds,
   which are only known *after* betting closes. Has lookahead bias. Valid for relative
   strategy comparison (EV vs by-prob vs max-odds) only.
3. **Synthetic-odds backtest** (`--synthetic-odds`) — a fixed course-position prior (same
   30 odds every race, flat 25% takeout), no race-specific market info. Circular: a model
   only has to beat a lane-number baseline. Valid for relative model-vs-model comparison
   (old vs new) only, never as a profit forecast.

Full rationale: the audit note at the top of `CLAUDE.md` (2026-06). Summarize it when
citing; don't restate it verbatim.

## Procedures

Commands below that use the bare `boat` alias assume `alias boat='./target/release/boatrace-cli'`
is defined in `~/.zshrc` (per this repo's `CLAUDE.md`) and that cwd is `rust-api/` — it won't
exist in a non-interactive session; use `./rust-api/target/release/boatrace-cli` (verified
binary path) from repo root as the portable form instead.

**Relative model comparison (safe today)**
```bash
cd rust-api
./target/release/boatrace-cli backtest --all-data --model-dir ../models/onnx --synthetic-odds
```
Compare ROI/hit-rate/NDCG deltas between two `--model-dir`s, not the absolute ROI.

**Real-odds strategy comparison (still lookahead-biased, but the least-bad "does this
strategy shape make sense" check)**
```bash
boat backtest --all-data --model-dir ../models/onnx --by-prob      # best ROI, lowest drawdown historically
boat backtest --all-data --model-dir ../models/onnx --max-odds 30  # EV strategy, longshot-filtered
```

**Lookahead-free (the real check — currently produces 0 bets until A1 snapshots exist)**
```bash
boat backtest --all-data --model-dir ../models/onnx \
  --lookahead-free --snapshot-dir ../data/odds_snapshots
```
Expect `races_excluded_no_predeadline_odds == total_races` today. If that count drops
below total, snapshots exist — re-read the audit note before trusting the resulting ROI,
since only some races will have pre-deadline coverage.

**Retraining decision** — don't retrain speculatively; the walk-forward test showed the
2023-only ranker did not decay on a genuine forward holdout (Dec-2025, +126.5% ROI vs
+127.4% for a 2024-updated model, <1pp difference). Only retrain with a stated reason
(new data window, a specific feature change to test), and always compare against the
current `models/onnx` via the synthetic-odds relative check above before replacing it.
```bash
uv run python src/models/train.py --historical --ranking --no-stadium-course
uv run python src/models/export_onnx.py --ranking
```

**Feature-addition check** — train once with the candidate feature(s) added, watch
`best_iteration` in the LightGBM log: if it early-stops far below the ~137 iterations of
the healthy baseline, the feature is redundant/harmful — do not export it to ONNX.

## Hard constraints (verified against source, with commit evidence)

- **50-feature boundary, and the CLI default is inverted.** `src/models/train.py:472`
  (`train_model`'s own default) and line 575 (`include_stadium_course=not
  args.no_stadium_course`) mean **plain `train.py --historical --ranking` includes the
  harmful stadium-course features by default** — you must pass `--no-stadium-course`
  explicitly to get the safe 50-feature baseline. This contradicts `features.py:724`'s
  own default (`include_stadium_course: bool = False`), which was flipped in commit
  `d66aeb5` but train.py's CLI wiring was not updated to match. Confirmed live:
  `models/onnx/metadata.json` currently has exactly 50 `feature_names`, so whoever last
  trained it passed the flag — but the footgun is real for the next run.
  Commit evidence: `a0a7164` (55-feature model regressed ROI 178%→14.5%), `d66aeb5`
  (M3 abandoned, default flip in features.py only), and a later retrain commit
  (`learned(train): stadium-course features must stay off ... a 53-feature retrain
  early-stopped at iteration 14 (NDCG@1=0.843)`).
- **PreDeadlineOdds is fail-closed, but only on the `--lookahead-free` path.**
  `rust-api/src/backtesting/simulator.rs:758-817` — when `lookahead_free` is set, a
  `SnapshotIndex` is built and a race yields usable odds only via
  `PreDeadlineOdds::Selected`; missing/unparseable deadline or timestamp, or a legacy
  single-file odds JSON with no deadline, is excluded (`OnlyPostClose`/`NoSnapshots`),
  never bet (`rust-api/src/data/odds_loader.rs`). Test proving legacy naive
  (offset-less) `scraped_at` values are unparseable and excluded:
  `odds_loader.rs::test_predeadline_naive_timestamp_excluded`. **The default (no-flags)
  `backtest` tier does not go through this filter at all** —
  `simulator.rs:818-834` calls `load_exacta_odds`/`load_trifecta_odds` directly against
  `odds_dir` by `(date, stadium, race_no)` filename, with no deadline/timestamp check;
  it uses whatever single `scraped_at` snapshot the file holds, pre- or post-deadline,
  indiscriminately. So the default tier's lookahead bias comes from not knowing whether
  a given file's snapshot predates the race deadline, not from a data-coverage cutoff —
  and it will happily use those same legacy naive-timestamp files that
  `--lookahead-free` excludes.
- **Data coverage windows.** `data/processed/` covers 2023-01→2024-12 full plus
  2025-12 only (verified: only 7 CSVs, matching `CLAUDE.md`'s stated coverage; no
  Jan–Nov 2025). Real scraped odds (`data/odds/`) do **not** end at 2024-12-29 — that
  claim (`CLAUDE.md` lines 350 and 379 in this repo) is stale. Verified 2026-07-07:
  `ls data/odds/ | cut -c1-4 | sort | uniq -c` shows 600 files from 2024, 432 from 2025
  (a full Dec-2025 exacta+trifecta batch is present, e.g. `20251230_01_01.json` /
  `20251230_01_01_3t.json`), and 24 already in 2026; the newest file is
  `20260531_11_12.json`. Combined with the point above: a default-tier backtest on 2025
  (or 2026) data is *not* starved of real-odds bets and does not silently fall back to
  near-zero real-odds coverage — real odds files exist for that range and load
  unfiltered. `--synthetic-odds` is still the right choice for tier-3 relative
  model-vs-model comparisons (see validity hierarchy above), but not because tier-2 data
  is missing for 2025.

## Re-verify (don't trust this file blindly — re-check before relying on any of it)

```bash
# 1. Re-read the current audit note (numbers/caveats may have changed)
sed -n '1,32p' CLAUDE.md   # or open in nvim

# 2. Confirm cli.rs subcommand/flag names haven't drifted
rg -n '^\s*(Backtest|Snapshot|SnapshotDay)\b' rust-api/src/bin/cli.rs
rg -n 'lookahead_free|no_stadium_course' rust-api/src/bin/cli.rs src/models/train.py

# 3. WARNING: a pre-built binary can lag the source. Observed here: the checked-in
#    rust-api/target/release/boatrace-cli's `backtest --help` does NOT list
#    --lookahead-free/--snapshot-dir even though cli.rs defines them — rebuild
#    (`cargo build --release --features full`) before trusting --help output.

# 4. Data coverage / snapshot progress
ls data/processed/
ls data/odds_snapshots/ 2>&1 | head   # non-empty means A1 has started

# 5. Current model's actual feature count (ground truth over any doc claim)
jq '.feature_names | length' models/onnx/metadata.json
```

## References

- `CLAUDE.md` (this repo) — audit note (top), M3 investigation, betting-strategy tables.
- Sibling `data-pipeline` skill (this repo) — data download/parse/scrape, out of scope here.
- `~/engineer/gamble_support/keiba-ai/.claude/skills/validation-and-qa` — same class of
  problem (optimistic backtest vs verified ROI), solved there via forward paper-trading
  (`docs/PAPER_TRADING.md`: paper-record → race runs → paper-settle → paper-report)
  instead of historical pre-deadline snapshots. boatrace-ai's A1/A2 path is the snapshot
  analog of that idea, not yet validated on live days.
