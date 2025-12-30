"""
Backtest Report Generation

Output results in CSV/HTML format
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.backtesting.simulator import BacktestResult

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"


def generate_csv_report(result: "BacktestResult", output_path: Path = None) -> Path:
    """
    Generate report in CSV format

    Args:
        result: Backtest result
        output_path: Output file path

    Returns:
        Output file path
    """
    output_path = output_path or RESULTS_DIR / "backtest_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not result.bets:
        return output_path

    # Bet details
    bets_df = pd.DataFrame([b.__dict__ for b in result.bets])
    bets_df.to_csv(output_path, index=False)

    return output_path


def generate_summary_report(result: "BacktestResult") -> str:
    """
    Generate text format summary report

    Args:
        result: Backtest result

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST SUMMARY REPORT")
    lines.append("=" * 60)

    # Overview
    lines.append("\n[Overview]")
    lines.append(f"Total Races: {result.total_races}")
    lines.append(f"Races with Bets: {result.races_with_bets}")
    lines.append(f"Total Bets: {len(result.bets)}")

    # Profit/Loss
    lines.append("\n[Profit & Loss]")
    lines.append(f"Total Stake: ¥{result.total_stake:,}")
    lines.append(f"Total Payout: ¥{result.total_payout:,}")
    lines.append(f"Net Profit: ¥{result.total_profit:,}")
    lines.append(f"ROI: {result.roi:.1%}")

    # Metrics
    if result.metrics:
        m = result.metrics
        lines.append("\n[Performance Metrics]")
        lines.append(f"Hit Rate: {m.hit_rate:.1%}")
        lines.append(f"Average EV: {m.avg_ev:.2f}")
        lines.append(f"Average Odds: {m.avg_odds:.1f}")
        lines.append(f"Profit Factor: {m.profit_factor:.2f}")
        lines.append(f"Max Drawdown: ¥{m.max_drawdown:,} ({m.max_drawdown_pct:.1%})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def generate_analysis_by_stadium(result: "BacktestResult") -> pd.DataFrame:
    """
    Generate analysis report by stadium

    Args:
        result: Backtest result

    Returns:
        DataFrame of analysis results
    """
    from src.backtesting.metrics import analyze_by_dimension
    from config.settings import STADIUM_CODES

    analysis = analyze_by_dimension(result.bets, "stadium")

    rows = []
    for stadium_code, stats in sorted(analysis.items()):
        rows.append({
            "stadium_code": stadium_code,
            "stadium_name": STADIUM_CODES.get(stadium_code, "Unknown"),
            "bets": stats["bets"],
            "wins": stats["wins"],
            "hit_rate": f"{stats['hit_rate']:.1%}",
            "stake": stats["stake"],
            "profit": stats["profit"],
            "roi": f"{stats['roi']:.1%}",
        })

    return pd.DataFrame(rows)


def generate_analysis_by_odds_range(result: "BacktestResult") -> pd.DataFrame:
    """
    Generate analysis report by odds range

    Args:
        result: Backtest result

    Returns:
        DataFrame of analysis results
    """
    from src.backtesting.metrics import analyze_by_dimension

    analysis = analyze_by_dimension(result.bets, "odds_range")

    rows = []
    for odds_range, stats in analysis.items():
        rows.append({
            "odds_range": odds_range,
            "bets": stats["bets"],
            "wins": stats["wins"],
            "hit_rate": f"{stats['hit_rate']:.1%}",
            "stake": stats["stake"],
            "profit": stats["profit"],
            "roi": f"{stats['roi']:.1%}",
        })

    return pd.DataFrame(rows)


def save_full_report(result: "BacktestResult", output_dir: Path = None) -> None:
    """
    Save full report to files

    Args:
        result: Backtest result
        output_dir: Output directory
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary = generate_summary_report(result)
    (output_dir / "backtest_summary.txt").write_text(summary)

    # Bet details
    generate_csv_report(result, output_dir / "backtest_bets.csv")

    # Analysis by stadium
    if result.bets:
        stadium_analysis = generate_analysis_by_stadium(result)
        stadium_analysis.to_csv(output_dir / "analysis_by_stadium.csv", index=False)

        odds_analysis = generate_analysis_by_odds_range(result)
        odds_analysis.to_csv(output_dir / "analysis_by_odds.csv", index=False)

    print(f"Reports saved to {output_dir}")
