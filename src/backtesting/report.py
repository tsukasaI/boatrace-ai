"""
バックテストレポート生成

結果をCSV/HTML形式で出力
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
    CSV形式でレポートを生成

    Args:
        result: バックテスト結果
        output_path: 出力先パス

    Returns:
        出力ファイルパス
    """
    output_path = output_path or RESULTS_DIR / "backtest_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not result.bets:
        return output_path

    # ベット詳細
    bets_df = pd.DataFrame([b.__dict__ for b in result.bets])
    bets_df.to_csv(output_path, index=False)

    return output_path


def generate_summary_report(result: "BacktestResult") -> str:
    """
    テキスト形式のサマリーレポートを生成

    Args:
        result: バックテスト結果

    Returns:
        レポート文字列
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST SUMMARY REPORT")
    lines.append("=" * 60)

    # 概要
    lines.append("\n[Overview]")
    lines.append(f"Total Races: {result.total_races}")
    lines.append(f"Races with Bets: {result.races_with_bets}")
    lines.append(f"Total Bets: {len(result.bets)}")

    # 収益
    lines.append("\n[Profit & Loss]")
    lines.append(f"Total Stake: ¥{result.total_stake:,}")
    lines.append(f"Total Payout: ¥{result.total_payout:,}")
    lines.append(f"Net Profit: ¥{result.total_profit:,}")
    lines.append(f"ROI: {result.roi:.1%}")

    # メトリクス
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
    レース場別の分析レポートを生成

    Args:
        result: バックテスト結果

    Returns:
        分析結果のDataFrame
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
    オッズレンジ別の分析レポートを生成

    Args:
        result: バックテスト結果

    Returns:
        分析結果のDataFrame
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
    完全なレポートをファイルに保存

    Args:
        result: バックテスト結果
        output_dir: 出力ディレクトリ
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # サマリー
    summary = generate_summary_report(result)
    (output_dir / "backtest_summary.txt").write_text(summary)

    # ベット詳細
    generate_csv_report(result, output_dir / "backtest_bets.csv")

    # レース場別分析
    if result.bets:
        stadium_analysis = generate_analysis_by_stadium(result)
        stadium_analysis.to_csv(output_dir / "analysis_by_stadium.csv", index=False)

        odds_analysis = generate_analysis_by_odds_range(result)
        odds_analysis.to_csv(output_dir / "analysis_by_odds.csv", index=False)

    print(f"Reports saved to {output_dir}")
