"""
Backtest Metrics

Calculate metrics such as ROI, hit rate, drawdown, etc.
"""

from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.backtesting.simulator import BacktestResult, BetRecord


@dataclass
class BacktestMetrics:
    """Backtest evaluation metrics"""
    # Basic metrics
    total_bets: int
    winning_bets: int
    hit_rate: float
    roi: float

    # Expected value related
    avg_ev: float
    avg_odds: float
    avg_probability: float

    # Risk metrics
    profit_factor: float
    max_drawdown: int
    max_drawdown_pct: float

    # Win/Loss
    gross_profit: int
    gross_loss: int
    net_profit: int


def calculate_metrics(result: "BacktestResult") -> BacktestMetrics:
    """
    Calculate metrics from backtest result

    Args:
        result: Backtest result

    Returns:
        Calculated metrics
    """
    bets = result.bets

    if not bets:
        return BacktestMetrics(
            total_bets=0,
            winning_bets=0,
            hit_rate=0.0,
            roi=0.0,
            avg_ev=0.0,
            avg_odds=0.0,
            avg_probability=0.0,
            profit_factor=0.0,
            max_drawdown=0,
            max_drawdown_pct=0.0,
            gross_profit=0,
            gross_loss=0,
            net_profit=0,
        )

    # Basic metrics
    total_bets = len(bets)
    winning_bets = sum(1 for b in bets if b.won)
    hit_rate = winning_bets / total_bets if total_bets > 0 else 0.0

    # Expected value related
    avg_ev = np.mean([b.expected_value for b in bets])
    avg_odds = np.mean([b.odds for b in bets])
    avg_probability = np.mean([b.probability for b in bets])

    # Profit/Loss calculation
    profits = [b.profit for b in bets]
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))
    net_profit = sum(profits)

    # Profit Factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown calculation
    cumulative = np.cumsum(profits)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_drawdown = int(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Drawdown percentage
    max_dd_pct = max_drawdown / result.total_stake if result.total_stake > 0 else 0.0

    # ROI
    roi = net_profit / result.total_stake if result.total_stake > 0 else 0.0

    return BacktestMetrics(
        total_bets=total_bets,
        winning_bets=winning_bets,
        hit_rate=hit_rate,
        roi=roi,
        avg_ev=avg_ev,
        avg_odds=avg_odds,
        avg_probability=avg_probability,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_dd_pct,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
    )


def analyze_by_dimension(
    bets: List["BetRecord"],
    dimension: str,
) -> dict:
    """
    Analyze bet results by specified dimension

    Args:
        bets: List of bet records
        dimension: Analysis dimension ("stadium", "race_no", "odds_range", "date")

    Returns:
        Analysis results by dimension
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    for bet in bets:
        if dimension == "stadium":
            key = bet.stadium_code
        elif dimension == "race_no":
            key = bet.race_no
        elif dimension == "odds_range":
            # Classify odds into ranges
            if bet.odds < 5:
                key = "low (<5)"
            elif bet.odds < 20:
                key = "mid (5-20)"
            else:
                key = "high (>20)"
        elif dimension == "date":
            key = bet.date
        else:
            key = "all"

        grouped[key].append(bet)

    results = {}
    for key, group_bets in grouped.items():
        total = len(group_bets)
        wins = sum(1 for b in group_bets if b.won)
        stake = sum(b.stake for b in group_bets)
        profit = sum(b.profit for b in group_bets)

        results[key] = {
            "bets": total,
            "wins": wins,
            "hit_rate": wins / total if total > 0 else 0,
            "stake": stake,
            "profit": profit,
            "roi": profit / stake if stake > 0 else 0,
        }

    return results


def calculate_sharpe_ratio(
    bets: List["BetRecord"],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe ratio

    Args:
        bets: List of bet records
        risk_free_rate: Risk-free rate (default: 0)

    Returns:
        Sharpe ratio
    """
    if not bets:
        return 0.0

    returns = [b.profit / b.stake for b in bets]
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    return (mean_return - risk_free_rate) / std_return
