"""
Backtest Module

Phase 3: Strategy validation using historical data
"""

from src.backtesting.simulator import BacktestSimulator, BacktestResult
from src.backtesting.metrics import calculate_metrics, BacktestMetrics

__all__ = [
    "BacktestSimulator",
    "BacktestResult",
    "calculate_metrics",
    "BacktestMetrics",
]
