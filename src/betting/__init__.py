"""
Betting Strategy Module

Includes Kelly criterion and other bet sizing strategies
"""

from src.betting.kelly import (
    KellyCalculator,
    BetSizing,
    calculate_kelly_fraction,
    calculate_optimal_stake,
)

__all__ = [
    "KellyCalculator",
    "BetSizing",
    "calculate_kelly_fraction",
    "calculate_optimal_stake",
]
