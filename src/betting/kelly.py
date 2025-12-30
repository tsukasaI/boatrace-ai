"""
Kelly Criterion Bet Sizing

Optimal bet sizing based on edge and odds using Kelly criterion.

The Kelly criterion formula:
    f* = (b*p - q) / b = (p*odds - 1) / (odds - 1)

Where:
    f* = fraction of bankroll to bet
    b = odds - 1 (net odds)
    p = probability of winning
    q = 1 - p (probability of losing)
    odds = decimal odds (e.g., 5.0 means 5x return)
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class BetSizing:
    """Bet sizing recommendation"""
    probability: float
    odds: float
    expected_value: float
    edge: float  # EV - 1
    kelly_fraction: float  # Full Kelly
    recommended_fraction: float  # After applying Kelly multiplier
    stake: int  # Recommended stake amount


def calculate_kelly_fraction(probability: float, odds: float) -> float:
    """
    Calculate Kelly fraction for a single bet

    Args:
        probability: Estimated probability of winning (0-1)
        odds: Decimal odds (e.g., 5.0 = 5x return)

    Returns:
        Kelly fraction (can be negative if EV < 1)

    Examples:
        >>> calculate_kelly_fraction(0.25, 5.0)  # EV = 1.25
        0.0625
        >>> calculate_kelly_fraction(0.10, 5.0)  # EV = 0.5
        -0.125
    """
    if odds <= 1.0:
        return 0.0

    # f* = (p * odds - 1) / (odds - 1)
    kelly = (probability * odds - 1) / (odds - 1)
    return kelly


def calculate_optimal_stake(
    probability: float,
    odds: float,
    bankroll: int,
    kelly_multiplier: float = 0.25,
    min_stake: int = 100,
    max_stake_pct: float = 0.10,
) -> int:
    """
    Calculate optimal stake amount

    Args:
        probability: Estimated probability of winning
        odds: Decimal odds
        bankroll: Current bankroll amount
        kelly_multiplier: Kelly fraction multiplier (default: 0.25 = quarter Kelly)
        min_stake: Minimum stake amount (default: 100 yen)
        max_stake_pct: Maximum stake as percentage of bankroll (default: 10%)

    Returns:
        Recommended stake amount (rounded to 100 yen)
    """
    kelly = calculate_kelly_fraction(probability, odds)

    if kelly <= 0:
        return 0

    # Apply Kelly multiplier (fractional Kelly)
    fraction = kelly * kelly_multiplier

    # Calculate stake
    stake = bankroll * fraction

    # Apply maximum stake limit
    max_stake = bankroll * max_stake_pct
    stake = min(stake, max_stake)

    # Round to nearest 100 yen
    stake = int(stake // 100) * 100

    # Apply minimum stake
    if stake < min_stake:
        return 0 if kelly * kelly_multiplier * bankroll < min_stake / 2 else min_stake

    return stake


class KellyCalculator:
    """
    Kelly criterion calculator for bet sizing

    Supports:
    - Full Kelly (aggressive)
    - Fractional Kelly (conservative, default 1/4)
    - Multiple simultaneous bets
    """

    def __init__(
        self,
        bankroll: int,
        kelly_multiplier: float = 0.25,
        min_stake: int = 100,
        max_stake_pct: float = 0.10,
        max_total_exposure: float = 0.30,
    ):
        """
        Args:
            bankroll: Initial bankroll
            kelly_multiplier: Fraction of Kelly to use (0.25 = quarter Kelly)
            min_stake: Minimum bet size
            max_stake_pct: Maximum single bet as % of bankroll
            max_total_exposure: Maximum total exposure across all bets
        """
        self.bankroll = bankroll
        self.kelly_multiplier = kelly_multiplier
        self.min_stake = min_stake
        self.max_stake_pct = max_stake_pct
        self.max_total_exposure = max_total_exposure

    def calculate_single(
        self,
        probability: float,
        odds: float,
    ) -> BetSizing:
        """
        Calculate bet sizing for a single bet

        Args:
            probability: Win probability
            odds: Decimal odds

        Returns:
            BetSizing with recommendation
        """
        ev = probability * odds
        edge = ev - 1
        kelly = calculate_kelly_fraction(probability, odds)
        recommended = max(0, kelly * self.kelly_multiplier)

        stake = calculate_optimal_stake(
            probability=probability,
            odds=odds,
            bankroll=self.bankroll,
            kelly_multiplier=self.kelly_multiplier,
            min_stake=self.min_stake,
            max_stake_pct=self.max_stake_pct,
        )

        return BetSizing(
            probability=probability,
            odds=odds,
            expected_value=ev,
            edge=edge,
            kelly_fraction=kelly,
            recommended_fraction=recommended,
            stake=stake,
        )

    def calculate_multiple(
        self,
        bets: List[tuple],
    ) -> List[BetSizing]:
        """
        Calculate bet sizing for multiple simultaneous bets

        When betting on multiple outcomes, we need to consider:
        1. Total exposure limit
        2. Overlapping events (same race = mutually exclusive)

        Args:
            bets: List of (probability, odds) tuples

        Returns:
            List of BetSizing recommendations
        """
        if not bets:
            return []

        # Calculate individual sizing first
        sizings = [self.calculate_single(p, o) for p, o in bets]

        # Calculate total requested stake
        total_stake = sum(s.stake for s in sizings)

        # Check if we exceed max total exposure
        max_exposure = int(self.bankroll * self.max_total_exposure)

        if total_stake > max_exposure:
            # Scale down proportionally
            scale_factor = max_exposure / total_stake

            for sizing in sizings:
                new_stake = int(sizing.stake * scale_factor // 100) * 100
                if new_stake < self.min_stake:
                    new_stake = 0
                sizing.stake = new_stake
                sizing.recommended_fraction *= scale_factor

        return sizings

    def update_bankroll(self, profit: int) -> None:
        """
        Update bankroll after bet result

        Args:
            profit: Profit (positive) or loss (negative)
        """
        self.bankroll += profit

    def get_edge_threshold(self) -> float:
        """
        Get minimum edge required for a bet

        With Kelly criterion, we only bet when EV > 1 (edge > 0)
        But with fractional Kelly, we might want higher threshold

        Returns:
            Minimum edge (e.g., 0.05 = 5% edge required)
        """
        # For 1/4 Kelly, still bet at edge > 0 but stake will be small
        return 0.0

    def simulate_growth(
        self,
        probabilities: List[float],
        odds_list: List[float],
        outcomes: List[bool],
    ) -> List[int]:
        """
        Simulate bankroll growth over a series of bets

        Args:
            probabilities: List of win probabilities
            odds_list: List of decimal odds
            outcomes: List of actual outcomes (True = win)

        Returns:
            List of bankroll values after each bet
        """
        bankroll_history = [self.bankroll]

        for prob, odds, won in zip(probabilities, odds_list, outcomes):
            sizing = self.calculate_single(prob, odds)

            if sizing.stake > 0:
                if won:
                    profit = int(sizing.stake * (odds - 1))
                else:
                    profit = -sizing.stake

                self.update_bankroll(profit)

            bankroll_history.append(self.bankroll)

        return bankroll_history


def compare_strategies(
    probabilities: List[float],
    odds_list: List[float],
    outcomes: List[bool],
    initial_bankroll: int = 100000,
) -> dict:
    """
    Compare different betting strategies

    Args:
        probabilities: Win probabilities for each bet
        odds_list: Odds for each bet
        outcomes: Actual outcomes
        initial_bankroll: Starting bankroll

    Returns:
        Dictionary with final bankrolls for each strategy
    """
    strategies = {
        "flat_100": {"stake": 100, "kelly_mult": None},
        "flat_1000": {"stake": 1000, "kelly_mult": None},
        "kelly_full": {"stake": None, "kelly_mult": 1.0},
        "kelly_half": {"stake": None, "kelly_mult": 0.5},
        "kelly_quarter": {"stake": None, "kelly_mult": 0.25},
    }

    results = {}

    for name, config in strategies.items():
        bankroll = initial_bankroll

        for prob, odds, won in zip(probabilities, odds_list, outcomes):
            ev = prob * odds
            if ev <= 1.0:
                continue  # Skip negative EV bets

            if config["kelly_mult"] is not None:
                # Kelly-based sizing
                kelly = calculate_kelly_fraction(prob, odds)
                fraction = max(0, kelly * config["kelly_mult"])
                stake = int(bankroll * fraction // 100) * 100
                stake = min(stake, int(bankroll * 0.10))  # Max 10%
            else:
                # Flat betting
                stake = config["stake"]

            stake = max(0, min(stake, bankroll))

            if stake > 0:
                if won:
                    bankroll += int(stake * (odds - 1))
                else:
                    bankroll -= stake

        results[name] = bankroll

    return results


if __name__ == "__main__":
    # Example usage
    print("Kelly Criterion Examples")
    print("=" * 50)

    # Example 1: Single bet
    print("\n1. Single bet example:")
    print("   Probability: 25%, Odds: 5.0x")
    kelly = calculate_kelly_fraction(0.25, 5.0)
    print(f"   Kelly fraction: {kelly:.4f} ({kelly*100:.2f}%)")
    print(f"   EV: {0.25 * 5.0:.2f}")

    # Example 2: Calculator
    print("\n2. Bet sizing with 100,000 yen bankroll:")
    calc = KellyCalculator(bankroll=100000, kelly_multiplier=0.25)

    test_bets = [
        (0.20, 6.0),   # EV = 1.20
        (0.15, 10.0),  # EV = 1.50
        (0.10, 15.0),  # EV = 1.50
    ]

    for prob, odds in test_bets:
        sizing = calc.calculate_single(prob, odds)
        print(f"\n   Prob: {prob:.0%}, Odds: {odds:.1f}x")
        print(f"   EV: {sizing.expected_value:.2f}, Edge: {sizing.edge:.2%}")
        print(f"   Kelly: {sizing.kelly_fraction:.4f}")
        print(f"   Recommended stake: ¥{sizing.stake:,}")

    # Example 3: Multiple bets in same race
    print("\n3. Multiple bets (same race, max exposure 30%):")
    sizings = calc.calculate_multiple(test_bets)
    total = sum(s.stake for s in sizings)
    print(f"   Total stake: ¥{total:,} ({total/calc.bankroll:.1%} of bankroll)")
