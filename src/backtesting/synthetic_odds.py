"""
Synthetic Odds Generation Module

Calculate exacta probabilities by course from historical data
and simulate market odds
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Historical win rate by course (general tendency in boat racing)
# Course 1 has overwhelming advantage
HISTORICAL_WIN_RATE = {
    1: 0.55,  # Course 1: approx. 55%
    2: 0.14,  # Course 2: approx. 14%
    3: 0.12,  # Course 3: approx. 12%
    4: 0.10,  # Course 4: approx. 10%
    5: 0.06,  # Course 5: approx. 6%
    6: 0.03,  # Course 6: approx. 3%
}

# 2nd place rate (relative probability from non-1st positions)
HISTORICAL_SECOND_RATE = {
    1: 0.20,  # Probability of course 1 finishing 2nd
    2: 0.22,
    3: 0.20,
    4: 0.18,
    5: 0.12,
    6: 0.08,
}


class SyntheticOddsGenerator:
    """Synthetic odds generator"""

    def __init__(self, margin: float = 0.25):
        """
        Args:
            margin: Commission rate (takeout). 25% is standard for boat racing
        """
        self.margin = margin
        self.exacta_probs = self._calculate_exacta_probs()

    def _calculate_exacta_probs(self) -> dict:
        """
        Calculate exacta probability for each course combination

        P(1st=i, 2nd=j) = P(1st=i) x P(2nd=j | 1st!=j)
        """
        probs = {}

        for first in range(1, 7):
            p_first = HISTORICAL_WIN_RATE[first]

            # When first place is 'first', select 2nd place from remaining
            remaining_second_total = sum(
                HISTORICAL_SECOND_RATE[s] for s in range(1, 7) if s != first
            )

            for second in range(1, 7):
                if first == second:
                    continue

                # Calculate 2nd place using conditional probability
                p_second_given = HISTORICAL_SECOND_RATE[second] / remaining_second_total
                p_exacta = p_first * p_second_given

                probs[(first, second)] = p_exacta

        # Normalize (so total equals 1)
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        return probs

    def get_odds(self, first: int, second: int) -> float:
        """
        Get odds for specified combination

        Args:
            first: Boat number for 1st place
            second: Boat number for 2nd place

        Returns:
            Odds (payout multiplier)
        """
        if first == second or first < 1 or first > 6 or second < 1 or second > 6:
            return 0.0

        prob = self.exacta_probs.get((first, second), 0.001)

        # Odds = 1 / (probability x (1 - commission rate))
        # With 25% commission, payout rate is 75%
        fair_odds = 1.0 / prob
        actual_odds = fair_odds * (1 - self.margin)

        return round(actual_odds, 1)

    def get_all_odds(self) -> dict:
        """
        Get odds for all 30 combinations

        Returns:
            Dictionary of {(first, second): odds}
        """
        return {
            (f, s): self.get_odds(f, s)
            for f in range(1, 7)
            for s in range(1, 7)
            if f != s
        }

    def get_race_odds(self, race_df: pd.DataFrame = None) -> dict:
        """
        Get odds for each race

        Currently returns fixed odds, but can consider race characteristics
        (racer class, motor performance, etc.) in the future

        Returns:
            Dictionary of {(first, second): odds}
        """
        # TODO: Adjust odds based on race characteristics
        return self.get_all_odds()


def calculate_historical_exacta_rates(results_df: pd.DataFrame) -> dict:
    """
    Calculate exacta occurrence rate from actual result data

    Args:
        results_df: Results dataframe

    Returns:
        Dictionary of {(first, second): rate}
    """
    exacta_counts = {}
    total_races = 0

    # Group by race
    race_groups = results_df.groupby(["date", "stadium_code", "race_no"])

    for _, race_df in race_groups:
        if len(race_df) != 6:
            continue

        first_place = race_df[race_df["rank"] == 1]
        second_place = race_df[race_df["rank"] == 2]

        if len(first_place) == 0 or len(second_place) == 0:
            continue

        first = int(first_place["boat_no"].values[0])
        second = int(second_place["boat_no"].values[0])

        key = (first, second)
        exacta_counts[key] = exacta_counts.get(key, 0) + 1
        total_races += 1

    # Convert to probability
    if total_races > 0:
        return {k: v / total_races for k, v in exacta_counts.items()}
    return {}


if __name__ == "__main__":
    # Test
    generator = SyntheticOddsGenerator()

    print("Synthetic Exacta Odds (25% margin):")
    print("-" * 40)

    all_odds = generator.get_all_odds()

    # Sort by odds ascending (popularity order)
    sorted_odds = sorted(all_odds.items(), key=lambda x: x[1])

    for (first, second), odds in sorted_odds[:10]:
        prob = generator.exacta_probs[(first, second)]
        print(f"  {first}-{second}: {odds:6.1f}x (prob: {prob:.4f})")

    print("...")
    print(f"\nTotal combinations: {len(all_odds)}")
    print(f"Min odds: {min(all_odds.values()):.1f}x")
    print(f"Max odds: {max(all_odds.values()):.1f}x")
