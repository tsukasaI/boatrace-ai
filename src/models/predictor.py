"""
Prediction and Expected Value Calculation

Calculate exacta (2-consecutive) probabilities and expected values from model predictions
"""

import sys
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.train import BoatracePredictor
from src.models.features import FeatureEngineering, get_feature_columns


@dataclass
class ExactaBet:
    """Exacta bet information"""
    first: int       # 1st place boat number (1-6)
    second: int      # 2nd place boat number (1-6)
    probability: float
    odds: float = 0.0
    expected_value: float = 0.0


class RacePredictor:
    """Race prediction class"""

    def __init__(self, model: BoatracePredictor = None):
        """
        Args:
            model: Trained model
        """
        self.model = model or BoatracePredictor()
        self.feature_eng = FeatureEngineering()

    def load_model(self, path: Path = None) -> None:
        """Load model"""
        self.model.load(path)

    def predict_positions(self, X: np.ndarray) -> np.ndarray:
        """
        Predict finishing position probabilities

        Args:
            X: Features (6, n_features) - for 6 boats in a race

        Returns:
            Probabilities (6, 6) - each boat's probability for each position
        """
        return self.model.predict(X)

    def calculate_exacta_probabilities(
        self,
        position_probs: np.ndarray,
    ) -> List[ExactaBet]:
        """
        Calculate exacta probabilities

        Args:
            position_probs: Position probabilities (6, 6) - [boat_no, position]

        Returns:
            List of 30 exacta bet information
        """
        exacta_bets = []

        for first in range(6):
            p_first = position_probs[first, 0]  # 1st place probability

            for second in range(6):
                if first == second:
                    continue

                p_second = position_probs[second, 1]  # 2nd place probability

                # Conditional probability: P(B=2nd | A=1st)
                # Approx P(B=2nd) / (1 - P(B=1st))
                p_second_given_first = p_second / max(1 - position_probs[second, 0], 0.01)

                # Exacta probability
                p_exacta = p_first * p_second_given_first

                exacta_bets.append(ExactaBet(
                    first=first + 1,  # 1-indexed
                    second=second + 1,
                    probability=p_exacta,
                ))

        # Sort by probability in descending order
        exacta_bets.sort(key=lambda x: x.probability, reverse=True)

        return exacta_bets

    def calculate_expected_values(
        self,
        exacta_bets: List[ExactaBet],
        odds: dict,
    ) -> List[ExactaBet]:
        """
        Calculate expected values

        Args:
            exacta_bets: List of exacta bet information
            odds: Odds dictionary {(first, second): odds}

        Returns:
            List of exacta bet information with expected values added
        """
        for bet in exacta_bets:
            key = (bet.first, bet.second)
            if key in odds:
                bet.odds = odds[key]
                bet.expected_value = bet.probability * bet.odds

        return exacta_bets

    def get_value_bets(
        self,
        exacta_bets: List[ExactaBet],
        threshold: float = 1.0,
    ) -> List[ExactaBet]:
        """
        Get bets with expected value exceeding threshold

        Args:
            exacta_bets: List of exacta bet information
            threshold: Expected value threshold

        Returns:
            List of value bets
        """
        return [bet for bet in exacta_bets if bet.expected_value > threshold]

    def predict_race(
        self,
        race_features: np.ndarray,
        odds: dict = None,
    ) -> Tuple[np.ndarray, List[ExactaBet]]:
        """
        Main function for race prediction

        Args:
            race_features: Features (6, n_features)
            odds: Odds dictionary

        Returns:
            (Position probabilities, List of exacta bet information)
        """
        # Predict position probabilities
        position_probs = self.predict_positions(race_features)

        # Calculate exacta probabilities
        exacta_bets = self.calculate_exacta_probabilities(position_probs)

        # Calculate expected values
        if odds:
            exacta_bets = self.calculate_expected_values(exacta_bets, odds)

        return position_probs, exacta_bets


def format_prediction_result(
    position_probs: np.ndarray,
    exacta_bets: List[ExactaBet],
    top_n: int = 10,
) -> str:
    """
    Format prediction results and return as string

    Args:
        position_probs: Position probabilities (6, 6)
        exacta_bets: List of exacta bet information
        top_n: Number of top entries to display

    Returns:
        Formatted string
    """
    lines = []

    # Position probabilities
    lines.append("=== Position Probabilities ===")
    lines.append("Boat  1st    2nd    3rd    4th    5th    6th")
    lines.append("-" * 50)

    for boat in range(6):
        probs = position_probs[boat]
        prob_str = " ".join(f"{p:.1%}" for p in probs)
        lines.append(f"  {boat + 1}  {prob_str}")

    lines.append("")

    # Exacta prediction
    lines.append(f"=== Exacta TOP{top_n} ===")
    lines.append("Rank  Combination  Prob    Odds     EV")
    lines.append("-" * 50)

    for i, bet in enumerate(exacta_bets[:top_n]):
        odds_str = f"{bet.odds:.1f}" if bet.odds > 0 else "-"
        ev_str = f"{bet.expected_value:.2f}" if bet.expected_value > 0 else "-"
        lines.append(
            f" {i + 1:2}    {bet.first}-{bet.second}      "
            f"{bet.probability:.1%}   {odds_str:>6}   {ev_str:>6}"
        )

    # Value bets
    value_bets = [b for b in exacta_bets if b.expected_value > 1.0]
    if value_bets:
        lines.append("")
        lines.append("=== Value Bets (EV > 1.0) ===")
        for bet in value_bets:
            lines.append(
                f"  {bet.first}-{bet.second}: Prob={bet.probability:.1%}, "
                f"Odds={bet.odds:.1f}, EV={bet.expected_value:.2f}"
            )

    return "\n".join(lines)
