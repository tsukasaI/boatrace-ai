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
from src.exceptions import ValidationError, PredictionError


def validate_position_probs(position_probs: np.ndarray) -> None:
    """
    Validate position probability array

    Args:
        position_probs: Position probabilities (6, 6)

    Raises:
        ValidationError: If array is invalid
    """
    if position_probs is None:
        raise ValidationError("Position probabilities cannot be None")

    if not isinstance(position_probs, np.ndarray):
        raise ValidationError(
            f"Position probabilities must be numpy array, got {type(position_probs)}"
        )

    if position_probs.shape != (6, 6):
        raise ValidationError(
            f"Position probabilities must have shape (6, 6), got {position_probs.shape}"
        )

    if np.any(position_probs < 0) or np.any(position_probs > 1):
        raise ValidationError("Position probabilities must be between 0 and 1")


def validate_boat_number(boat_no: int, name: str = "boat") -> None:
    """
    Validate boat number

    Args:
        boat_no: Boat number to validate
        name: Parameter name for error message

    Raises:
        ValidationError: If boat number is invalid
    """
    if not isinstance(boat_no, int):
        raise ValidationError(f"{name} must be an integer, got {type(boat_no)}")
    if boat_no < 1 or boat_no > 6:
        raise ValidationError(f"{name} must be between 1 and 6, got {boat_no}")


@dataclass
class ExactaBet:
    """Exacta bet information"""
    first: int       # 1st place boat number (1-6)
    second: int      # 2nd place boat number (1-6)
    probability: float
    odds: float = 0.0
    expected_value: float = 0.0


@dataclass
class TrifectaBet:
    """Trifecta (3連単) bet information"""
    first: int       # 1st place boat number (1-6)
    second: int      # 2nd place boat number (1-6)
    third: int       # 3rd place boat number (1-6)
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

        Raises:
            ValidationError: If position_probs is invalid
        """
        validate_position_probs(position_probs)

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

    def calculate_trifecta_probabilities(
        self,
        position_probs: np.ndarray,
    ) -> List[TrifectaBet]:
        """
        Calculate trifecta (3連単) probabilities

        Args:
            position_probs: Position probabilities (6, 6) - [boat_no, position]

        Returns:
            List of 120 trifecta bet information (6 × 5 × 4)

        Raises:
            ValidationError: If position_probs is invalid
        """
        validate_position_probs(position_probs)

        trifecta_bets = []

        for first in range(6):
            p_first = position_probs[first, 0]  # 1st place probability

            for second in range(6):
                if first == second:
                    continue

                # P(second=2nd | first=1st)
                p_second_given_first = position_probs[second, 1] / max(
                    1 - position_probs[second, 0], 0.01
                )

                for third in range(6):
                    if third == first or third == second:
                        continue

                    # P(third=3rd | first=1st, second=2nd)
                    # Approximate as P(third=3rd) / (1 - P(third=1st) - P(third=2nd))
                    p_third = position_probs[third, 2]
                    p_third_not_top2 = max(
                        1 - position_probs[third, 0] - position_probs[third, 1],
                        0.01
                    )
                    p_third_given = p_third / p_third_not_top2

                    # Trifecta probability
                    p_trifecta = p_first * p_second_given_first * p_third_given

                    trifecta_bets.append(TrifectaBet(
                        first=first + 1,  # 1-indexed
                        second=second + 1,
                        third=third + 1,
                        probability=p_trifecta,
                    ))

        # Sort by probability in descending order
        trifecta_bets.sort(key=lambda x: x.probability, reverse=True)

        return trifecta_bets

    def calculate_trifecta_expected_values(
        self,
        trifecta_bets: List[TrifectaBet],
        odds: dict,
    ) -> List[TrifectaBet]:
        """
        Calculate expected values for trifecta bets

        Args:
            trifecta_bets: List of trifecta bet information
            odds: Odds dictionary {(first, second, third): odds}

        Returns:
            List of trifecta bet information with expected values added
        """
        for bet in trifecta_bets:
            key = (bet.first, bet.second, bet.third)
            if key in odds:
                bet.odds = odds[key]
                bet.expected_value = bet.probability * bet.odds

        return trifecta_bets

    def get_trifecta_value_bets(
        self,
        trifecta_bets: List[TrifectaBet],
        threshold: float = 1.0,
    ) -> List[TrifectaBet]:
        """
        Get trifecta bets with expected value exceeding threshold

        Args:
            trifecta_bets: List of trifecta bet information
            threshold: Expected value threshold

        Returns:
            List of value bets
        """
        return [bet for bet in trifecta_bets if bet.expected_value > threshold]

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
