"""
Backtest Simulator

Validate profitability of EV>1.0 strategy using historical data
"""

import sys
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.models.train import BoatracePredictor
from src.models.predictor import RacePredictor, ExactaBet
from src.models.features import FeatureEngineering, get_feature_columns
from src.backtesting.metrics import calculate_metrics, BacktestMetrics
from src.backtesting.synthetic_odds import SyntheticOddsGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BetRecord:
    """Individual bet record"""
    date: int
    stadium_code: int
    race_no: int
    first: int
    second: int
    probability: float
    odds: float
    expected_value: float
    stake: int
    actual_first: int
    actual_second: int
    won: bool
    profit: int  # Profit (win: payout - stake, loss: -stake)


@dataclass
class BacktestResult:
    """Backtest result"""
    bets: List[BetRecord] = field(default_factory=list)
    total_races: int = 0
    races_with_bets: int = 0
    total_stake: int = 0
    total_payout: int = 0
    metrics: Optional[BacktestMetrics] = None

    @property
    def total_profit(self) -> int:
        return self.total_payout - self.total_stake

    @property
    def roi(self) -> float:
        if self.total_stake == 0:
            return 0.0
        return self.total_profit / self.total_stake


class BacktestSimulator:
    """Backtest simulator"""

    def __init__(
        self,
        model: BoatracePredictor = None,
        ev_threshold: float = 1.0,
        stake: int = 100,
        max_bets_per_race: int = 3,
        use_synthetic_odds: bool = False,
    ):
        """
        Args:
            model: Prediction model
            ev_threshold: Expected value threshold
            stake: Stake amount per bet (yen)
            max_bets_per_race: Maximum number of bets per race
            use_synthetic_odds: Whether to use synthetic odds
        """
        self.predictor = RacePredictor(model)
        self.ev_threshold = ev_threshold
        self.stake = stake
        self.max_bets_per_race = max_bets_per_race
        self.feature_eng = FeatureEngineering()
        self.use_synthetic_odds = use_synthetic_odds
        self.synthetic_odds_gen = SyntheticOddsGenerator() if use_synthetic_odds else None

    def load_data(
        self,
        data_dir: Path = None,
        test_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data

        Args:
            data_dir: Data directory
            test_only: Use only test data

        Returns:
            (programs_df, results_df, payouts_df)
        """
        data_dir = data_dir or PROCESSED_DATA_DIR

        programs_df = pd.read_csv(data_dir / "programs_entries.csv")
        results_df = pd.read_csv(data_dir / "results_entries.csv")
        payouts_df = pd.read_csv(data_dir / "payouts.csv")

        # Test data only (second half of 2024)
        if test_only:
            test_start = 20240701
            programs_df = programs_df[programs_df["date"] >= test_start]
            results_df = results_df[results_df["date"] >= test_start]
            payouts_df = payouts_df[payouts_df["date"] >= test_start]

        return programs_df, results_df, payouts_df

    def run(
        self,
        programs_df: pd.DataFrame,
        results_df: pd.DataFrame,
        payouts_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest

        Args:
            programs_df: Race program data
            results_df: Race results data
            payouts_df: Payout data

        Returns:
            Backtest result
        """
        result = BacktestResult()

        # Group by race
        race_groups = programs_df.groupby(["date", "stadium_code", "race_no"])

        # Feature columns
        feature_cols = get_feature_columns()
        # Exclude historical features (simple version)
        base_cols = [c for c in feature_cols if not c.startswith("recent_") and not c.startswith("local_recent") and c != "course_win_rate"]

        for (date, stadium, race_no), race_df in tqdm(
            race_groups, desc="Backtesting", unit="race"
        ):
            result.total_races += 1

            if len(race_df) != 6:
                continue

            # Get results
            race_results = results_df[
                (results_df["date"] == date) &
                (results_df["stadium_code"] == stadium) &
                (results_df["race_no"] == race_no)
            ]
            if len(race_results) != 6:
                continue

            # Get actual 1st and 2nd place
            first_place = race_results[race_results["rank"] == 1]
            second_place = race_results[race_results["rank"] == 2]
            if len(first_place) == 0 or len(second_place) == 0:
                continue

            actual_first = int(first_place["boat_no"].values[0])
            actual_second = int(second_place["boat_no"].values[0])

            # Get odds
            if self.use_synthetic_odds:
                # Use synthetic odds
                odds = self.synthetic_odds_gen.get_all_odds()
            else:
                # Use real odds (from payout data)
                race_payouts = payouts_df[
                    (payouts_df["date"] == date) &
                    (payouts_df["stadium_code"] == stadium) &
                    (payouts_df["race_no"] == race_no) &
                    (payouts_df["bet_type"] == "exacta")
                ]
                if len(race_payouts) == 0:
                    continue

                # Create odds dictionary
                odds = {}
                for _, row in race_payouts.iterrows():
                    odds[(int(row["first"]), int(row["second"]))] = row["odds"]

            # Generate features
            features = self.feature_eng.create_base_features(race_df)
            features = self.feature_eng.create_relative_features(features)

            # Extract only available features
            available_cols = [c for c in base_cols if c in features.columns]
            X = features[available_cols].values

            # Fill missing values with zeros
            X = np.nan_to_num(X, nan=0.0)

            # Predict
            try:
                position_probs = self.predictor.predict_positions(X)
            except Exception as e:
                logger.debug(f"Prediction error: {e}")
                continue

            # Calculate exacta probabilities
            exacta_bets = self.predictor.calculate_exacta_probabilities(position_probs)

            # Calculate expected values
            exacta_bets = self.predictor.calculate_expected_values(exacta_bets, odds)

            # Filter bets with EV > threshold
            value_bets = [b for b in exacta_bets if b.expected_value > self.ev_threshold]

            if not value_bets:
                continue

            # Limit to top N
            value_bets = value_bets[:self.max_bets_per_race]

            result.races_with_bets += 1

            # Execute bets
            for bet in value_bets:
                won = (bet.first == actual_first and bet.second == actual_second)
                payout = int(bet.odds * self.stake) if won else 0
                profit = payout - self.stake

                record = BetRecord(
                    date=date,
                    stadium_code=stadium,
                    race_no=race_no,
                    first=bet.first,
                    second=bet.second,
                    probability=bet.probability,
                    odds=bet.odds,
                    expected_value=bet.expected_value,
                    stake=self.stake,
                    actual_first=actual_first,
                    actual_second=actual_second,
                    won=won,
                    profit=profit,
                )
                result.bets.append(record)
                result.total_stake += self.stake
                result.total_payout += payout

        # Calculate metrics
        result.metrics = calculate_metrics(result)

        return result

    def print_summary(self, result: BacktestResult) -> None:
        """Output result summary"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"EV Threshold: {self.ev_threshold}")
        print(f"Stake per bet: ¥{self.stake}")
        print(f"Max bets per race: {self.max_bets_per_race}")
        print(f"Odds type: {'Synthetic' if self.use_synthetic_odds else 'Real'}")
        print("-" * 60)
        print(f"Total races: {result.total_races}")
        print(f"Races with bets: {result.races_with_bets}")
        print(f"Total bets: {len(result.bets)}")
        print(f"Winning bets: {sum(1 for b in result.bets if b.won)}")
        print("-" * 60)
        print(f"Total stake: ¥{result.total_stake:,}")
        print(f"Total payout: ¥{result.total_payout:,}")
        print(f"Total profit: ¥{result.total_profit:,}")
        print(f"ROI: {result.roi:.1%}")

        if result.metrics:
            print("-" * 60)
            print(f"Hit rate: {result.metrics.hit_rate:.1%}")
            print(f"Average EV: {result.metrics.avg_ev:.2f}")
            print(f"Profit factor: {result.metrics.profit_factor:.2f}")
            print(f"Max drawdown: ¥{result.metrics.max_drawdown:,}")

        print("=" * 60)


def main():
    """Main processing"""
    parser = argparse.ArgumentParser(description="Run backtest simulation")
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help="EV threshold for betting (default: 1.0)"
    )
    parser.add_argument(
        "--stake", type=int, default=100,
        help="Stake per bet in yen (default: 100)"
    )
    parser.add_argument(
        "--max-bets", type=int, default=3,
        help="Max bets per race (default: 3)"
    )
    parser.add_argument(
        "--all-data", action="store_true",
        help="Use all data, not just test set"
    )
    parser.add_argument(
        "--synthetic-odds", action="store_true",
        help="Use synthetic odds based on historical rates"
    )
    args = parser.parse_args()

    # Load model
    logger.info("Loading model...")
    model = BoatracePredictor()
    try:
        model.load()
    except FileNotFoundError:
        logger.error("Model not found. Please train the model first.")
        logger.info("Run: uv run python src/models/train.py")
        return

    # Initialize simulator
    simulator = BacktestSimulator(
        model=model,
        ev_threshold=args.threshold,
        stake=args.stake,
        max_bets_per_race=args.max_bets,
        use_synthetic_odds=args.synthetic_odds,
    )

    if args.synthetic_odds:
        logger.info("Using synthetic odds based on historical rates")

    # Load data
    logger.info("Loading data...")
    try:
        programs_df, results_df, payouts_df = simulator.load_data(
            test_only=not args.all_data
        )
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Run: uv run python src/preprocessing/parser.py")
        return

    logger.info(f"Loaded {len(programs_df)} program entries")
    logger.info(f"Loaded {len(payouts_df)} payout records")

    # Run backtest
    logger.info("Running backtest...")
    result = simulator.run(programs_df, results_df, payouts_df)

    # Output results
    simulator.print_summary(result)

    # Save results to CSV
    if result.bets:
        bets_df = pd.DataFrame([b.__dict__ for b in result.bets])
        output_path = PROJECT_ROOT / "results" / "backtest_bets.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bets_df.to_csv(output_path, index=False)
        logger.info(f"Saved bet records to {output_path}")


if __name__ == "__main__":
    main()
