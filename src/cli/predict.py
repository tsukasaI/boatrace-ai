#!/usr/bin/env python
"""
Boatrace AI Prediction CLI

Usage:
    uv run python -m src.cli.predict --help
    uv run python -m src.cli.predict --date 20240115 --stadium 23 --race 1
    uv run python -m src.cli.predict --interactive
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import PROCESSED_DATA_DIR, STADIUM_CODES, PROJECT_ROOT
from src.models.train import BoatracePredictor
from src.models.predictor import RacePredictor, format_prediction_result, TrifectaBet
from src.models.features import FeatureEngineering, get_feature_columns
from src.betting.kelly import KellyCalculator


class PredictionCLI:
    """CLI for boat race predictions"""

    def __init__(self, bankroll: int = 100000, kelly_multiplier: float = 0.25):
        """
        Args:
            bankroll: Initial bankroll for Kelly sizing
            kelly_multiplier: Kelly fraction (default: quarter Kelly)
        """
        self.bankroll = bankroll
        self.kelly_calc = KellyCalculator(
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
        )
        self.feature_eng = FeatureEngineering()
        self.predictor = None
        self.programs_df = None
        self.payouts_df = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model"""
        try:
            model = BoatracePredictor()
            model.load()
            self.predictor = RacePredictor(model)
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Warning: Model not found. Using fallback predictor.")
            self.predictor = RacePredictor(model=None)

    def load_data(self) -> bool:
        """Load program and payout data"""
        try:
            self.programs_df = pd.read_csv(PROCESSED_DATA_DIR / "programs_entries.csv")
            self.payouts_df = pd.read_csv(PROCESSED_DATA_DIR / "payouts.csv")
            print(f"Loaded {len(self.programs_df)} program entries")
            return True
        except FileNotFoundError as e:
            print(f"Error: Data not found - {e}")
            print("Run: uv run python src/preprocessing/parser.py")
            return False

    def get_race_data(
        self,
        date: int,
        stadium_code: int,
        race_no: int,
    ) -> Optional[pd.DataFrame]:
        """Get race program data"""
        if self.programs_df is None:
            return None

        race_df = self.programs_df[
            (self.programs_df["date"] == date) &
            (self.programs_df["stadium_code"] == stadium_code) &
            (self.programs_df["race_no"] == race_no)
        ]

        if len(race_df) != 6:
            return None

        return race_df

    def get_race_odds(
        self,
        date: int,
        stadium_code: int,
        race_no: int,
        bet_type: str = "exacta",
    ) -> dict:
        """Get actual odds from payout data"""
        if self.payouts_df is None:
            return {}

        race_payouts = self.payouts_df[
            (self.payouts_df["date"] == date) &
            (self.payouts_df["stadium_code"] == stadium_code) &
            (self.payouts_df["race_no"] == race_no) &
            (self.payouts_df["bet_type"] == bet_type)
        ]

        odds = {}
        for _, row in race_payouts.iterrows():
            if bet_type == "exacta":
                odds[(int(row["first"]), int(row["second"]))] = row["odds"]
            elif bet_type == "trifecta":
                odds[(int(row["first"]), int(row["second"]), int(row["third"]))] = row["odds"]

        return odds

    def predict_race(
        self,
        date: int,
        stadium_code: int,
        race_no: int,
        include_trifecta: bool = False,
        top_n: int = 10,
    ) -> None:
        """Predict and display race results"""
        # Get race data
        race_df = self.get_race_data(date, stadium_code, race_no)
        if race_df is None:
            print(f"Error: Race not found (date={date}, stadium={stadium_code}, race={race_no})")
            return

        stadium_name = STADIUM_CODES.get(stadium_code, f"Stadium {stadium_code}")
        print(f"\n{'=' * 60}")
        print(f"Race Prediction: {stadium_name} R{race_no} ({date})")
        print("=" * 60)

        # Display entries
        print("\nEntries:")
        print("-" * 50)
        for _, row in race_df.iterrows():
            print(f"  {int(row['boat_no'])}. {row['racer_name']} ({row['racer_class']}) "
                  f"Win: {row['national_win_rate']:.2f} Motor: {row['motor_in2_rate']:.1f}%")

        # Create features
        features = self.feature_eng.create_base_features(race_df)
        features = self.feature_eng.create_relative_features(features)

        # Get feature columns
        feature_cols = get_feature_columns()
        base_cols = [c for c in feature_cols
                     if not c.startswith("recent_")
                     and not c.startswith("local_recent")
                     and c != "course_win_rate"]
        available_cols = [c for c in base_cols if c in features.columns]
        X = features[available_cols].values
        X = np.nan_to_num(X, nan=0.0)

        # Predict
        position_probs = self.predictor.predict_positions(X)

        # Exacta
        exacta_bets = self.predictor.calculate_exacta_probabilities(position_probs)
        odds = self.get_race_odds(date, stadium_code, race_no, "exacta")
        exacta_bets = self.predictor.calculate_expected_values(exacta_bets, odds)

        # Display results
        print("\n" + format_prediction_result(position_probs, exacta_bets, top_n))

        # Kelly recommendations
        value_bets = self.predictor.get_value_bets(exacta_bets, threshold=1.0)
        if value_bets:
            print("\n=== Kelly Bet Sizing ===")
            print(f"Bankroll: ¥{self.bankroll:,}")
            print("-" * 50)

            for bet in value_bets[:5]:
                sizing = self.kelly_calc.calculate_single(bet.probability, bet.odds)
                if sizing.stake > 0:
                    print(f"  {bet.first}-{bet.second}: "
                          f"Prob={bet.probability:.1%}, Odds={bet.odds:.1f}, "
                          f"EV={bet.expected_value:.2f} -> Stake: ¥{sizing.stake:,}")

        # Trifecta
        if include_trifecta:
            print("\n=== Trifecta TOP10 ===")
            trifecta_bets = self.predictor.calculate_trifecta_probabilities(position_probs)
            trifecta_odds = self.get_race_odds(date, stadium_code, race_no, "trifecta")
            trifecta_bets = self.predictor.calculate_trifecta_expected_values(
                trifecta_bets, trifecta_odds
            )

            print("Rank  Combination  Prob    Odds      EV")
            print("-" * 50)
            for i, bet in enumerate(trifecta_bets[:10]):
                odds_str = f"{bet.odds:.1f}" if bet.odds > 0 else "-"
                ev_str = f"{bet.expected_value:.2f}" if bet.expected_value > 0 else "-"
                print(f" {i+1:2}    {bet.first}-{bet.second}-{bet.third}    "
                      f"{bet.probability:.2%}   {odds_str:>8}  {ev_str:>6}")

            # Trifecta value bets
            trifecta_value = self.predictor.get_trifecta_value_bets(trifecta_bets, threshold=1.0)
            if trifecta_value:
                print("\n=== Trifecta Value Bets ===")
                for bet in trifecta_value[:5]:
                    sizing = self.kelly_calc.calculate_single(bet.probability, bet.odds)
                    if sizing.stake > 0:
                        print(f"  {bet.first}-{bet.second}-{bet.third}: "
                              f"EV={bet.expected_value:.2f} -> Stake: ¥{sizing.stake:,}")

    def list_available_races(self, date: int) -> None:
        """List available races for a date"""
        if self.programs_df is None:
            print("No data loaded")
            return

        races = self.programs_df[self.programs_df["date"] == date]
        if len(races) == 0:
            print(f"No races found for {date}")
            return

        stadiums = races.groupby("stadium_code")["race_no"].nunique().to_dict()

        print(f"\nAvailable races for {date}:")
        print("-" * 40)
        for code, num_races in sorted(stadiums.items()):
            name = STADIUM_CODES.get(code, f"Stadium {code}")
            print(f"  {code:2}: {name} ({num_races} races)")

    def interactive_mode(self) -> None:
        """Interactive prediction mode"""
        print("\n" + "=" * 60)
        print("Boatrace AI - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  predict <date> <stadium> <race>  - Predict a race")
        print("  list <date>                      - List available races")
        print("  bankroll <amount>                - Set bankroll")
        print("  trifecta on/off                  - Toggle trifecta predictions")
        print("  quit                             - Exit")
        print("-" * 60)

        include_trifecta = False

        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0]

            if action == "quit" or action == "exit" or action == "q":
                print("Goodbye!")
                break

            elif action == "predict" and len(parts) >= 4:
                try:
                    date = int(parts[1])
                    stadium = int(parts[2])
                    race = int(parts[3])
                    self.predict_race(date, stadium, race, include_trifecta)
                except ValueError:
                    print("Error: Invalid arguments. Use: predict <date> <stadium> <race>")

            elif action == "list" and len(parts) >= 2:
                try:
                    date = int(parts[1])
                    self.list_available_races(date)
                except ValueError:
                    print("Error: Invalid date")

            elif action == "bankroll" and len(parts) >= 2:
                try:
                    self.bankroll = int(parts[1])
                    self.kelly_calc = KellyCalculator(
                        bankroll=self.bankroll,
                        kelly_multiplier=self.kelly_calc.kelly_multiplier,
                    )
                    print(f"Bankroll set to ¥{self.bankroll:,}")
                except ValueError:
                    print("Error: Invalid amount")

            elif action == "trifecta":
                if len(parts) >= 2:
                    include_trifecta = parts[1] in ("on", "true", "1", "yes")
                else:
                    include_trifecta = not include_trifecta
                print(f"Trifecta predictions: {'ON' if include_trifecta else 'OFF'}")

            else:
                print("Unknown command. Type 'quit' to exit.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Boatrace AI Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive
  %(prog)s --date 20240115 --stadium 23 --race 1
  %(prog)s --date 20240115 --stadium 23 --race 1 --trifecta
  %(prog)s --list 20240115
        """
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--date", "-d",
        type=int,
        help="Race date (YYYYMMDD format)"
    )
    parser.add_argument(
        "--stadium", "-s",
        type=int,
        help="Stadium code (1-24)"
    )
    parser.add_argument(
        "--race", "-r",
        type=int,
        help="Race number (1-12)"
    )
    parser.add_argument(
        "--trifecta", "-t",
        action="store_true",
        help="Include trifecta predictions"
    )
    parser.add_argument(
        "--bankroll", "-b",
        type=int,
        default=100000,
        help="Bankroll for Kelly sizing (default: 100000)"
    )
    parser.add_argument(
        "--kelly",
        type=float,
        default=0.25,
        help="Kelly multiplier (default: 0.25 = quarter Kelly)"
    )
    parser.add_argument(
        "--list", "-l",
        type=int,
        dest="list_date",
        help="List available races for date"
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = PredictionCLI(
        bankroll=args.bankroll,
        kelly_multiplier=args.kelly,
    )

    # Load data
    if not cli.load_data():
        return

    # Run appropriate mode
    if args.interactive:
        cli.interactive_mode()
    elif args.list_date:
        cli.list_available_races(args.list_date)
    elif args.date and args.stadium and args.race:
        cli.predict_race(
            args.date,
            args.stadium,
            args.race,
            include_trifecta=args.trifecta,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
