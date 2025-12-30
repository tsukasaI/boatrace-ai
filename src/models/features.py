"""
Feature Engineering

Generate features from racer, motor, boat statistics and past performance
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineering:
    """Feature generation class"""

    # Class rank encoding
    CLASS_ENCODING = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

    def __init__(self, n_recent_races: int = 30):
        """
        Args:
            n_recent_races: Number of races to use for past performance calculation
        """
        self.n_recent_races = n_recent_races

    def create_base_features(self, programs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate base features from program data

        Args:
            programs_df: Program entry data

        Returns:
            DataFrame of base features
        """
        df = programs_df.copy()

        # Encode class rank
        df["class_encoded"] = df["racer_class"].map(self.CLASS_ENCODING).fillna(0)

        # Base features
        features = df[[
            "date", "stadium_code", "race_no", "boat_no", "racer_id",
            # Racer features
            "national_win_rate", "national_in2_rate",
            "local_win_rate", "local_in2_rate",
            "age", "weight", "class_encoded",
            # Equipment features
            "motor_no", "motor_in2_rate",
            "boat_no_equip", "boat_in2_rate",
        ]].copy()

        return features

    def create_historical_features(
        self,
        programs_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate historical features from past performance

        Args:
            programs_df: Program entry data
            results_df: Race results data

        Returns:
            DataFrame of historical features
        """
        # Sort results data by date
        results = results_df.sort_values("date").copy()

        # Aggregate past performance by racer_id
        historical_features = []

        for (date, stadium, race_no), group in programs_df.groupby(
            ["date", "stadium_code", "race_no"]
        ):
            # Get results before this race
            past_results = results[results["date"] < date]

            for _, row in group.iterrows():
                racer_id = row["racer_id"]

                # This racer's past performance
                racer_results = past_results[
                    past_results["racer_id"] == racer_id
                ].tail(self.n_recent_races)

                # Statistics of past performance
                if len(racer_results) > 0:
                    recent_win_rate = (racer_results["rank"] == 1).mean()
                    recent_in2_rate = (racer_results["rank"] <= 2).mean()
                    recent_in3_rate = (racer_results["rank"] <= 3).mean()
                    avg_rank = racer_results["rank"].mean()
                    avg_start_timing = racer_results["start_timing"].mean()
                    race_count = len(racer_results)
                else:
                    recent_win_rate = 0.0
                    recent_in2_rate = 0.0
                    recent_in3_rate = 0.0
                    avg_rank = 3.5  # Median value
                    avg_start_timing = 0.15  # Average ST
                    race_count = 0

                # Past performance at same stadium
                local_results = past_results[
                    (past_results["racer_id"] == racer_id) &
                    (past_results["stadium_code"] == stadium)
                ].tail(self.n_recent_races)

                if len(local_results) > 0:
                    local_recent_win_rate = (local_results["rank"] == 1).mean()
                    local_race_count = len(local_results)
                else:
                    local_recent_win_rate = 0.0
                    local_race_count = 0

                # Win rate by course (entry course)
                course_results = past_results[
                    (past_results["racer_id"] == racer_id) &
                    (past_results["course"] == row["boat_no"])  # Assuming lane number = course
                ].tail(self.n_recent_races)

                if len(course_results) > 0:
                    course_win_rate = (course_results["rank"] == 1).mean()
                else:
                    course_win_rate = 0.0

                historical_features.append({
                    "date": date,
                    "stadium_code": stadium,
                    "race_no": race_no,
                    "boat_no": row["boat_no"],
                    "racer_id": racer_id,
                    # Historical features
                    "recent_win_rate": recent_win_rate,
                    "recent_in2_rate": recent_in2_rate,
                    "recent_in3_rate": recent_in3_rate,
                    "recent_avg_rank": avg_rank,
                    "recent_avg_st": avg_start_timing,
                    "recent_race_count": race_count,
                    "local_recent_win_rate": local_recent_win_rate,
                    "local_race_count": local_race_count,
                    "course_win_rate": course_win_rate,
                })

        return pd.DataFrame(historical_features)

    def create_relative_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate relative features within a race

        Args:
            features_df: Features DataFrame

        Returns:
            DataFrame with relative features added
        """
        df = features_df.copy()

        # Group by race
        group_cols = ["date", "stadium_code", "race_no"]

        # Win rate rank (1=highest)
        df["win_rate_rank"] = df.groupby(group_cols)["national_win_rate"].rank(
            ascending=False, method="min"
        )

        # Difference between win rate and race average
        race_avg_win_rate = df.groupby(group_cols)["national_win_rate"].transform("mean")
        df["win_rate_diff_from_avg"] = df["national_win_rate"] - race_avg_win_rate

        # Motor top-2 rate rank
        df["motor_rate_rank"] = df.groupby(group_cols)["motor_in2_rate"].rank(
            ascending=False, method="min"
        )

        # Boat top-2 rate rank
        df["boat_rate_rank"] = df.groupby(group_cols)["boat_in2_rate"].rank(
            ascending=False, method="min"
        )

        # Lane advantage/disadvantage (lane 1 is advantageous)
        # Typical lane 1 win rate is about 55%, lane 6 is about 5%
        course_advantage = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.03}
        df["course_advantage"] = df["boat_no"].map(course_advantage)

        return df

    def create_all_features(
        self,
        programs_df: pd.DataFrame,
        results_df: pd.DataFrame,
        include_historical: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all features

        Args:
            programs_df: Program entry data
            results_df: Race results data
            include_historical: Whether to include historical features

        Returns:
            DataFrame of all features
        """
        # Base features
        features = self.create_base_features(programs_df)

        # Historical features
        if include_historical and results_df is not None:
            historical = self.create_historical_features(programs_df, results_df)
            features = features.merge(
                historical,
                on=["date", "stadium_code", "race_no", "boat_no", "racer_id"],
                how="left",
            )

        # Relative features
        features = self.create_relative_features(features)

        return features


def get_feature_columns() -> list[str]:
    """List of feature column names to input to the model"""
    return [
        # Base features
        "national_win_rate", "national_in2_rate",
        "local_win_rate", "local_in2_rate",
        "age", "weight", "class_encoded",
        "motor_in2_rate", "boat_in2_rate",
        # Historical features
        "recent_win_rate", "recent_in2_rate", "recent_in3_rate",
        "recent_avg_rank", "recent_avg_st", "recent_race_count",
        "local_recent_win_rate", "local_race_count",
        "course_win_rate",
        # Relative features
        "win_rate_rank", "win_rate_diff_from_avg",
        "motor_rate_rank", "boat_rate_rank",
        "course_advantage",
    ]
