"""
Feature Engineering

Generate features from racer, motor, boat statistics and past performance
"""

import numpy as np
import pandas as pd


class FeatureEngineering:
    """Feature generation class"""

    # Class rank encoding
    CLASS_ENCODING = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

    # Branch/Region encoding (grouped by geography)
    BRANCH_ENCODING = {
        # Kanto
        "群馬": 1, "埼玉": 1, "東京": 1,
        # Tokai
        "静岡": 2, "愛知": 2, "三重": 2,
        # Kinki
        "滋賀": 3, "大阪": 3, "兵庫": 3,
        # Chugoku/Shikoku
        "岡山": 4, "広島": 4, "山口": 4, "徳島": 4, "香川": 4,
        # Kyushu
        "福岡": 5, "佐賀": 5, "長崎": 5, "大分": 5,
    }

    # Race type encoding
    RACE_TYPE_ENCODING = {
        "予選": 1,
        "一般": 1,
        "特選": 2,
        "選抜": 2,
        "準優": 3,
        "準優勝戦": 3,
        "優勝戦": 4,
        "優": 4,
    }

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

        # Encode branch/region
        if "branch" in df.columns:
            df["branch_encoded"] = df["branch"].map(self.BRANCH_ENCODING).fillna(0)
        else:
            df["branch_encoded"] = 0

        # Base feature columns
        base_cols = [
            "date", "stadium_code", "race_no", "boat_no", "racer_id",
            # Racer features
            "national_win_rate", "national_in2_rate",
            "local_win_rate", "local_in2_rate",
            "age", "weight", "class_encoded", "branch_encoded",
            # Equipment features
            "motor_no", "motor_in2_rate",
            "boat_no_equip", "boat_in2_rate",
        ]

        # Include exhibition_time if available (from merged results data)
        if "exhibition_time" in df.columns:
            base_cols.append("exhibition_time")

        # Include race_type if available
        if "race_type" in df.columns:
            base_cols.append("race_type")

        features = df[base_cols].copy()

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

        # Pre-compute course difference (actual course - boat_no)
        if "course" in results.columns and "boat_no" in results.columns:
            results["course_diff"] = results["course"] - results["boat_no"]

        # Aggregate past performance by racer_id
        historical_features = []

        for (date, stadium, race_no), group in programs_df.groupby(
            ["date", "stadium_code", "race_no"]
        ):
            # Get results before this race
            past_results = results[results["date"] < date]

            for _, row in group.iterrows():
                racer_id = row["racer_id"]
                boat_no = row["boat_no"]

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

                    # Start timing features
                    st_values = racer_results["start_timing"].dropna()
                    if len(st_values) > 1:
                        st_std = st_values.std()
                        flying_start_rate = (st_values < 0).mean()  # Negative = flying
                        late_start_rate = (st_values > 0.20).mean()  # Late start
                    else:
                        st_std = 0.05  # Default
                        flying_start_rate = 0.0
                        late_start_rate = 0.0

                    # Course-taking features
                    if "course_diff" in racer_results.columns:
                        # Average course difference (negative = takes inside)
                        avg_course_diff = racer_results["course_diff"].mean()
                        # Rate of taking inside course (course < boat_no)
                        inside_take_rate = (racer_results["course_diff"] < 0).mean()
                    else:
                        avg_course_diff = 0.0
                        inside_take_rate = 0.0
                else:
                    recent_win_rate = 0.0
                    recent_in2_rate = 0.0
                    recent_in3_rate = 0.0
                    avg_rank = 3.5  # Median value
                    avg_start_timing = 0.15  # Average ST
                    race_count = 0
                    st_std = 0.05
                    flying_start_rate = 0.0
                    late_start_rate = 0.0
                    avg_course_diff = 0.0
                    inside_take_rate = 0.0

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
                    (past_results["course"] == boat_no)
                ].tail(self.n_recent_races)

                if len(course_results) > 0:
                    course_win_rate = (course_results["rank"] == 1).mean()
                    course_in2_rate = (course_results["rank"] <= 2).mean()
                else:
                    course_win_rate = 0.0
                    course_in2_rate = 0.0

                # Recent form (weighted: more recent = higher weight)
                if len(racer_results) >= 5:
                    recent_5 = racer_results.tail(5)
                    weighted_recent_win = (recent_5["rank"] == 1).mean()
                else:
                    weighted_recent_win = recent_win_rate

                historical_features.append({
                    "date": date,
                    "stadium_code": stadium,
                    "race_no": race_no,
                    "boat_no": boat_no,
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
                    # New features
                    "course_in2_rate": course_in2_rate,
                    "st_consistency": st_std,
                    "flying_start_rate": flying_start_rate,
                    "late_start_rate": late_start_rate,
                    "avg_course_diff": avg_course_diff,
                    "inside_take_rate": inside_take_rate,
                    "weighted_recent_win": weighted_recent_win,
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

        # Exhibition time features (if available)
        if "exhibition_time" in df.columns:
            # Exhibition time rank (lower is better, 1=fastest)
            df["exhibition_time_rank"] = df.groupby(group_cols)["exhibition_time"].rank(
                ascending=True, method="min"
            )

            # Difference from race average
            race_avg_exhibition = df.groupby(group_cols)["exhibition_time"].transform("mean")
            df["exhibition_time_diff"] = df["exhibition_time"] - race_avg_exhibition

        # Race context features
        if "race_type" in df.columns:
            # Encode race type (予選=1, 準優=3, 優勝戦=4)
            df["race_grade"] = df["race_type"].apply(self._encode_race_type)
            df["is_final"] = (df["race_grade"] >= 3).astype(int)
        else:
            df["race_grade"] = 1
            df["is_final"] = 0

        # Interaction features

        # Class × Course: High class racer in inside course is very strong
        df["class_x_course"] = df["class_encoded"] * df["course_advantage"]

        # Motor × Exhibition: Good motor + fast exhibition = strong signal
        if "exhibition_time" in df.columns:
            # Normalize exhibition time (inverse, lower is better)
            df["exhibition_score"] = 7.0 - df["exhibition_time"].clip(6.5, 7.5)
            df["motor_x_exhibition"] = df["motor_in2_rate"] * df["exhibition_score"] / 100
        else:
            df["motor_x_exhibition"] = 0.0

        # Equipment combined score
        df["equipment_score"] = (df["motor_in2_rate"] + df["boat_in2_rate"]) / 2

        # Equipment rank in race
        df["equipment_rank"] = df.groupby(group_cols)["equipment_score"].rank(
            ascending=False, method="min"
        )

        # Strong favorite indicator (best class + best equipment + inside course)
        df["favorite_score"] = (
            df["class_encoded"] / 4 +
            (7 - df["win_rate_rank"]) / 6 +
            (7 - df["equipment_rank"]) / 6 +
            df["course_advantage"]
        ) / 4

        # Upset potential (high class but outside course)
        df["upset_potential"] = df["class_encoded"] * (1 - df["course_advantage"])

        return df

    def _encode_race_type(self, race_type: str) -> int:
        """Encode race type string to numeric value."""
        if pd.isna(race_type):
            return 1
        race_type_str = str(race_type)
        for key, value in self.RACE_TYPE_ENCODING.items():
            if key in race_type_str:
                return value
        return 1  # Default to qualifying race

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
        "age", "weight", "class_encoded", "branch_encoded",
        "motor_in2_rate", "boat_in2_rate",
        # Historical features
        "recent_win_rate", "recent_in2_rate", "recent_in3_rate",
        "recent_avg_rank", "recent_avg_st", "recent_race_count",
        "local_recent_win_rate", "local_race_count",
        "course_win_rate", "course_in2_rate",
        # New historical features
        "st_consistency", "flying_start_rate", "late_start_rate",
        "avg_course_diff", "inside_take_rate",
        "weighted_recent_win",
        # Relative features
        "win_rate_rank", "win_rate_diff_from_avg",
        "motor_rate_rank", "boat_rate_rank",
        "course_advantage",
        # Exhibition time features
        "exhibition_time",
        "exhibition_time_rank",
        "exhibition_time_diff",
        # Race context features
        "race_grade", "is_final",
        # Interaction features
        "class_x_course", "motor_x_exhibition",
        "equipment_score", "equipment_rank",
        "favorite_score", "upset_potential",
    ]
