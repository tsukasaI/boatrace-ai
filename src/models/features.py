"""
Feature Engineering

Generate features from racer, motor, boat statistics and past performance
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class StadiumCourseIndex:
    """
    Pre-computed per-stadium course statistics for O(1) lookup.

    Computes win rates and top-2 rates for each (stadium, course) combination
    from historical results data.
    """

    # Global average course win rates (fallback when no stadium-specific data)
    GLOBAL_COURSE_WIN_RATE = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.03}
    GLOBAL_COURSE_IN2_RATE = {1: 0.75, 2: 0.35, 3: 0.30, 4: 0.25, 5: 0.20, 6: 0.15}

    def __init__(self):
        # (stadium_code, course) -> (win_count, in2_count, race_count)
        self.stadium_course_stats: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        # (racer_id, stadium_code, course) -> (win_count, in2_count, race_count)
        self.racer_stadium_course_stats: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        # Global course stats for normalization
        self.global_course_stats: Dict[int, Tuple[int, int, int]] = {}

    def build_from_results(self, results_df: pd.DataFrame, before_date: Optional[int] = None):
        """
        Build the index from historical results data using vectorized operations.

        Args:
            results_df: Results DataFrame with columns: stadium_code, course, rank, racer_id
            before_date: Only use data before this date (for training/test split)
        """
        df = results_df
        if before_date is not None:
            df = df[df["date"] < before_date]

        # Filter valid courses and ranks
        df = df[(df["course"] >= 1) & (df["course"] <= 6)]
        df = df[(df["rank"] >= 1) & (df["rank"] <= 6)]

        # Pre-compute boolean columns for aggregation
        df = df.assign(
            is_win=(df["rank"] == 1).astype(int),
            is_in2=(df["rank"] <= 2).astype(int)
        )

        # Stadium-course stats using groupby (vectorized, fast)
        stadium_course_grouped = df.groupby(["stadium_code", "course"]).agg(
            wins=("is_win", "sum"),
            in2=("is_in2", "sum"),
            total=("rank", "count")
        )
        self.stadium_course_stats = {
            (int(idx[0]), int(idx[1])): (int(row["wins"]), int(row["in2"]), int(row["total"]))
            for idx, row in stadium_course_grouped.iterrows()
        }

        # Global course stats
        global_grouped = df.groupby("course").agg(
            wins=("is_win", "sum"),
            in2=("is_in2", "sum"),
            total=("rank", "count")
        )
        self.global_course_stats = {
            int(idx): (int(row["wins"]), int(row["in2"]), int(row["total"]))
            for idx, row in global_grouped.iterrows()
        }

        # Racer-stadium-course stats (this is the largest but still vectorized)
        racer_grouped = df.groupby(["racer_id", "stadium_code", "course"]).agg(
            wins=("is_win", "sum"),
            in2=("is_in2", "sum"),
            total=("rank", "count")
        )
        self.racer_stadium_course_stats = {
            (int(idx[0]), int(idx[1]), int(idx[2])): (int(row["wins"]), int(row["in2"]), int(row["total"]))
            for idx, row in racer_grouped.iterrows()
        }

    def get_stadium_course_rates(self, stadium: int, course: int) -> Tuple[float, float]:
        """
        Get win rate and in2 rate for a stadium-course combination.

        Args:
            stadium: Stadium code (1-24)
            course: Course number (1-6)

        Returns:
            (win_rate, in2_rate)
        """
        key = (stadium, course)
        if key in self.stadium_course_stats:
            wins, in2, total = self.stadium_course_stats[key]
            if total >= 10:  # Minimum sample size
                return wins / total, in2 / total

        # Fallback to global average
        return self.GLOBAL_COURSE_WIN_RATE.get(course, 0.1), \
               self.GLOBAL_COURSE_IN2_RATE.get(course, 0.3)

    def get_global_course_rates(self, course: int) -> Tuple[float, float]:
        """Get global win rate and in2 rate for a course."""
        if course in self.global_course_stats:
            wins, in2, total = self.global_course_stats[course]
            if total > 0:
                return wins / total, in2 / total
        return self.GLOBAL_COURSE_WIN_RATE.get(course, 0.1), \
               self.GLOBAL_COURSE_IN2_RATE.get(course, 0.3)

    def get_advantage_diff(self, stadium: int, course: int) -> float:
        """
        Get the difference between stadium-specific and global course advantage.

        Positive value = this stadium favors this course more than average.
        """
        stadium_win, _ = self.get_stadium_course_rates(stadium, course)
        global_win, _ = self.get_global_course_rates(course)
        return stadium_win - global_win

    def get_racer_stadium_course_rates(
        self, racer_id: int, stadium: int, course: int
    ) -> Tuple[float, float, int]:
        """
        Get a racer's win rate and in2 rate for a specific stadium-course.

        Returns:
            (win_rate, in2_rate, race_count)
        """
        key = (racer_id, stadium, course)
        if key in self.racer_stadium_course_stats:
            wins, in2, total = self.racer_stadium_course_stats[key]
            if total >= 3:  # Lower threshold for racer-specific
                return wins / total, in2 / total, total

        # No data - return 0 to indicate unknown
        return 0.0, 0.0, 0


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

    # Weather condition encoding
    WEATHER_ENCODING = {
        "晴": 0,
        "曇り": 1,
        "曇": 1,
        "雨": 2,
        "雪": 3,
        "霧": 4,
    }

    # Wind direction to degrees (for circular encoding)
    WIND_DIRECTION_DEGREES = {
        "北": 0,
        "北東": 45,
        "東": 90,
        "南東": 135,
        "南": 180,
        "南西": 225,
        "西": 270,
        "北西": 315,
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

    def create_stadium_course_features(
        self,
        df: pd.DataFrame,
        stadium_course_index: StadiumCourseIndex,
    ) -> pd.DataFrame:
        """
        Generate per-stadium course advantage features using vectorized operations.

        Uses boat_no as the expected course (since actual course is unknown at prediction time).

        Features created:
        - stadium_course_win_rate: Win rate for this course at this stadium
        - stadium_course_in2_rate: Top-2 rate for this course at this stadium
        - stadium_course_advantage_diff: Difference from global course average
        - racer_course_win_at_stadium: Racer's win rate on this course at this stadium
        - racer_course_in2_at_stadium: Racer's top-2 rate on this course at this stadium

        Args:
            df: DataFrame with stadium_code, boat_no, racer_id
            stadium_course_index: Pre-computed stadium-course statistics

        Returns:
            DataFrame with stadium course features added
        """
        result = df.copy()

        # Convert stadium-course stats to DataFrame for merge
        if stadium_course_index.stadium_course_stats:
            sc_data = [
                {"stadium_code": k[0], "boat_no": k[1],
                 "sc_wins": v[0], "sc_in2": v[1], "sc_total": v[2]}
                for k, v in stadium_course_index.stadium_course_stats.items()
            ]
            sc_df = pd.DataFrame(sc_data)
            sc_df["stadium_course_win_rate"] = sc_df["sc_wins"] / sc_df["sc_total"]
            sc_df["stadium_course_in2_rate"] = sc_df["sc_in2"] / sc_df["sc_total"]
            sc_df = sc_df[["stadium_code", "boat_no", "stadium_course_win_rate", "stadium_course_in2_rate"]]

            result = result.merge(sc_df, on=["stadium_code", "boat_no"], how="left")
        else:
            result["stadium_course_win_rate"] = np.nan
            result["stadium_course_in2_rate"] = np.nan

        # Convert global stats to DataFrame
        if stadium_course_index.global_course_stats:
            gc_data = [
                {"boat_no": k, "gc_wins": v[0], "gc_total": v[2]}
                for k, v in stadium_course_index.global_course_stats.items()
            ]
            gc_df = pd.DataFrame(gc_data)
            gc_df["global_win_rate"] = gc_df["gc_wins"] / gc_df["gc_total"]
            gc_df = gc_df[["boat_no", "global_win_rate"]]

            result = result.merge(gc_df, on="boat_no", how="left")
            result["stadium_course_advantage_diff"] = (
                result["stadium_course_win_rate"].fillna(result["global_win_rate"])
                - result["global_win_rate"]
            )
            result = result.drop(columns=["global_win_rate"])
        else:
            result["stadium_course_advantage_diff"] = 0.0

        # Fill missing stadium-course rates with global defaults
        for course in range(1, 7):
            mask = (result["boat_no"] == course) & result["stadium_course_win_rate"].isna()
            result.loc[mask, "stadium_course_win_rate"] = stadium_course_index.GLOBAL_COURSE_WIN_RATE.get(course, 0.1)
            result.loc[mask, "stadium_course_in2_rate"] = stadium_course_index.GLOBAL_COURSE_IN2_RATE.get(course, 0.3)

        # Convert racer-stadium-course stats to DataFrame for merge
        if stadium_course_index.racer_stadium_course_stats:
            rsc_data = [
                {"racer_id": k[0], "stadium_code": k[1], "boat_no": k[2],
                 "rsc_wins": v[0], "rsc_in2": v[1], "rsc_total": v[2]}
                for k, v in stadium_course_index.racer_stadium_course_stats.items()
                if v[2] >= 3  # Minimum threshold
            ]
            if rsc_data:
                rsc_df = pd.DataFrame(rsc_data)
                rsc_df["racer_course_win_at_stadium"] = rsc_df["rsc_wins"] / rsc_df["rsc_total"]
                rsc_df["racer_course_in2_at_stadium"] = rsc_df["rsc_in2"] / rsc_df["rsc_total"]
                rsc_df = rsc_df[["racer_id", "stadium_code", "boat_no",
                                 "racer_course_win_at_stadium", "racer_course_in2_at_stadium"]]

                result = result.merge(rsc_df, on=["racer_id", "stadium_code", "boat_no"], how="left")
            else:
                result["racer_course_win_at_stadium"] = 0.0
                result["racer_course_in2_at_stadium"] = 0.0
        else:
            result["racer_course_win_at_stadium"] = 0.0
            result["racer_course_in2_at_stadium"] = 0.0

        # Fill NaN racer stats with 0 (no history)
        result["racer_course_win_at_stadium"] = result["racer_course_win_at_stadium"].fillna(0.0)
        result["racer_course_in2_at_stadium"] = result["racer_course_in2_at_stadium"].fillna(0.0)

        return result

    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate weather features from race-level weather data.

        Features created:
        - weather_encoded: 0=sunny, 1=cloudy, 2=rain, 3=snow, 4=fog
        - wind_speed: numeric (meters)
        - wave_height: numeric (cm)
        - wind_direction_sin: sin(direction_degrees)
        - wind_direction_cos: cos(direction_degrees)
        - wind_wave_interaction: wind_speed * wave_height (rough conditions)
        - inside_wave_penalty: wave_height * (7 - boat_no) / 6 (inside worse in waves)

        Args:
            df: DataFrame with weather columns and boat_no

        Returns:
            DataFrame with weather features added
        """
        result = df.copy()

        # Check if weather columns exist
        if "weather" not in result.columns:
            # No weather data - use defaults (moderate conditions)
            result["weather_encoded"] = 1.0  # Cloudy
            result["wind_speed"] = 0.0
            result["wave_height"] = 0.0
            result["wind_direction_sin"] = 0.0
            result["wind_direction_cos"] = 1.0  # North
            result["wind_wave_interaction"] = 0.0
            result["inside_wave_penalty"] = 0.0
            return result

        # Weather condition encoding
        result["weather_encoded"] = result["weather"].map(
            self.WEATHER_ENCODING
        ).fillna(1).astype(float)  # Default to cloudy

        # Wind direction to sin/cos (circular encoding)
        wind_degrees = result["wind_direction"].map(
            self.WIND_DIRECTION_DEGREES
        ).fillna(0)
        wind_radians = np.deg2rad(wind_degrees)
        result["wind_direction_sin"] = np.sin(wind_radians)
        result["wind_direction_cos"] = np.cos(wind_radians)

        # Numeric features
        result["wind_speed"] = pd.to_numeric(
            result["wind_speed"], errors="coerce"
        ).fillna(0).astype(float)
        result["wave_height"] = pd.to_numeric(
            result["wave_height"], errors="coerce"
        ).fillna(0).astype(float)

        # Interaction features
        # Wind × wave interaction (rough conditions indicator)
        result["wind_wave_interaction"] = (
            result["wind_speed"] * result["wave_height"] / 100.0
        )

        # Inside lanes are disadvantaged in high waves
        # boat_no 1 gets highest penalty, boat_no 6 gets lowest
        result["inside_wave_penalty"] = (
            result["wave_height"] * (7 - result["boat_no"]) / 600.0
        )

        return result

    def create_all_features(
        self,
        programs_df: pd.DataFrame,
        results_df: pd.DataFrame,
        include_historical: bool = True,
        weather_df: pd.DataFrame = None,
        stadium_course_index: StadiumCourseIndex = None,
    ) -> pd.DataFrame:
        """
        Generate all features

        Args:
            programs_df: Program entry data
            results_df: Race results data
            include_historical: Whether to include historical features
            weather_df: Race-level weather data (from results_races.csv)
            stadium_course_index: Pre-computed stadium-course statistics

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

        # Stadium course features (before relative features so we can use stadium-specific course_advantage)
        if stadium_course_index is not None:
            features = self.create_stadium_course_features(features, stadium_course_index)

        # Relative features
        features = self.create_relative_features(features)

        # Weather features (merge race-level weather data)
        if weather_df is not None:
            weather_cols = ["date", "stadium_code", "race_no",
                          "weather", "wind_direction", "wind_speed", "wave_height"]
            weather_subset = weather_df[
                [c for c in weather_cols if c in weather_df.columns]
            ].drop_duplicates()
            features = features.merge(
                weather_subset,
                on=["date", "stadium_code", "race_no"],
                how="left",
            )

        # Create weather features (even if no weather data - uses defaults)
        features = self.create_weather_features(features)

        return features


def get_feature_columns(include_stadium_course: bool = False) -> list[str]:
    """List of feature column names to input to the model.

    Args:
        include_stadium_course: If True, includes 53 features with stadium course.
                               If False (default), returns 50 features.

    Note:
        Stadium course features are disabled by default because M3 investigation
        showed they hurt model performance. The existing course_advantage feature
        already captures most stadium-specific patterns.
    """
    features = [
        # Base features (11)
        "stadium_code",  # Stadium-specific effects
        "national_win_rate", "national_in2_rate",
        "local_win_rate", "local_in2_rate",
        "age", "weight", "class_encoded", "branch_encoded",
        "motor_in2_rate", "boat_in2_rate",
        # Historical features (16)
        "recent_win_rate", "recent_in2_rate", "recent_in3_rate",
        "recent_avg_rank", "recent_avg_st", "recent_race_count",
        "local_recent_win_rate", "local_race_count",
        "course_win_rate", "course_in2_rate",
        # New historical features
        "st_consistency", "flying_start_rate", "late_start_rate",
        "avg_course_diff", "inside_take_rate",
        "weighted_recent_win",
    ]

    # Stadium course features - optional
    # Note: Only add unique information, not redundant features
    # stadium_course_win_rate has 0.98 correlation with course_advantage (redundant)
    if include_stadium_course:
        features.extend([
            # Only add the unique delta from global average (not redundant with course_advantage)
            "stadium_course_advantage_diff", # Difference from global course average
            # Racer-specific performance at this stadium+course (sparse but potentially valuable)
            "racer_course_win_at_stadium",  # Racer's win rate on this course at this stadium
            "racer_course_in2_at_stadium",  # Racer's top-2 rate on this course at this stadium
        ])

    features.extend([
        # Relative features (5)
        "win_rate_rank", "win_rate_diff_from_avg",
        "motor_rate_rank", "boat_rate_rank",
        "course_advantage",
        # Exhibition time features (3)
        "exhibition_time",
        "exhibition_time_rank",
        "exhibition_time_diff",
        # Race context features (2)
        "race_grade", "is_final",
        # Interaction features (6)
        "class_x_course", "motor_x_exhibition",
        "equipment_score", "equipment_rank",
        "favorite_score", "upset_potential",
        # Weather features (7)
        "weather_encoded",      # 0=sunny, 1=cloudy, 2=rain, 3=snow, 4=fog
        "wind_speed",           # Wind speed in meters
        "wave_height",          # Wave height in cm
        "wind_direction_sin",   # Circular encoding of wind direction
        "wind_direction_cos",   # Circular encoding of wind direction
        "wind_wave_interaction",  # wind_speed * wave_height / 100
        "inside_wave_penalty",    # wave_height * (7 - boat_no) / 600
    ])

    return features
