"""
Dataset Construction

Merge program and results data to generate training dataset
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.features import FeatureEngineering, get_feature_columns


class DatasetBuilder:
    """Dataset construction class"""

    def __init__(
        self,
        train_end_date: int = 20231231,
        val_end_date: int = 20240630,
    ):
        """
        Args:
            train_end_date: End date for training data (YYYYMMDD format)
            val_end_date: End date for validation data (YYYYMMDD format)
        """
        self.train_end_date = train_end_date
        self.val_end_date = val_end_date
        self.feature_eng = FeatureEngineering()

    def load_data(
        self,
        data_dir: Path = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load CSV data

        Returns:
            (programs_df, results_df, races_df, weather_df)
        """
        data_dir = data_dir or PROCESSED_DATA_DIR

        programs_df = pd.read_csv(data_dir / "programs_entries.csv")
        results_df = pd.read_csv(data_dir / "results_entries.csv")

        # Load races data for race_type
        races_path = data_dir / "programs_races.csv"
        if races_path.exists():
            races_df = pd.read_csv(races_path)
        else:
            races_df = None

        # Load results races for weather data
        results_races_path = data_dir / "results_races.csv"
        if results_races_path.exists():
            weather_df = pd.read_csv(results_races_path)
        else:
            weather_df = None

        return programs_df, results_df, races_df, weather_df

    def merge_data(
        self,
        programs_df: pd.DataFrame,
        results_df: pd.DataFrame,
        races_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Merge program and results data

        Args:
            programs_df: Program data
            results_df: Results data
            races_df: Race info data (for race_type)

        Returns:
            Merged DataFrame
        """
        # Merge keys
        merge_keys = ["date", "stadium_code", "race_no", "boat_no"]
        race_keys = ["date", "stadium_code", "race_no"]

        # Select required columns from results data
        results_cols = ["racer_id", "rank", "course", "start_timing"]
        if "exhibition_time" in results_df.columns:
            results_cols.append("exhibition_time")
        results_subset = results_df[merge_keys + results_cols]

        # Merge programs with results
        merged = programs_df.merge(
            results_subset,
            on=merge_keys,
            how="inner",
            suffixes=("", "_result"),
        )

        # Merge with races data for race_type
        if races_df is not None and "race_type" in races_df.columns:
            races_subset = races_df[race_keys + ["race_type"]].drop_duplicates()
            merged = merged.merge(
                races_subset,
                on=race_keys,
                how="left",
            )

        return merged

    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate 6-class probability labels from finishing position

        Args:
            df: DataFrame containing rank column

        Returns:
            One-hot labels with shape (n_samples, 6)
        """
        n_samples = len(df)
        labels = np.zeros((n_samples, 6))

        for i, rank in enumerate(df["rank"].values):
            if 1 <= rank <= 6:
                labels[i, rank - 1] = 1.0

        return labels

    def create_ranking_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate ranking labels for LambdaRank training.

        Labels are relevance scores: 5 for 1st place, 4 for 2nd, ..., 0 for 6th.
        Invalid ranks (disqualified, etc.) get 0.

        Args:
            df: DataFrame containing rank column

        Returns:
            Ranking labels with shape (n_samples,) - integer relevance scores
        """
        labels = np.zeros(len(df), dtype=np.int32)

        for i, rank in enumerate(df["rank"].values):
            if 1 <= rank <= 6:
                # Higher relevance for better position
                # 1st place = 5 (highest), 6th place = 0 (lowest)
                labels[i] = 6 - int(rank)
            else:
                # Disqualified boats get 0 relevance
                labels[i] = 0

        return labels

    def create_group_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create group array for LambdaRank training.

        Each race is a group. Groups are identified by (date, stadium_code, race_no).
        Returns array where each element is the size of a group (should be 6 for valid races).

        Args:
            df: DataFrame with race identifiers (must be sorted by race)

        Returns:
            Group sizes array (n_groups,)
        """
        group_sizes = df.groupby(
            ["date", "stadium_code", "race_no"], sort=False
        ).size().values
        return group_sizes

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by time series

        Args:
            df: All data

        Returns:
            (train_df, val_df, test_df)
        """
        train_df = df[df["date"] <= self.train_end_date].copy()
        val_df = df[
            (df["date"] > self.train_end_date) &
            (df["date"] <= self.val_end_date)
        ].copy()
        test_df = df[df["date"] > self.val_end_date].copy()

        return train_df, val_df, test_df

    def build_dataset(
        self,
        data_dir: Path = None,
        include_historical: bool = True,
        for_ranking: bool = False,
    ) -> dict:
        """
        Build training dataset

        Args:
            data_dir: Data directory
            include_historical: Whether to include historical features
            for_ranking: Whether to build dataset for LambdaRank training

        Returns:
            Dataset dictionary
        """
        # Load data
        programs_df, results_df, races_df, weather_df = self.load_data(data_dir)

        # Merge data
        merged_df = self.merge_data(programs_df, results_df, races_df)

        # Generate features
        if include_historical:
            # Historical features require all results data
            features_df = self.feature_eng.create_all_features(
                merged_df, results_df, include_historical=True, weather_df=weather_df
            )
        else:
            features_df = self.feature_eng.create_all_features(
                merged_df, None, include_historical=False, weather_df=weather_df
            )

        # Add labels
        features_df["rank"] = merged_df["rank"].values

        # Split data
        train_df, val_df, test_df = self.split_data(features_df)

        # Feature columns
        feature_cols = get_feature_columns()

        # Fill missing values
        for col in feature_cols:
            if col in train_df.columns:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                val_df[col] = val_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)

        # Extract features and labels
        available_cols = [c for c in feature_cols if c in train_df.columns]

        if for_ranking:
            # For LambdaRank: sort by race and create ranking labels + groups
            # Sort to ensure boats within each race are contiguous
            sort_cols = ["date", "stadium_code", "race_no", "boat_no"]
            train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
            val_df = val_df.sort_values(sort_cols).reset_index(drop=True)
            test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

            X_train = train_df[available_cols].values
            y_train = self.create_ranking_labels(train_df)
            group_train = self.create_group_array(train_df)

            X_val = val_df[available_cols].values
            y_val = self.create_ranking_labels(val_df)
            group_val = self.create_group_array(val_df)

            X_test = test_df[available_cols].values
            y_test = self.create_ranking_labels(test_df)
            group_test = self.create_group_array(test_df)

            return {
                "X_train": X_train,
                "y_train": y_train,
                "group_train": group_train,
                "X_val": X_val,
                "y_val": y_val,
                "group_val": group_val,
                "X_test": X_test,
                "y_test": y_test,
                "group_test": group_test,
                "feature_names": available_cols,
                "train_df": train_df,
                "val_df": val_df,
                "test_df": test_df,
            }
        else:
            # Original binary classification format
            X_train = train_df[available_cols].values
            y_train = self.create_labels(train_df)

            X_val = val_df[available_cols].values
            y_val = self.create_labels(val_df)

            X_test = test_df[available_cols].values
            y_test = self.create_labels(test_df)

            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "feature_names": available_cols,
                "train_df": train_df,
                "val_df": val_df,
                "test_df": test_df,
            }


def build_simple_dataset(data_dir: Path = None) -> dict:
    """
    Build simple dataset without historical features

    Args:
        data_dir: Data directory

    Returns:
        Dataset dictionary
    """
    builder = DatasetBuilder()
    return builder.build_dataset(data_dir, include_historical=False)


def build_full_dataset(data_dir: Path = None) -> dict:
    """
    Build full dataset with historical features

    Args:
        data_dir: Data directory

    Returns:
        Dataset dictionary
    """
    builder = DatasetBuilder()
    return builder.build_dataset(data_dir, include_historical=True)
