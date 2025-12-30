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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CSV data

        Returns:
            (programs_df, results_df)
        """
        data_dir = data_dir or PROCESSED_DATA_DIR

        programs_df = pd.read_csv(data_dir / "programs_entries.csv")
        results_df = pd.read_csv(data_dir / "results_entries.csv")

        return programs_df, results_df

    def merge_data(
        self,
        programs_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge program and results data

        Args:
            programs_df: Program data
            results_df: Results data

        Returns:
            Merged DataFrame
        """
        # Merge keys
        merge_keys = ["date", "stadium_code", "race_no", "boat_no"]

        # Select required columns from results data
        results_subset = results_df[merge_keys + ["racer_id", "rank", "course", "start_timing"]]

        # Merge
        merged = programs_df.merge(
            results_subset,
            on=merge_keys,
            how="inner",
            suffixes=("", "_result"),
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
    ) -> dict:
        """
        Build training dataset

        Args:
            data_dir: Data directory
            include_historical: Whether to include historical features

        Returns:
            Dataset dictionary
        """
        # Load data
        programs_df, results_df = self.load_data(data_dir)

        # Merge data
        merged_df = self.merge_data(programs_df, results_df)

        # Generate features
        if include_historical:
            # Historical features require all results data
            features_df = self.feature_eng.create_all_features(
                merged_df, results_df, include_historical=True
            )
        else:
            features_df = self.feature_eng.create_all_features(
                merged_df, None, include_historical=False
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
