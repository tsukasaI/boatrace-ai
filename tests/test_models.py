"""
Tests for models module (features, dataset)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.features import FeatureEngineering, get_feature_columns
from src.models.dataset import DatasetBuilder


class TestFeatureEngineering:
    """Tests for FeatureEngineering class"""

    def test_class_encoding(self):
        """Test class encoding dictionary"""
        assert FeatureEngineering.CLASS_ENCODING["A1"] == 4
        assert FeatureEngineering.CLASS_ENCODING["A2"] == 3
        assert FeatureEngineering.CLASS_ENCODING["B1"] == 2
        assert FeatureEngineering.CLASS_ENCODING["B2"] == 1

    def test_create_base_features(self, sample_programs_df):
        """Test base feature creation"""
        fe = FeatureEngineering()
        features = fe.create_base_features(sample_programs_df)

        assert len(features) == 6
        assert "class_encoded" in features.columns
        assert "national_win_rate" in features.columns
        assert "motor_in2_rate" in features.columns

        # Check class encoding is correctly mapped (A2 -> 3)
        # Racer 3527 is A2 class, so their class_encoded should be 3
        a2_racer = features[features["racer_id"] == 3527]
        assert a2_racer["class_encoded"].values[0] == 3

    def test_create_relative_features(self, sample_programs_df):
        """Test relative feature creation"""
        fe = FeatureEngineering()
        base_features = fe.create_base_features(sample_programs_df)
        relative_features = fe.create_relative_features(base_features)

        assert "win_rate_rank" in relative_features.columns
        assert "win_rate_diff_from_avg" in relative_features.columns
        assert "motor_rate_rank" in relative_features.columns
        assert "boat_rate_rank" in relative_features.columns
        assert "course_advantage" in relative_features.columns

        # Check rankings are 1-6
        assert relative_features["win_rate_rank"].min() >= 1
        assert relative_features["win_rate_rank"].max() <= 6

        # Check course advantage values
        boat_1_advantage = relative_features[
            relative_features["boat_no"] == 1
        ]["course_advantage"].values[0]
        assert boat_1_advantage == 0.55

    def test_create_historical_features(self, sample_programs_df, sample_results_df):
        """Test historical feature creation"""
        fe = FeatureEngineering(n_recent_races=10)

        # Need programs and results with overlapping data
        historical = fe.create_historical_features(sample_programs_df, sample_results_df)

        # Should have one row per boat
        assert len(historical) == 6

        expected_cols = [
            "recent_win_rate", "recent_in2_rate", "recent_in3_rate",
            "recent_avg_rank", "recent_avg_st", "recent_race_count",
            "local_recent_win_rate", "local_race_count", "course_win_rate"
        ]
        for col in expected_cols:
            assert col in historical.columns

    def test_create_all_features_without_historical(self, sample_programs_df):
        """Test all features without historical"""
        fe = FeatureEngineering()
        features = fe.create_all_features(
            sample_programs_df, None, include_historical=False
        )

        assert len(features) == 6
        assert "national_win_rate" in features.columns
        assert "course_advantage" in features.columns
        # Historical columns should not be present
        assert "recent_win_rate" not in features.columns


class TestGetFeatureColumns:
    """Tests for get_feature_columns function"""

    def test_returns_list(self):
        """Test that function returns a list"""
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_contains_expected_columns(self):
        """Test that expected columns are present"""
        cols = get_feature_columns()

        # Base features
        assert "national_win_rate" in cols
        assert "motor_in2_rate" in cols

        # Relative features
        assert "win_rate_rank" in cols
        assert "course_advantage" in cols


class TestDatasetBuilder:
    """Tests for DatasetBuilder class"""

    def test_init_default_dates(self):
        """Test default date initialization"""
        builder = DatasetBuilder()
        assert builder.train_end_date == 20231231
        assert builder.val_end_date == 20240630

    def test_init_custom_dates(self):
        """Test custom date initialization"""
        builder = DatasetBuilder(
            train_end_date=20230630,
            val_end_date=20230930
        )
        assert builder.train_end_date == 20230630
        assert builder.val_end_date == 20230930

    def test_create_labels(self, sample_results_df):
        """Test label creation from ranks"""
        builder = DatasetBuilder()
        labels = builder.create_labels(sample_results_df)

        assert labels.shape == (6, 6)
        # Each row should have exactly one 1.0
        assert np.allclose(labels.sum(axis=1), 1.0)

        # First row has rank=1, so first column should be 1
        first_place = sample_results_df[sample_results_df["rank"] == 1].index[0]
        row_idx = sample_results_df.index.get_loc(first_place)
        assert labels[row_idx, 0] == 1.0

    def test_create_labels_invalid_rank(self):
        """Test label creation with invalid ranks"""
        builder = DatasetBuilder()
        df = pd.DataFrame({"rank": [0, 7, -1]})
        labels = builder.create_labels(df)

        # Invalid ranks should result in all zeros
        assert labels.shape == (3, 6)
        assert np.allclose(labels.sum(), 0.0)

    def test_merge_data(self, sample_programs_df, sample_results_df):
        """Test merging program and results data"""
        builder = DatasetBuilder()
        merged = builder.merge_data(sample_programs_df, sample_results_df)

        assert "national_win_rate" in merged.columns  # From programs
        assert "rank" in merged.columns  # From results
        assert len(merged) == 6

    def test_split_data(self):
        """Test time-based data splitting"""
        builder = DatasetBuilder(
            train_end_date=20231231,
            val_end_date=20240630
        )

        df = pd.DataFrame({
            "date": [20230601, 20231115, 20240315, 20240815],
            "value": [1, 2, 3, 4]
        })

        train, val, test = builder.split_data(df)

        assert len(train) == 2  # 20230601, 20231115
        assert len(val) == 1    # 20240315
        assert len(test) == 1   # 20240815
