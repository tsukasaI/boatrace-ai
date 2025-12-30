"""
データセット構築

番組表と結果データを結合し、学習用データセットを生成
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
    """データセット構築クラス"""

    def __init__(
        self,
        train_end_date: int = 20231231,
        val_end_date: int = 20240630,
    ):
        """
        Args:
            train_end_date: 訓練データの終了日 (YYYYMMDD形式)
            val_end_date: 検証データの終了日 (YYYYMMDD形式)
        """
        self.train_end_date = train_end_date
        self.val_end_date = val_end_date
        self.feature_eng = FeatureEngineering()

    def load_data(
        self,
        data_dir: Path = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        CSV データを読み込み

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
        番組表と結果データを結合

        Args:
            programs_df: 番組表データ
            results_df: 結果データ

        Returns:
            結合されたDataFrame
        """
        # 結合キー
        merge_keys = ["date", "stadium_code", "race_no", "boat_no"]

        # 結果データから必要なカラムを選択
        results_subset = results_df[merge_keys + ["racer_id", "rank", "course", "start_timing"]]

        # 結合
        merged = programs_df.merge(
            results_subset,
            on=merge_keys,
            how="inner",
            suffixes=("", "_result"),
        )

        return merged

    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        着順から6クラスの確率ラベルを生成

        Args:
            df: rankカラムを含むDataFrame

        Returns:
            shape (n_samples, 6) の one-hot ラベル
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
        時系列でデータを分割

        Args:
            df: 全データ

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
        学習用データセットを構築

        Args:
            data_dir: データディレクトリ
            include_historical: 履歴特徴量を含めるか

        Returns:
            データセット辞書
        """
        # データ読み込み
        programs_df, results_df = self.load_data(data_dir)

        # データ結合
        merged_df = self.merge_data(programs_df, results_df)

        # 特徴量生成
        if include_historical:
            # 履歴特徴量には全結果データが必要
            features_df = self.feature_eng.create_all_features(
                merged_df, results_df, include_historical=True
            )
        else:
            features_df = self.feature_eng.create_all_features(
                merged_df, None, include_historical=False
            )

        # ラベル追加
        features_df["rank"] = merged_df["rank"].values

        # データ分割
        train_df, val_df, test_df = self.split_data(features_df)

        # 特徴量カラム
        feature_cols = get_feature_columns()

        # 欠損値を埋める
        for col in feature_cols:
            if col in train_df.columns:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                val_df[col] = val_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)

        # 特徴量とラベルを抽出
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
    履歴特徴量なしの簡易データセットを構築

    Args:
        data_dir: データディレクトリ

    Returns:
        データセット辞書
    """
    builder = DatasetBuilder()
    return builder.build_dataset(data_dir, include_historical=False)


def build_full_dataset(data_dir: Path = None) -> dict:
    """
    履歴特徴量ありの完全データセットを構築

    Args:
        data_dir: データディレクトリ

    Returns:
        データセット辞書
    """
    builder = DatasetBuilder()
    return builder.build_dataset(data_dir, include_historical=True)
