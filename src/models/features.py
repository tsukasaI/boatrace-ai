"""
特徴量エンジニアリング

レーサー、モーター、ボートの統計情報と過去成績から特徴量を生成
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineering:
    """特徴量生成クラス"""

    # 級別のエンコーディング
    CLASS_ENCODING = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

    def __init__(self, n_recent_races: int = 30):
        """
        Args:
            n_recent_races: 過去成績の計算に使用するレース数
        """
        self.n_recent_races = n_recent_races

    def create_base_features(self, programs_df: pd.DataFrame) -> pd.DataFrame:
        """
        番組表データから基本特徴量を生成

        Args:
            programs_df: 番組表エントリーデータ

        Returns:
            基本特徴量のDataFrame
        """
        df = programs_df.copy()

        # 級別をエンコード
        df["class_encoded"] = df["racer_class"].map(self.CLASS_ENCODING).fillna(0)

        # 基本特徴量
        features = df[[
            "date", "stadium_code", "race_no", "boat_no", "racer_id",
            # レーサー特徴量
            "national_win_rate", "national_in2_rate",
            "local_win_rate", "local_in2_rate",
            "age", "weight", "class_encoded",
            # 機材特徴量
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
        過去成績から履歴特徴量を生成

        Args:
            programs_df: 番組表エントリーデータ
            results_df: レース結果データ

        Returns:
            履歴特徴量のDataFrame
        """
        # 結果データを日付でソート
        results = results_df.sort_values("date").copy()

        # racer_idごとの過去成績を集計
        historical_features = []

        for (date, stadium, race_no), group in programs_df.groupby(
            ["date", "stadium_code", "race_no"]
        ):
            # このレースより前の結果を取得
            past_results = results[results["date"] < date]

            for _, row in group.iterrows():
                racer_id = row["racer_id"]

                # このレーサーの過去成績
                racer_results = past_results[
                    past_results["racer_id"] == racer_id
                ].tail(self.n_recent_races)

                # 過去成績の統計
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
                    avg_rank = 3.5  # 中央値
                    avg_start_timing = 0.15  # 平均的なST
                    race_count = 0

                # 同会場での過去成績
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

                # コース別勝率（進入コース）
                course_results = past_results[
                    (past_results["racer_id"] == racer_id) &
                    (past_results["course"] == row["boat_no"])  # 枠番=コースと仮定
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
                    # 履歴特徴量
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
        レース内での相対特徴量を生成

        Args:
            features_df: 特徴量DataFrame

        Returns:
            相対特徴量を追加したDataFrame
        """
        df = features_df.copy()

        # レースごとにグループ化
        group_cols = ["date", "stadium_code", "race_no"]

        # 勝率の順位（1=最高）
        df["win_rate_rank"] = df.groupby(group_cols)["national_win_rate"].rank(
            ascending=False, method="min"
        )

        # 勝率とレース内平均の差
        race_avg_win_rate = df.groupby(group_cols)["national_win_rate"].transform("mean")
        df["win_rate_diff_from_avg"] = df["national_win_rate"] - race_avg_win_rate

        # モーター2連率の順位
        df["motor_rate_rank"] = df.groupby(group_cols)["motor_in2_rate"].rank(
            ascending=False, method="min"
        )

        # ボート2連率の順位
        df["boat_rate_rank"] = df.groupby(group_cols)["boat_in2_rate"].rank(
            ascending=False, method="min"
        )

        # 枠番の有利不利（1コースが有利）
        # 一般的な1コース勝率は約55%、6コースは約5%
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
        全ての特徴量を生成

        Args:
            programs_df: 番組表エントリーデータ
            results_df: レース結果データ
            include_historical: 履歴特徴量を含めるか

        Returns:
            全特徴量のDataFrame
        """
        # 基本特徴量
        features = self.create_base_features(programs_df)

        # 履歴特徴量
        if include_historical and results_df is not None:
            historical = self.create_historical_features(programs_df, results_df)
            features = features.merge(
                historical,
                on=["date", "stadium_code", "race_no", "boat_no", "racer_id"],
                how="left",
            )

        # 相対特徴量
        features = self.create_relative_features(features)

        return features


def get_feature_columns() -> list[str]:
    """モデルに入力する特徴量カラム名のリスト"""
    return [
        # 基本特徴量
        "national_win_rate", "national_in2_rate",
        "local_win_rate", "local_in2_rate",
        "age", "weight", "class_encoded",
        "motor_in2_rate", "boat_in2_rate",
        # 履歴特徴量
        "recent_win_rate", "recent_in2_rate", "recent_in3_rate",
        "recent_avg_rank", "recent_avg_st", "recent_race_count",
        "local_recent_win_rate", "local_race_count",
        "course_win_rate",
        # 相対特徴量
        "win_rate_rank", "win_rate_diff_from_avg",
        "motor_rate_rank", "boat_rate_rank",
        "course_advantage",
    ]
