"""
合成オッズ生成モジュール

過去データからコース別の2連単確率を計算し、
市場オッズを模擬する
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 歴史的なコース別1着率（ボートレースの一般的な傾向）
# コース1が圧倒的に有利
HISTORICAL_WIN_RATE = {
    1: 0.55,  # 1コース: 約55%
    2: 0.14,  # 2コース: 約14%
    3: 0.12,  # 3コース: 約12%
    4: 0.10,  # 4コース: 約10%
    5: 0.06,  # 5コース: 約6%
    6: 0.03,  # 6コース: 約3%
}

# 2着率（1着以外からの相対確率）
HISTORICAL_SECOND_RATE = {
    1: 0.20,  # 1コースが2着になる確率
    2: 0.22,
    3: 0.20,
    4: 0.18,
    5: 0.12,
    6: 0.08,
}


class SyntheticOddsGenerator:
    """合成オッズ生成器"""

    def __init__(self, margin: float = 0.25):
        """
        Args:
            margin: 控除率（テラ銭）。25%がボートレースの標準
        """
        self.margin = margin
        self.exacta_probs = self._calculate_exacta_probs()

    def _calculate_exacta_probs(self) -> dict:
        """
        コース組み合わせごとの2連単確率を計算

        P(1st=i, 2nd=j) ≈ P(1st=i) × P(2nd=j | 1st≠j)
        """
        probs = {}

        for first in range(1, 7):
            p_first = HISTORICAL_WIN_RATE[first]

            # 1着がfirstの場合、残りから2着を選ぶ
            remaining_second_total = sum(
                HISTORICAL_SECOND_RATE[s] for s in range(1, 7) if s != first
            )

            for second in range(1, 7):
                if first == second:
                    continue

                # 条件付き確率で2着を計算
                p_second_given = HISTORICAL_SECOND_RATE[second] / remaining_second_total
                p_exacta = p_first * p_second_given

                probs[(first, second)] = p_exacta

        # 正規化（合計が1になるように）
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        return probs

    def get_odds(self, first: int, second: int) -> float:
        """
        指定した組み合わせのオッズを取得

        Args:
            first: 1着の艇番
            second: 2着の艇番

        Returns:
            オッズ（配当倍率）
        """
        if first == second or first < 1 or first > 6 or second < 1 or second > 6:
            return 0.0

        prob = self.exacta_probs.get((first, second), 0.001)

        # オッズ = 1 / (確率 × (1 - 控除率))
        # 控除率25%の場合、還元率は75%
        fair_odds = 1.0 / prob
        actual_odds = fair_odds * (1 - self.margin)

        return round(actual_odds, 1)

    def get_all_odds(self) -> dict:
        """
        全30組み合わせのオッズを取得

        Returns:
            {(first, second): odds} の辞書
        """
        return {
            (f, s): self.get_odds(f, s)
            for f in range(1, 7)
            for s in range(1, 7)
            if f != s
        }

    def get_race_odds(self, race_df: pd.DataFrame = None) -> dict:
        """
        レースごとのオッズを取得

        現在は固定オッズを返すが、将来的にはレース特性
        （選手クラス、モーター性能など）を考慮可能

        Returns:
            {(first, second): odds} の辞書
        """
        # TODO: レース特性を考慮したオッズ調整
        return self.get_all_odds()


def calculate_historical_exacta_rates(results_df: pd.DataFrame) -> dict:
    """
    実際の結果データから2連単の出現率を計算

    Args:
        results_df: 結果データフレーム

    Returns:
        {(first, second): rate} の辞書
    """
    exacta_counts = {}
    total_races = 0

    # レースごとにグループ化
    race_groups = results_df.groupby(["date", "stadium_code", "race_no"])

    for _, race_df in race_groups:
        if len(race_df) != 6:
            continue

        first_place = race_df[race_df["rank"] == 1]
        second_place = race_df[race_df["rank"] == 2]

        if len(first_place) == 0 or len(second_place) == 0:
            continue

        first = int(first_place["boat_no"].values[0])
        second = int(second_place["boat_no"].values[0])

        key = (first, second)
        exacta_counts[key] = exacta_counts.get(key, 0) + 1
        total_races += 1

    # 確率に変換
    if total_races > 0:
        return {k: v / total_races for k, v in exacta_counts.items()}
    return {}


if __name__ == "__main__":
    # テスト
    generator = SyntheticOddsGenerator()

    print("Synthetic Exacta Odds (25% margin):")
    print("-" * 40)

    all_odds = generator.get_all_odds()

    # オッズが低い順（人気順）にソート
    sorted_odds = sorted(all_odds.items(), key=lambda x: x[1])

    for (first, second), odds in sorted_odds[:10]:
        prob = generator.exacta_probs[(first, second)]
        print(f"  {first}-{second}: {odds:6.1f}x (prob: {prob:.4f})")

    print("...")
    print(f"\nTotal combinations: {len(all_odds)}")
    print(f"Min odds: {min(all_odds.values()):.1f}x")
    print(f"Max odds: {max(all_odds.values()):.1f}x")
