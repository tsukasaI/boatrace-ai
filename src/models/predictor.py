"""
予測・期待値計算

モデルの予測結果から2連単確率と期待値を計算
"""

import sys
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.train import BoatracePredictor
from src.models.features import FeatureEngineering, get_feature_columns


@dataclass
class ExactaBet:
    """2連単の賭け情報"""
    first: int       # 1着の艇番 (1-6)
    second: int      # 2着の艇番 (1-6)
    probability: float
    odds: float = 0.0
    expected_value: float = 0.0


class RacePredictor:
    """レース予測クラス"""

    def __init__(self, model: BoatracePredictor = None):
        """
        Args:
            model: 訓練済みモデル
        """
        self.model = model or BoatracePredictor()
        self.feature_eng = FeatureEngineering()

    def load_model(self, path: Path = None) -> None:
        """モデルを読み込み"""
        self.model.load(path)

    def predict_positions(self, X: np.ndarray) -> np.ndarray:
        """
        着順確率を予測

        Args:
            X: 特徴量 (6, n_features) - レース内の6艇分

        Returns:
            確率 (6, 6) - 各艇の各着順確率
        """
        return self.model.predict(X)

    def calculate_exacta_probabilities(
        self,
        position_probs: np.ndarray,
    ) -> List[ExactaBet]:
        """
        2連単確率を計算

        Args:
            position_probs: 着順確率 (6, 6) - [艇番, 着順]

        Returns:
            30通りの2連単賭け情報リスト
        """
        exacta_bets = []

        for first in range(6):
            p_first = position_probs[first, 0]  # 1着確率

            for second in range(6):
                if first == second:
                    continue

                p_second = position_probs[second, 1]  # 2着確率

                # 条件付き確率: P(B=2nd | A=1st)
                # ≈ P(B=2nd) / (1 - P(B=1st))
                p_second_given_first = p_second / max(1 - position_probs[second, 0], 0.01)

                # 2連単確率
                p_exacta = p_first * p_second_given_first

                exacta_bets.append(ExactaBet(
                    first=first + 1,  # 1-indexed
                    second=second + 1,
                    probability=p_exacta,
                ))

        # 確率で降順ソート
        exacta_bets.sort(key=lambda x: x.probability, reverse=True)

        return exacta_bets

    def calculate_expected_values(
        self,
        exacta_bets: List[ExactaBet],
        odds: dict,
    ) -> List[ExactaBet]:
        """
        期待値を計算

        Args:
            exacta_bets: 2連単賭け情報リスト
            odds: オッズ辞書 {(first, second): odds}

        Returns:
            期待値を追加した2連単賭け情報リスト
        """
        for bet in exacta_bets:
            key = (bet.first, bet.second)
            if key in odds:
                bet.odds = odds[key]
                bet.expected_value = bet.probability * bet.odds

        return exacta_bets

    def get_value_bets(
        self,
        exacta_bets: List[ExactaBet],
        threshold: float = 1.0,
    ) -> List[ExactaBet]:
        """
        期待値がしきい値を超える賭けを取得

        Args:
            exacta_bets: 2連単賭け情報リスト
            threshold: 期待値のしきい値

        Returns:
            バリューベットのリスト
        """
        return [bet for bet in exacta_bets if bet.expected_value > threshold]

    def predict_race(
        self,
        race_features: np.ndarray,
        odds: dict = None,
    ) -> Tuple[np.ndarray, List[ExactaBet]]:
        """
        レース予測のメイン関数

        Args:
            race_features: 特徴量 (6, n_features)
            odds: オッズ辞書

        Returns:
            (着順確率, 2連単賭け情報リスト)
        """
        # 着順確率を予測
        position_probs = self.predict_positions(race_features)

        # 2連単確率を計算
        exacta_bets = self.calculate_exacta_probabilities(position_probs)

        # 期待値を計算
        if odds:
            exacta_bets = self.calculate_expected_values(exacta_bets, odds)

        return position_probs, exacta_bets


def format_prediction_result(
    position_probs: np.ndarray,
    exacta_bets: List[ExactaBet],
    top_n: int = 10,
) -> str:
    """
    予測結果を整形して文字列で返す

    Args:
        position_probs: 着順確率 (6, 6)
        exacta_bets: 2連単賭け情報リスト
        top_n: 表示する上位件数

    Returns:
        整形済み文字列
    """
    lines = []

    # 着順確率
    lines.append("=== 着順確率 ===")
    lines.append("艇番  1着    2着    3着    4着    5着    6着")
    lines.append("-" * 50)

    for boat in range(6):
        probs = position_probs[boat]
        prob_str = " ".join(f"{p:.1%}" for p in probs)
        lines.append(f"  {boat + 1}  {prob_str}")

    lines.append("")

    # 2連単予測
    lines.append(f"=== 2連単 TOP{top_n} ===")
    lines.append("順位  組み合わせ  確率    オッズ   期待値")
    lines.append("-" * 50)

    for i, bet in enumerate(exacta_bets[:top_n]):
        odds_str = f"{bet.odds:.1f}" if bet.odds > 0 else "-"
        ev_str = f"{bet.expected_value:.2f}" if bet.expected_value > 0 else "-"
        lines.append(
            f" {i + 1:2}    {bet.first}-{bet.second}      "
            f"{bet.probability:.1%}   {odds_str:>6}   {ev_str:>6}"
        )

    # バリューベット
    value_bets = [b for b in exacta_bets if b.expected_value > 1.0]
    if value_bets:
        lines.append("")
        lines.append("=== バリューベット (期待値 > 1.0) ===")
        for bet in value_bets:
            lines.append(
                f"  {bet.first}-{bet.second}: 確率={bet.probability:.1%}, "
                f"オッズ={bet.odds:.1f}, 期待値={bet.expected_value:.2f}"
            )

    return "\n".join(lines)
