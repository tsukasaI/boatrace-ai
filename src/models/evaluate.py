"""
モデル評価

予測精度、キャリブレーション、収益性を評価
"""

import sys
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT
from src.models.train import BoatracePredictor
from src.models.dataset import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 結果保存ディレクトリ
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    予測精度を計算

    Args:
        y_true: 正解ラベル (n_samples, 6) - one-hot
        y_pred: 予測確率 (n_samples, 6)

    Returns:
        精度指標の辞書
    """
    # 実際の着順（1-indexed）
    true_rank = np.argmax(y_true, axis=1) + 1

    # 予測された最も確率の高い着順
    pred_rank = np.argmax(y_pred, axis=1) + 1

    # Top-1精度（1着を正しく予測）
    top1_accuracy = np.mean(pred_rank == 1)  # 1着と予測したものが実際に1着か
    # 正しくは: 予測確率が最も高い艇が実際に1着か
    actual_first = y_true[:, 0] == 1
    pred_first = np.argmax(y_pred, axis=1) == 0  # 1着確率が最も高い
    # これは間違い。レースごとに評価する必要がある

    # 簡易評価: 予測確率が最も高い艇の実際の着順分布
    pred_top_boat = np.argmax(y_pred, axis=1)  # 予測で最も1着確率が高い艇のインデックス
    # レースごとの評価が必要なので、ここでは個別のサンプルで評価

    metrics = {
        "mean_pred_prob_for_winner": 0.0,
        "mean_pred_prob_for_second": 0.0,
    }

    # 1着の艇の予測確率の平均
    winner_probs = []
    second_probs = []
    for i in range(len(y_true)):
        if y_true[i, 0] == 1:  # この艇が1着
            winner_probs.append(y_pred[i, 0])
        if y_true[i, 1] == 1:  # この艇が2着
            second_probs.append(y_pred[i, 1])

    if winner_probs:
        metrics["mean_pred_prob_for_winner"] = np.mean(winner_probs)
    if second_probs:
        metrics["mean_pred_prob_for_second"] = np.mean(second_probs)

    return metrics


def calculate_race_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    race_ids: np.ndarray,
) -> dict:
    """
    レース単位での予測精度を計算

    Args:
        y_true: 正解ラベル (n_samples, 6)
        y_pred: 予測確率 (n_samples, 6)
        race_ids: レースID (n_samples,)

    Returns:
        精度指標の辞書
    """
    unique_races = np.unique(race_ids)

    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    total_races = len(unique_races)

    for race_id in unique_races:
        mask = race_ids == race_id
        race_true = y_true[mask]
        race_pred = y_pred[mask]

        if len(race_true) != 6:
            total_races -= 1
            continue

        # 1着の予測
        pred_first_idx = np.argmax(race_pred[:, 0])  # 1着確率が最も高い艇
        true_first_idx = np.argmax(race_true[:, 0])  # 実際の1着

        if pred_first_idx == true_first_idx:
            top1_correct += 1

        # Top-2: 予測上位2艇に実際の1着が含まれるか
        pred_top2 = np.argsort(race_pred[:, 0])[-2:]
        if true_first_idx in pred_top2:
            top2_correct += 1

        # Top-3
        pred_top3 = np.argsort(race_pred[:, 0])[-3:]
        if true_first_idx in pred_top3:
            top3_correct += 1

    if total_races == 0:
        return {"top1": 0, "top2": 0, "top3": 0}

    return {
        "top1_accuracy": top1_correct / total_races,
        "top2_accuracy": top2_correct / total_races,
        "top3_accuracy": top3_correct / total_races,
        "total_races": total_races,
    }


def calculate_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    キャリブレーションを計算

    Args:
        y_true: 正解ラベル (n_samples, 6)
        y_pred: 予測確率 (n_samples, 6)
        n_bins: ビン数

    Returns:
        (予測確率の平均, 実際の確率)
    """
    # 1着の確率でキャリブレーションを評価
    pred_probs = y_pred[:, 0]
    true_labels = y_true[:, 0]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    pred_means = []
    true_means = []

    for i in range(n_bins):
        mask = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            pred_means.append(pred_probs[mask].mean())
            true_means.append(true_labels[mask].mean())

    return np.array(pred_means), np.array(true_means)


def plot_calibration(
    pred_means: np.ndarray,
    true_means: np.ndarray,
    save_path: Path = None,
) -> None:
    """キャリブレーションプロットを描画"""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.scatter(pred_means, true_means, s=100, label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot (1st place prediction)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Calibration plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def evaluate_model(
    model: BoatracePredictor = None,
    use_historical: bool = False,
) -> dict:
    """
    モデルを評価

    Args:
        model: 評価するモデル（Noneの場合は読み込み）
        use_historical: 履歴特徴量を使用するか

    Returns:
        評価結果の辞書
    """
    # モデル読み込み
    if model is None:
        model = BoatracePredictor()
        model.load()

    # データセット構築
    logger.info("Building dataset...")
    builder = DatasetBuilder()
    dataset = builder.build_dataset(include_historical=use_historical)

    # テストデータで評価
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    test_df = dataset["test_df"]

    logger.info(f"Test samples: {len(X_test)}")

    # 予測
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)

    # レースID作成
    race_ids = (
        test_df["date"].astype(str) + "_" +
        test_df["stadium_code"].astype(str) + "_" +
        test_df["race_no"].astype(str)
    ).values

    # 精度評価
    logger.info("Calculating accuracy...")
    sample_metrics = calculate_accuracy(y_test, y_pred)
    race_metrics = calculate_race_accuracy(y_test, y_pred, race_ids)

    # キャリブレーション
    logger.info("Calculating calibration...")
    pred_means, true_means = calculate_calibration(y_test, y_pred)
    plot_calibration(pred_means, true_means, RESULTS_DIR / "calibration.png")

    # 結果をまとめる
    results = {
        **sample_metrics,
        **race_metrics,
        "calibration": {
            "pred_means": pred_means.tolist(),
            "true_means": true_means.tolist(),
        },
    }

    # 結果を表示
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  Top-1 Accuracy: {race_metrics['top1_accuracy']:.1%}")
    logger.info(f"  Top-2 Accuracy: {race_metrics['top2_accuracy']:.1%}")
    logger.info(f"  Top-3 Accuracy: {race_metrics['top3_accuracy']:.1%}")
    logger.info(f"  Mean predicted prob for winners: {sample_metrics['mean_pred_prob_for_winner']:.1%}")
    logger.info(f"  Total races evaluated: {race_metrics['total_races']}")

    return results


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate boatrace prediction model")
    parser.add_argument("--historical", action="store_true", help="Use historical features")
    args = parser.parse_args()

    evaluate_model(use_historical=args.historical)


if __name__ == "__main__":
    main()
