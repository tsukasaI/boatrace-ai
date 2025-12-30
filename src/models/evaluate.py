"""
Model Evaluation

Evaluate prediction accuracy, calibration, and profitability
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

# Results save directory
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate prediction accuracy

    Args:
        y_true: Ground truth labels (n_samples, 6) - one-hot
        y_pred: Predicted probabilities (n_samples, 6)

    Returns:
        Dictionary of accuracy metrics
    """
    # Actual finishing position (1-indexed)
    true_rank = np.argmax(y_true, axis=1) + 1

    # Predicted position with highest probability
    pred_rank = np.argmax(y_pred, axis=1) + 1

    # Top-1 accuracy (correctly predict 1st place)
    top1_accuracy = np.mean(pred_rank == 1)  # Whether prediction of 1st is actually 1st
    # Correct approach: Whether the boat with highest predicted probability actually finished 1st
    actual_first = y_true[:, 0] == 1
    pred_first = np.argmax(y_pred, axis=1) == 0  # Highest probability for 1st place
    # This is incorrect. Need to evaluate per race.

    # Simple evaluation: Distribution of actual positions for boats with highest predicted probability
    pred_top_boat = np.argmax(y_pred, axis=1)  # Index of boat with highest 1st place probability
    # Need per-race evaluation, so here we evaluate individual samples

    metrics = {
        "mean_pred_prob_for_winner": 0.0,
        "mean_pred_prob_for_second": 0.0,
    }

    # Average predicted probability for 1st place boats
    winner_probs = []
    second_probs = []
    for i in range(len(y_true)):
        if y_true[i, 0] == 1:  # This boat finished 1st
            winner_probs.append(y_pred[i, 0])
        if y_true[i, 1] == 1:  # This boat finished 2nd
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
    Calculate prediction accuracy per race

    Args:
        y_true: Ground truth labels (n_samples, 6)
        y_pred: Predicted probabilities (n_samples, 6)
        race_ids: Race IDs (n_samples,)

    Returns:
        Dictionary of accuracy metrics
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

        # 1st place prediction
        pred_first_idx = np.argmax(race_pred[:, 0])  # Boat with highest 1st place probability
        true_first_idx = np.argmax(race_true[:, 0])  # Actual 1st place

        if pred_first_idx == true_first_idx:
            top1_correct += 1

        # Top-2: Whether actual 1st place is in predicted top 2 boats
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
    Calculate calibration

    Args:
        y_true: Ground truth labels (n_samples, 6)
        y_pred: Predicted probabilities (n_samples, 6)
        n_bins: Number of bins

    Returns:
        (Mean predicted probabilities, Actual probabilities)
    """
    # Evaluate calibration using 1st place probability
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
    """Draw calibration plot"""
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
    Evaluate the model

    Args:
        model: Model to evaluate (loads from file if None)
        use_historical: Whether to use historical features

    Returns:
        Dictionary of evaluation results
    """
    # Load model
    if model is None:
        model = BoatracePredictor()
        model.load()

    # Build dataset
    logger.info("Building dataset...")
    builder = DatasetBuilder()
    dataset = builder.build_dataset(include_historical=use_historical)

    # Evaluate on test data
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    test_df = dataset["test_df"]

    logger.info(f"Test samples: {len(X_test)}")

    # Predict
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)

    # Create race IDs
    race_ids = (
        test_df["date"].astype(str) + "_" +
        test_df["stadium_code"].astype(str) + "_" +
        test_df["race_no"].astype(str)
    ).values

    # Accuracy evaluation
    logger.info("Calculating accuracy...")
    sample_metrics = calculate_accuracy(y_test, y_pred)
    race_metrics = calculate_race_accuracy(y_test, y_pred, race_ids)

    # Calibration
    logger.info("Calculating calibration...")
    pred_means, true_means = calculate_calibration(y_test, y_pred)
    plot_calibration(pred_means, true_means, RESULTS_DIR / "calibration.png")

    # Summarize results
    results = {
        **sample_metrics,
        **race_metrics,
        "calibration": {
            "pred_means": pred_means.tolist(),
            "true_means": true_means.tolist(),
        },
    }

    # Display results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  Top-1 Accuracy: {race_metrics['top1_accuracy']:.1%}")
    logger.info(f"  Top-2 Accuracy: {race_metrics['top2_accuracy']:.1%}")
    logger.info(f"  Top-3 Accuracy: {race_metrics['top3_accuracy']:.1%}")
    logger.info(f"  Mean predicted prob for winners: {sample_metrics['mean_pred_prob_for_winner']:.1%}")
    logger.info(f"  Total races evaluated: {race_metrics['total_races']}")

    return results


def main():
    """Main process"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate boatrace prediction model")
    parser.add_argument("--historical", action="store_true", help="Use historical features")
    args = parser.parse_args()

    evaluate_model(use_historical=args.historical)


if __name__ == "__main__":
    main()
