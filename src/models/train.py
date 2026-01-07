"""
Model Training

Train finishing position prediction model using LightGBM
with optional Platt scaling calibration for probability estimates
"""

import sys
import logging
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT
from src.models.dataset import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model save directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class BoatracePredictor:
    """Boat race prediction model with optional Platt scaling calibration"""

    def __init__(self, params: dict = None):
        """
        Args:
            params: LightGBM parameters
        """
        self.params = params or self._default_params()
        self.models = None
        self.feature_names = None
        self.calibrators = None  # Platt scaling models for each position

    def _default_params(self) -> dict:
        """Default LightGBM parameters"""
        return {
            "objective": "binary",  # Binary classification for each position
            "metric": "binary_logloss",  # Log loss for probability calibration
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
    ) -> None:
        """
        Train the model

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples, 6) - probability for each position
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
        """
        self.feature_names = feature_names

        # Train a model for each of the 6 finishing positions
        self.models = []

        for pos in range(6):
            logger.info(f"Training model for position {pos + 1}...")

            # LightGBM dataset
            train_data = lgb.Dataset(
                X_train, label=y_train[:, pos],
                feature_name=feature_names,
            )

            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(
                    X_val, label=y_val[:, pos],
                    reference=train_data,
                )
                callbacks = [lgb.early_stopping(self.params.get("early_stopping_rounds", 50))]
            else:
                val_data = None
                callbacks = None

            # Training
            model = lgb.train(
                self.params,
                train_data,
                valid_sets=[val_data] if val_data else None,
                callbacks=callbacks,
            )

            self.models.append(model)

        logger.info("Training completed!")

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """
        Calibrate model predictions using Platt scaling

        Platt scaling fits a logistic regression to transform raw predictions
        into well-calibrated probabilities. This helps fix overconfidence
        issues, especially for extreme probabilities.

        Args:
            X_cal: Calibration features (n_samples, n_features)
            y_cal: Calibration labels (n_samples, 6) - binary for each position
        """
        if self.models is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info("Calibrating model with Platt scaling...")

        self.calibrators = []

        for pos, model in enumerate(self.models):
            # Get raw predictions
            raw_preds = model.predict(X_cal)

            # Fit Platt scaling (logistic regression on raw predictions)
            # Binary target: 1 if racer finished in this position, 0 otherwise
            y_binary = y_cal[:, pos]

            calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            calibrator.fit(raw_preds.reshape(-1, 1), (y_binary > 0.5).astype(int))

            self.calibrators.append(calibrator)
            logger.info(f"  Position {pos + 1}: calibrated")

        logger.info("Calibration completed!")

    def predict(self, X: np.ndarray, use_calibration: bool = True) -> np.ndarray:
        """
        Predict finishing position probabilities

        Args:
            X: Features (n_samples, n_features)
            use_calibration: Whether to apply Platt scaling calibration

        Returns:
            Probabilities (n_samples, 6) - predicted probability for each position
        """
        if self.models is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = np.zeros((X.shape[0], 6))

        for pos, model in enumerate(self.models):
            raw_pred = model.predict(X)

            # Apply Platt scaling if available and requested
            if use_calibration and self.calibrators is not None:
                calibrator = self.calibrators[pos]
                # Get calibrated probability (probability of class 1)
                predictions[:, pos] = calibrator.predict_proba(
                    raw_pred.reshape(-1, 1)
                )[:, 1]
            else:
                predictions[:, pos] = raw_pred

        # Convert to probability (normalize to 0-1)
        predictions = np.clip(predictions, 0, 1)

        # Normalize each row (so that sum equals 1)
        row_sums = predictions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        predictions = predictions / row_sums

        return predictions

    def save(self, path: Path = None) -> None:
        """Save the model"""
        path = path or MODEL_DIR / "boatrace_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "params": self.params,
                "feature_names": self.feature_names,
                "calibrators": self.calibrators,
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path = None) -> None:
        """Load the model"""
        path = path or MODEL_DIR / "boatrace_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.models = data["models"]
            self.params = data["params"]
            self.feature_names = data["feature_names"]
            self.calibrators = data.get("calibrators")  # May not exist in old models
        logger.info(f"Model loaded from {path}")


class BoatraceRanker:
    """Boat race ranking model using LightGBM LambdaRank.

    Instead of 6 independent binary classifiers, this uses a single
    learning-to-rank model that directly learns to rank boats within each race.
    """

    def __init__(self, params: dict = None):
        """
        Args:
            params: LightGBM LambdaRank parameters
        """
        self.params = params or self._default_params()
        self.model = None
        self.feature_names = None

    def _default_params(self) -> dict:
        """Default LambdaRank parameters"""
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 2, 3],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 500,
            "label_gain": [0, 1, 2, 3, 4, 5],  # Gain for each relevance level
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        group_val: np.ndarray = None,
        feature_names: list = None,
    ) -> None:
        """
        Train LambdaRank model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - relevance scores 0-5
            group_train: Group sizes (n_groups,) - sum must equal n_samples
            X_val: Validation features
            y_val: Validation labels
            group_val: Validation group sizes
            feature_names: List of feature names
        """
        self.feature_names = feature_names

        logger.info("Training LambdaRank model...")
        logger.info(f"  Train samples: {len(X_train)}, groups: {len(group_train)}")
        if X_val is not None:
            logger.info(f"  Val samples: {len(X_val)}, groups: {len(group_val)}")

        # Create LightGBM ranker
        ranker = lgb.LGBMRanker(**self.params)

        # Fit with group parameter
        eval_set = [(X_val, y_val)] if X_val is not None else None
        eval_group = [group_val] if group_val is not None else None

        callbacks = [lgb.early_stopping(50)] if eval_set else None

        ranker.fit(
            X_train, y_train,
            group=group_train,
            eval_set=eval_set,
            eval_group=eval_group,
            callbacks=callbacks,
        )

        self.model = ranker.booster_
        logger.info("Training completed!")

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ranking scores for each sample.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Ranking scores (n_samples,) - higher = better rank
        """
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_race_probs(self, X_race: np.ndarray) -> np.ndarray:
        """
        Predict position probabilities for a single race using Plackett-Luce.

        Args:
            X_race: Features for 6 boats (6, n_features)

        Returns:
            Position probabilities (6, 6) - probs[boat][position]
        """
        if X_race.shape[0] != 6:
            raise ValueError("Expected 6 boats")

        # Get ranking scores
        scores = self.predict_scores(X_race)

        # Convert to Plackett-Luce probabilities
        return self._plackett_luce_probs(scores)

    def _plackett_luce_probs(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert ranking scores to position probabilities via Plackett-Luce.

        Uses Monte Carlo sampling for efficiency.

        Args:
            scores: Ranking scores for 6 boats

        Returns:
            (6, 6) matrix where [i][j] = P(boat i finishes in position j+1)
        """
        # Convert scores to worth parameters
        max_score = np.max(scores)
        alphas = np.exp(scores - max_score)  # Normalized to prevent overflow

        n_boats = len(scores)
        probs = np.zeros((n_boats, n_boats))

        # Monte Carlo sampling
        n_samples = 5000
        np.random.seed(42)  # Deterministic for reproducibility

        for _ in range(n_samples):
            remaining = list(range(n_boats))
            remaining_alphas = alphas.copy()

            for pos in range(n_boats):
                # Probability of each remaining boat taking this position
                total = sum(remaining_alphas[i] for i in remaining)
                boat_probs = [remaining_alphas[i] / total for i in remaining]

                # Sample which boat takes this position
                cumsum = np.cumsum(boat_probs)
                r = np.random.random()
                chosen_idx = np.searchsorted(cumsum, r)
                chosen_idx = min(chosen_idx, len(remaining) - 1)
                chosen_boat = remaining[chosen_idx]

                probs[chosen_boat, pos] += 1
                remaining.remove(chosen_boat)

        probs /= n_samples
        return probs

    def save(self, path: Path = None) -> None:
        """Save the model"""
        path = path or MODEL_DIR / "boatrace_ranker.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "params": self.params,
                "feature_names": self.feature_names,
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path = None) -> None:
        """Load the model"""
        path = path or MODEL_DIR / "boatrace_ranker.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.params = data["params"]
            self.feature_names = data["feature_names"]
        logger.info(f"Model loaded from {path}")


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """
    Optimize hyperparameters using Optuna

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of trials

    Returns:
        Optimal parameters
    """
    def objective(trial):
        params = {
            "objective": "binary",  # Binary classification
            "metric": "binary_logloss",  # Log loss for probability calibration
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "verbose": -1,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
        }

        # Evaluate using the 1st place prediction model
        train_data = lgb.Dataset(X_train, label=y_train[:, 0])
        val_data = lgb.Dataset(X_val, label=y_val[:, 0], reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # Prediction on validation data - use log loss for binary classification
        y_pred = model.predict(X_val)
        # Binary log loss: -mean(y * log(p) + (1-y) * log(1-p))
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = y_val[:, 0]
        logloss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return logloss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.value}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


def train_model(
    use_historical: bool = False,
    optimize: bool = False,
    n_trials: int = 50,
    calibrate: bool = True,
    use_ranking: bool = False,
    include_stadium_course: bool = True,
):
    """
    Main function to train the model

    Args:
        use_historical: Whether to use historical features
        optimize: Whether to perform hyperparameter optimization
        n_trials: Number of optimization trials
        calibrate: Whether to apply Platt scaling calibration (binary only)
        use_ranking: Whether to use LambdaRank instead of binary classification
        include_stadium_course: Whether to include stadium course features

    Returns:
        Trained model (BoatracePredictor or BoatraceRanker)
    """
    logger.info("Building dataset...")
    builder = DatasetBuilder()
    dataset = builder.build_dataset(
        include_historical=use_historical,
        for_ranking=use_ranking,
        include_stadium_course=include_stadium_course,
    )

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    feature_names = dataset["feature_names"]

    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    logger.info(f"Features: {len(feature_names)}")

    if use_ranking:
        # LambdaRank training
        group_train = dataset["group_train"]
        group_val = dataset["group_val"]

        logger.info(f"Train groups: {len(group_train)}")
        logger.info(f"Val groups: {len(group_val)}")

        model = BoatraceRanker()
        model.train(
            X_train, y_train, group_train,
            X_val, y_val, group_val,
            feature_names,
        )
        model.save()
    else:
        # Binary classification training
        # Hyperparameter optimization
        if optimize:
            logger.info("Optimizing hyperparameters...")
            best_params = optimize_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials
            )
            params = {**BoatracePredictor()._default_params(), **best_params}
        else:
            params = None

        # Model training
        model = BoatracePredictor(params=params)
        model.train(X_train, y_train, X_val, y_val, feature_names)

        # Platt scaling calibration using validation set
        if calibrate:
            model.calibrate(X_val, y_val)

        # Save
        model.save()

    return model


def main():
    """Main process"""
    import argparse

    parser = argparse.ArgumentParser(description="Train boatrace prediction model")
    parser.add_argument("--historical", action="store_true", help="Use historical features")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip Platt scaling calibration")
    parser.add_argument("--ranking", action="store_true", help="Use LambdaRank instead of binary classification")
    parser.add_argument("--no-stadium-course", action="store_true", help="Exclude stadium course features (use 50 features)")
    args = parser.parse_args()

    train_model(
        use_historical=args.historical,
        optimize=args.optimize,
        n_trials=args.n_trials,
        calibrate=not args.no_calibrate,
        use_ranking=args.ranking,
        include_stadium_course=not args.no_stadium_course,
    )


if __name__ == "__main__":
    main()
