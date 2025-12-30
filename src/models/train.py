"""
モデル訓練

LightGBMを使用して着順予測モデルを訓練
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT
from src.models.dataset import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# モデル保存ディレクトリ
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class BoatracePredictor:
    """競艇予測モデル"""

    def __init__(self, params: dict = None):
        """
        Args:
            params: LightGBMのパラメータ
        """
        self.params = params or self._default_params()
        self.models = None
        self.feature_names = None

    def _default_params(self) -> dict:
        """デフォルトのLightGBMパラメータ"""
        return {
            "objective": "regression",
            "metric": "rmse",
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
        モデルを訓練

        Args:
            X_train: 訓練特徴量 (n_samples, n_features)
            y_train: 訓練ラベル (n_samples, 6) - 各着順の確率
            X_val: 検証特徴量
            y_val: 検証ラベル
            feature_names: 特徴量名のリスト
        """
        self.feature_names = feature_names

        # 6つの着順それぞれに対するモデルを訓練
        self.models = []

        for pos in range(6):
            logger.info(f"Training model for position {pos + 1}...")

            # LightGBMデータセット
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

            # 訓練
            model = lgb.train(
                self.params,
                train_data,
                valid_sets=[val_data] if val_data else None,
                callbacks=callbacks,
            )

            self.models.append(model)

        logger.info("Training completed!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        着順確率を予測

        Args:
            X: 特徴量 (n_samples, n_features)

        Returns:
            確率 (n_samples, 6) - 各着順の予測確率
        """
        if self.models is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = np.zeros((X.shape[0], 6))

        for pos, model in enumerate(self.models):
            predictions[:, pos] = model.predict(X)

        # 確率に変換（0-1に正規化）
        predictions = np.clip(predictions, 0, 1)

        # 行ごとに正規化（合計が1になるように）
        row_sums = predictions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        predictions = predictions / row_sums

        return predictions

    def save(self, path: Path = None) -> None:
        """モデルを保存"""
        path = path or MODEL_DIR / "boatrace_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "params": self.params,
                "feature_names": self.feature_names,
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path = None) -> None:
        """モデルを読み込み"""
        path = path or MODEL_DIR / "boatrace_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.models = data["models"]
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
    Optunaでハイパーパラメータを最適化

    Args:
        X_train: 訓練特徴量
        y_train: 訓練ラベル
        X_val: 検証特徴量
        y_val: 検証ラベル
        n_trials: 試行回数

    Returns:
        最適なパラメータ
    """
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
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

        # 1位予測のモデルで評価
        train_data = lgb.Dataset(X_train, label=y_train[:, 0])
        val_data = lgb.Dataset(X_val, label=y_val[:, 0], reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # 検証データでの予測
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_pred - y_val[:, 0]) ** 2))

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.value}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


def train_model(
    use_historical: bool = False,
    optimize: bool = False,
    n_trials: int = 50,
) -> BoatracePredictor:
    """
    モデルを訓練するメイン関数

    Args:
        use_historical: 履歴特徴量を使用するか
        optimize: ハイパーパラメータ最適化を行うか
        n_trials: 最適化の試行回数

    Returns:
        訓練済みモデル
    """
    logger.info("Building dataset...")
    builder = DatasetBuilder()
    dataset = builder.build_dataset(include_historical=use_historical)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    feature_names = dataset["feature_names"]

    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    logger.info(f"Features: {len(feature_names)}")

    # ハイパーパラメータ最適化
    if optimize:
        logger.info("Optimizing hyperparameters...")
        best_params = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials
        )
        params = {**BoatracePredictor()._default_params(), **best_params}
    else:
        params = None

    # モデル訓練
    model = BoatracePredictor(params=params)
    model.train(X_train, y_train, X_val, y_val, feature_names)

    # 保存
    model.save()

    return model


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="Train boatrace prediction model")
    parser.add_argument("--historical", action="store_true", help="Use historical features")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    args = parser.parse_args()

    train_model(
        use_historical=args.historical,
        optimize=args.optimize,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
