"""
ONNX Export Script

LightGBMモデルをONNX形式にエクスポート
Rust APIで使用するため
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT
from src.models.train import BoatracePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"
ONNX_DIR = MODEL_DIR / "onnx"


def export_to_onnx(
    model_path: Path = None,
    output_dir: Path = None,
    n_features: int = 14,
) -> list[Path]:
    """
    LightGBMモデルをONNXにエクスポート

    Args:
        model_path: 入力モデルパス (.pkl)
        output_dir: 出力ディレクトリ
        n_features: 特徴量の数

    Returns:
        エクスポートされたONNXファイルのパスリスト
    """
    model_path = model_path or MODEL_DIR / "boatrace_model.pkl"
    output_dir = output_dir or ONNX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルを読み込み
    logger.info(f"Loading model from {model_path}")
    predictor = BoatracePredictor()
    predictor.load(model_path)

    if predictor.models is None:
        raise ValueError("No models found in predictor")

    # 入力の型定義
    initial_type = [("input", FloatTensorType([None, n_features]))]

    exported_paths = []

    for i, model in enumerate(predictor.models):
        output_path = output_dir / f"position_{i + 1}.onnx"
        logger.info(f"Exporting position {i + 1} model to {output_path}")

        # LightGBMからONNXに変換
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=12,
        )

        # モデルを保存
        onnx.save_model(onnx_model, str(output_path))
        exported_paths.append(output_path)

        logger.info(f"  Saved: {output_path}")

    # メタデータファイルを作成
    metadata_path = output_dir / "metadata.json"
    import json
    metadata = {
        "n_models": len(predictor.models),
        "n_features": n_features,
        "feature_names": predictor.feature_names,
        "model_files": [f"position_{i + 1}.onnx" for i in range(len(predictor.models))],
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return exported_paths


def verify_onnx_models(
    onnx_dir: Path = None,
    n_features: int = 14,
) -> bool:
    """
    エクスポートしたONNXモデルを検証

    Args:
        onnx_dir: ONNXモデルのディレクトリ
        n_features: 特徴量の数

    Returns:
        検証成功かどうか
    """
    onnx_dir = onnx_dir or ONNX_DIR

    # テスト入力を作成
    test_input = np.random.randn(6, n_features).astype(np.float32)

    logger.info("Verifying ONNX models...")

    for i in range(6):
        onnx_path = onnx_dir / f"position_{i + 1}.onnx"

        if not onnx_path.exists():
            logger.error(f"Model not found: {onnx_path}")
            return False

        # ONNXモデルを検証
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # 推論テスト
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: test_input})

        logger.info(f"  Position {i + 1}: OK (output shape: {output[0].shape})")

    logger.info("All models verified successfully!")
    return True


def compare_predictions(
    model_path: Path = None,
    onnx_dir: Path = None,
    n_features: int = 14,
    n_samples: int = 100,
) -> dict:
    """
    LightGBMとONNXの予測結果を比較

    Args:
        model_path: LightGBMモデルパス
        onnx_dir: ONNXモデルのディレクトリ
        n_features: 特徴量の数
        n_samples: テストサンプル数

    Returns:
        比較結果
    """
    model_path = model_path or MODEL_DIR / "boatrace_model.pkl"
    onnx_dir = onnx_dir or ONNX_DIR

    # LightGBMモデルを読み込み
    predictor = BoatracePredictor()
    predictor.load(model_path)

    # テスト入力を作成
    test_input = np.random.randn(n_samples, n_features).astype(np.float32)

    results = {"max_diff": [], "mean_diff": []}

    logger.info(f"Comparing predictions on {n_samples} samples...")

    for i in range(6):
        # LightGBM予測
        lgb_pred = predictor.models[i].predict(test_input)

        # ONNX予測
        onnx_path = onnx_dir / f"position_{i + 1}.onnx"
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_pred = session.run(None, {input_name: test_input})[0].flatten()

        # 差分を計算
        diff = np.abs(lgb_pred - onnx_pred)
        max_diff = diff.max()
        mean_diff = diff.mean()

        results["max_diff"].append(max_diff)
        results["mean_diff"].append(mean_diff)

        logger.info(f"  Position {i + 1}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    logger.info(f"Overall max diff: {max(results['max_diff']):.6f}")
    logger.info(f"Overall mean diff: {np.mean(results['mean_diff']):.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Export LightGBM model to ONNX")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Input model path (default: models/boatrace_model.pkl)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: models/onnx)"
    )
    parser.add_argument(
        "--features", type=int, default=14,
        help="Number of features (default: 14)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify exported models"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare LightGBM and ONNX predictions"
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    output_dir = Path(args.output) if args.output else None

    # エクスポート
    exported_paths = export_to_onnx(
        model_path=model_path,
        output_dir=output_dir,
        n_features=args.features,
    )

    logger.info(f"Exported {len(exported_paths)} models")

    # 検証
    if args.verify:
        verify_onnx_models(output_dir, args.features)

    # 比較
    if args.compare:
        compare_predictions(model_path, output_dir, args.features)


if __name__ == "__main__":
    main()
