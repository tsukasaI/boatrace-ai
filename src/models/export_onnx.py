"""
ONNX Export Script

Export LightGBM model to ONNX format
For use with Rust API
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
from src.models.train import BoatracePredictor, BoatraceRanker

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
    n_features: int = None,
) -> list[Path]:
    """
    Export LightGBM model to ONNX

    Args:
        model_path: Input model path (.pkl)
        output_dir: Output directory
        n_features: Number of features (auto-detected from model if None)

    Returns:
        List of exported ONNX file paths
    """
    model_path = model_path or MODEL_DIR / "boatrace_model.pkl"
    output_dir = output_dir or ONNX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    predictor = BoatracePredictor()
    predictor.load(model_path)

    if predictor.models is None:
        raise ValueError("No models found in predictor")

    # Auto-detect feature count from model
    if n_features is None:
        if predictor.feature_names:
            n_features = len(predictor.feature_names)
            logger.info(f"Auto-detected {n_features} features from model")
        else:
            n_features = 14  # Fallback to default
            logger.warning(f"Could not detect features, using default: {n_features}")

    # Define input type
    initial_type = [("input", FloatTensorType([None, n_features]))]

    exported_paths = []

    for i, model in enumerate(predictor.models):
        output_path = output_dir / f"position_{i + 1}.onnx"
        logger.info(f"Exporting position {i + 1} model to {output_path}")

        # Convert from LightGBM to ONNX
        # zipmap=False produces simple probability arrays instead of map structures
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=12,
            zipmap=False,
        )

        # Save model
        onnx.save_model(onnx_model, str(output_path))
        exported_paths.append(output_path)

        logger.info(f"  Saved: {output_path}")

    # Create metadata file
    metadata_path = output_dir / "metadata.json"
    import json
    metadata = {
        "n_models": len(predictor.models),
        "n_features": n_features,
        "feature_names": predictor.feature_names,
        "model_files": [f"position_{i + 1}.onnx" for i in range(len(predictor.models))],
    }

    # Export Platt scaling calibrators if available
    if predictor.calibrators is not None:
        calibrators_data = []
        for i, cal in enumerate(predictor.calibrators):
            # LogisticRegression stores coef_ as 2D array [[coef]] and intercept_ as 1D [intercept]
            coef = float(cal.coef_[0, 0])
            intercept = float(cal.intercept_[0])
            calibrators_data.append({
                "position": i + 1,
                "coef": coef,
                "intercept": intercept,
            })
            logger.info(f"  Position {i + 1} calibrator: coef={coef:.4f}, intercept={intercept:.4f}")
        metadata["calibrators"] = calibrators_data
        logger.info(f"Exported {len(calibrators_data)} Platt scaling calibrators")
    else:
        logger.warning("No calibrators found in model - predictions will be uncalibrated")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return exported_paths


def export_ranker_to_onnx(
    model_path: Path = None,
    output_dir: Path = None,
    n_features: int = None,
) -> Path:
    """
    Export LambdaRank model to ONNX

    Args:
        model_path: Input model path (.pkl)
        output_dir: Output directory
        n_features: Number of features (auto-detected from model if None)

    Returns:
        Exported ONNX file path
    """
    model_path = model_path or MODEL_DIR / "boatrace_ranker.pkl"
    output_dir = output_dir or ONNX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading ranker model from {model_path}")
    ranker = BoatraceRanker()
    ranker.load(model_path)

    if ranker.model is None:
        raise ValueError("No model found in ranker")

    # Auto-detect feature count from model
    if n_features is None:
        if ranker.feature_names:
            n_features = len(ranker.feature_names)
            logger.info(f"Auto-detected {n_features} features from model")
        else:
            n_features = 50  # Fallback to default
            logger.warning(f"Could not detect features, using default: {n_features}")

    # Define input type
    initial_type = [("input", FloatTensorType([None, n_features]))]

    output_path = output_dir / "ranker.onnx"
    logger.info(f"Exporting ranker model to {output_path}")

    # Convert from LightGBM to ONNX
    # Note: onnxmltools doesn't support lambdarank directly, but the inference
    # is identical to regression (just outputs scores). We work around this
    # by creating a copy of the model with modified objective.
    import tempfile
    import lightgbm as lgb

    # Save the booster to a model string, modify objective, reload
    model_str = ranker.model.model_to_string()

    # Replace objective with regression (inference is identical)
    lines = model_str.split('\n')
    modified_lines = []
    for line in lines:
        if line.startswith('[objective:'):
            modified_lines.append('[objective: regression]')
        elif line.startswith('objective='):
            modified_lines.append('objective=regression')
        else:
            modified_lines.append(line)

    modified_str = '\n'.join(modified_lines)

    # Create a temporary booster from modified string
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(modified_str)
        temp_path = f.name

    import os
    try:
        modified_booster = lgb.Booster(model_file=temp_path)
        onnx_model = convert_lightgbm(
            modified_booster,
            initial_types=initial_type,
            target_opset=12,
        )
    finally:
        os.unlink(temp_path)

    # Save model
    onnx.save_model(onnx_model, str(output_path))
    logger.info(f"  Saved: {output_path}")

    # Create metadata file
    import json
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "model_type": "lambdarank",
        "n_models": 1,
        "n_features": n_features,
        "feature_names": ranker.feature_names,
        "model_files": ["ranker.onnx"],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return output_path


def verify_onnx_models(
    onnx_dir: Path = None,
    n_features: int = None,
) -> bool:
    """
    Verify exported ONNX models

    Args:
        onnx_dir: Directory containing ONNX models
        n_features: Number of features (auto-detected from metadata if None)

    Returns:
        Whether verification succeeded
    """
    onnx_dir = onnx_dir or ONNX_DIR

    # Auto-detect feature count from metadata
    if n_features is None:
        import json
        metadata_path = onnx_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            n_features = metadata.get("n_features", 14)
            logger.info(f"Auto-detected {n_features} features from metadata")
        else:
            n_features = 14
            logger.warning(f"No metadata found, using default: {n_features}")

    # Create test input
    test_input = np.random.randn(6, n_features).astype(np.float32)

    logger.info("Verifying ONNX models...")

    for i in range(6):
        onnx_path = onnx_dir / f"position_{i + 1}.onnx"

        if not onnx_path.exists():
            logger.error(f"Model not found: {onnx_path}")
            return False

        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Inference test
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: test_input})

        logger.info(f"  Position {i + 1}: OK (output shape: {output[0].shape})")

    logger.info("All models verified successfully!")
    return True


def compare_predictions(
    model_path: Path = None,
    onnx_dir: Path = None,
    n_features: int = None,
    n_samples: int = 100,
) -> dict:
    """
    Compare predictions between LightGBM and ONNX

    Args:
        model_path: LightGBM model path
        onnx_dir: Directory containing ONNX models
        n_features: Number of features (auto-detected from model if None)
        n_samples: Number of test samples

    Returns:
        Comparison results
    """
    model_path = model_path or MODEL_DIR / "boatrace_model.pkl"
    onnx_dir = onnx_dir or ONNX_DIR

    # Load LightGBM model
    predictor = BoatracePredictor()
    predictor.load(model_path)

    # Auto-detect feature count from model
    if n_features is None:
        if predictor.feature_names:
            n_features = len(predictor.feature_names)
            logger.info(f"Auto-detected {n_features} features from model")
        else:
            n_features = 14
            logger.warning(f"Could not detect features, using default: {n_features}")

    # Create test input
    test_input = np.random.randn(n_samples, n_features).astype(np.float32)

    results = {"max_diff": [], "mean_diff": []}

    logger.info(f"Comparing predictions on {n_samples} samples...")

    for i in range(6):
        # LightGBM prediction
        lgb_pred = predictor.models[i].predict(test_input)

        # ONNX prediction
        onnx_path = onnx_dir / f"position_{i + 1}.onnx"
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_pred = session.run(None, {input_name: test_input})[0].flatten()

        # Calculate difference
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
        help="Input model path (default: models/boatrace_model.pkl or boatrace_ranker.pkl)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: models/onnx)"
    )
    parser.add_argument(
        "--features", type=int, default=None,
        help="Number of features (auto-detected from model if not specified)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify exported models"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare LightGBM and ONNX predictions"
    )
    parser.add_argument(
        "--ranking", action="store_true",
        help="Export LambdaRank ranker model instead of binary classifiers"
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    output_dir = Path(args.output) if args.output else None

    if args.ranking:
        # Export LambdaRank ranker
        exported_path = export_ranker_to_onnx(
            model_path=model_path,
            output_dir=output_dir,
            n_features=args.features,
        )
        logger.info(f"Exported ranker model: {exported_path}")
    else:
        # Export binary classifiers
        exported_paths = export_to_onnx(
            model_path=model_path,
            output_dir=output_dir,
            n_features=args.features,
        )
        logger.info(f"Exported {len(exported_paths)} models")

        # Verify (only for binary classifiers currently)
        if args.verify:
            verify_onnx_models(output_dir, args.features)

        # Compare
        if args.compare:
            compare_predictions(model_path, output_dir, args.features)


if __name__ == "__main__":
    main()
