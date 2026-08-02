"""
PSL (Pakistan Sign Language) Inference Service

Loads the trained TCN + Transformer model and provides inference functionality.
Model expects input shape (60, 188) - 60 frames of 188-dimensional feature vectors.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import TensorFlow (heavy dependency)
_model = None
_normalization_params = None
_class_labels = None
_model_loaded = False
_custom_objects = None


def _get_model_paths():
    """Get absolute paths to model files"""
    # Navigate from voice-backend/app/services/ to transformer/models/saved_models/
    base_dir = Path(__file__).parent.parent.parent.parent
    model_dir = base_dir / "transformer" / "models" / "saved_models" / "tcn_transformer_20260417_011511"
    metadata_file = base_dir / "transformer" / "data" / "extracted_landmarks" / "metadata.json"
    training_module = base_dir / "transformer" / "models" / "training"

    return {
        "model": model_dir / "best_model.h5",
        "normalization": model_dir / "normalization_params.json",
        "metadata": metadata_file,
        "training_module": training_module
    }


def load_model():
    """
    Load the trained model, normalization parameters, and class labels.
    This function is called once on application startup.
    """
    global _model, _normalization_params, _class_labels, _model_loaded, _custom_objects

    if _model_loaded:
        logger.info("Model already loaded, skipping...")
        return

    try:
        # Import TensorFlow only when needed
        import tensorflow as tf

        paths = _get_model_paths()

        # Verify all files exist
        for name, path in paths.items():
            if name == "training_module":
                continue  # This is a directory, not a file
            if not path.exists():
                raise FileNotFoundError(f"{name.capitalize()} file not found at: {path}")

        logger.info("Loading PSL recognition model...")

        # Add the training module to sys.path to import custom layers
        training_module_path = str(paths['training_module'])
        if training_module_path not in sys.path:
            sys.path.insert(0, training_module_path)
        
        # Import custom layers from the transformer training module
        from model_architecture import TCNBlock, TransformerBlock, PositionalEncoding
        
        # Create custom objects dictionary for model loading
        _custom_objects = {
            'TCNBlock': TCNBlock,
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }

        # Load the Keras model with custom objects
        logger.info(f"Loading model from: {paths['model']}")
        _model = tf.keras.models.load_model(str(paths['model']), custom_objects=_custom_objects, compile=False)
        logger.info(f"Model loaded successfully. Input shape: {_model.input_shape}")

        # Load normalization parameters
        logger.info(f"Loading normalization params from: {paths['normalization']}")
        with open(paths['normalization'], 'r') as f:
            _normalization_params = json.load(f)

        # Convert to numpy arrays for faster computation
        _normalization_params['mean'] = np.array(_normalization_params['mean'])
        _normalization_params['std'] = np.array(_normalization_params['std'])

        # Validate normalization params shape
        if len(_normalization_params['mean']) != 188 or len(_normalization_params['std']) != 188:
            raise ValueError(f"Normalization params must have 188 features, got mean: {len(_normalization_params['mean'])}, std: {len(_normalization_params['std'])}")

        logger.info(f"Normalization params loaded: mean shape {_normalization_params['mean'].shape}, std shape {_normalization_params['std'].shape}")

        # Load metadata (class labels)
        logger.info(f"Loading metadata from: {paths['metadata']}")
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        # Check for class_names first, then classes (for compatibility)
        _class_labels = metadata.get('class_names', metadata.get('classes', []))

        expected_classes = int(_model.output_shape[-1])
        if len(_class_labels) != expected_classes:
            raise ValueError(f"Class label count mismatch. Model outputs {expected_classes}, metadata has {len(_class_labels)}")

        logger.info(f"Loaded {len(_class_labels)} class labels: {', '.join(_class_labels[:5])}...")

        _model_loaded = True
        logger.info("PSL model initialization complete!")

    except Exception as e:
        logger.error(f"Failed to load PSL model: {str(e)}")
        raise


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to the input sequence.

    Args:
        sequence: numpy array of shape (60, 188)

    Returns:
        Normalized sequence of shape (60, 188)
    """
    if _normalization_params is None:
        raise RuntimeError("Normalization parameters not loaded. Call load_model() first.")

    mean = _normalization_params['mean']
    std = _normalization_params['std']

    # Avoid division by zero
    std_safe = np.where(std == 0, 1.0, std)

    # Apply normalization: (x - mean) / std
    normalized = (sequence - mean) / std_safe

    return normalized


def predict_psl(sequence: List[List[float]]) -> Dict:
    """
    Run PSL inference on a sequence of feature vectors.

    Args:
        sequence: List of 60 frames, each containing 188 features.
                 Shape should be (60, 188).

    Returns:
        Dictionary containing:
        - label: str (predicted PSL word)
        - class_id: int (predicted class index)
        - confidence: float (confidence score 0-1)
        - top_predictions: list of top 5 predictions with labels and scores

    Raises:
        ValueError: If input shape is invalid
        RuntimeError: If model is not loaded
    """
    if not _model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Validate input
    if not isinstance(sequence, (list, np.ndarray)):
        raise ValueError("Sequence must be a list or numpy array")

    # Convert to numpy array
    sequence_array = np.array(sequence, dtype=np.float32)

    # Validate shape
    if sequence_array.shape != (60, 188):
        raise ValueError(
            f"Invalid input shape. Expected (60, 188), got {sequence_array.shape}. "
            f"Sequence must contain exactly 60 frames of 188 features each."
        )

    # Check for NaN or Inf values
    if np.isnan(sequence_array).any():
        raise ValueError("Input sequence contains NaN values")
    if np.isinf(sequence_array).any():
        raise ValueError("Input sequence contains infinite values")

    try:
        # Normalize the sequence
        normalized_sequence = normalize_sequence(sequence_array)

        # Add batch dimension: (60, 188) -> (1, 60, 188)
        batch_input = np.expand_dims(normalized_sequence, axis=0)

        # Run inference
        predictions = _model.predict(batch_input, verbose=0)

        # predictions shape: (1, num_classes)
        # Extract the single prediction
        probs = predictions[0]

        # Get predicted class
        predicted_class_id = int(np.argmax(probs))
        predicted_confidence = float(probs[predicted_class_id])
        predicted_label = _class_labels[predicted_class_id]

        # Get top 5 predictions
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]  # Descending order

        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "label": _class_labels[int(idx)],
                "class_id": int(idx),
                "confidence": float(probs[idx])
            })

        result = {
            "label": predicted_label,
            "class_id": predicted_class_id,
            "confidence": predicted_confidence,
            "top_predictions": top_predictions
        }

        logger.info(f"Prediction: {predicted_label} (confidence: {predicted_confidence:.3f})")

        return result

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise


def get_model_info() -> Dict:
    """
    Get information about the loaded model.

    Returns:
        Dictionary with model metadata
    """
    if not _model_loaded:
        return {"loaded": False, "error": "Model not loaded"}

    return {
        "loaded": True,
        "input_shape": str(_model.input_shape),
        "output_shape": str(_model.output_shape),
        "num_classes": len(_class_labels),
        "classes": _class_labels,
        "normalization_features": len(_normalization_params['mean'])
    }


# Auto-load model when module is imported (optional, can be called explicitly from main.py)
# Uncomment the line below to auto-load on import:
# load_model()
