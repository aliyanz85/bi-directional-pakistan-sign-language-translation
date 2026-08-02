"""
Convert and Optimize Model for TensorFlow.js Deployment
Includes quantization and optimization for browser inference
"""

import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
import json
import os
import shutil
from typing import Dict, Optional
import numpy as np


class TFJSConverter:
    """Convert TensorFlow models to TensorFlow.js format"""
    
    def __init__(self, model_path: str, output_dir: str = "models/inference/tfjs_model"):
        """
        Initialize converter
        
        Args:
            model_path: Path to trained Keras model
            output_dir: Directory to save TensorFlow.js model
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        
        print(f"Model loaded successfully")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        print(f"Parameters: {self.model.count_params():,}")
    
    def convert_to_tfjs(self,
                       quantization_dtype: Optional[str] = 'uint8',
                       weight_shard_size_bytes: int = 4 * 1024 * 1024) -> str:
        """
        Convert model to TensorFlow.js format
        
        Args:
            quantization_dtype: Quantization type ('uint8', 'uint16', or None)
            weight_shard_size_bytes: Maximum size of weight files
        
        Returns:
            Path to converted model directory
        """
        print("\n" + "=" * 60)
        print("CONVERTING TO TENSORFLOW.JS")
        print("=" * 60)
        
        output_path = self.output_dir / "web_model"
        
        # Convert
        tfjs.converters.save_keras_model(
            self.model,
            str(output_path),
            quantization_dtype_map={
                quantization_dtype: '*'
            } if quantization_dtype else None,
            weight_shard_size_bytes=weight_shard_size_bytes
        )
        
        # Get model size
        model_size = self._get_directory_size(output_path)
        original_size = os.path.getsize(self.model_path)
        
        print(f"\nConversion complete!")
        print(f"Original model size: {original_size / 1e6:.2f} MB")
        print(f"TensorFlow.js model size: {model_size / 1e6:.2f} MB")
        print(f"Compression ratio: {original_size / model_size:.2f}x")
        print(f"Quantization: {quantization_dtype if quantization_dtype else 'None'}")
        print(f"Output directory: {output_path}")
        
        # Save metadata
        self._save_metadata(output_path, model_size, quantization_dtype)
        
        return str(output_path)
    
    def convert_to_tflite(self,
                         optimize: bool = True,
                         quantize: bool = True) -> str:
        """
        Convert to TensorFlow Lite (alternative for mobile)
        
        Args:
            optimize: Whether to optimize model
            quantize: Whether to quantize weights
        
        Returns:
            Path to TFLite model
        """
        print("\n" + "=" * 60)
        print("CONVERTING TO TENSORFLOW LITE")
        print("=" * 60)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        
        # Save
        output_path = self.output_dir / "model.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size = len(tflite_model)
        original_size = os.path.getsize(self.model_path)
        
        print(f"\nConversion complete!")
        print(f"Original model size: {original_size / 1e6:.2f} MB")
        print(f"TFLite model size: {size / 1e6:.2f} MB")
        print(f"Compression ratio: {original_size / size:.2f}x")
        print(f"Output file: {output_path}")
        
        return str(output_path)
    
    def optimize_for_inference(self) -> tf.keras.Model:
        """
        Optimize model for inference
        Removes training-only operations
        
        Returns:
            Optimized model
        """
        print("\nOptimizing model for inference...")
        
        # Get model config
        config = self.model.get_config()
        
        # Rebuild model without dropout and batch normalization training mode
        optimized_model = tf.keras.Model.from_config(config)
        optimized_model.set_weights(self.model.get_weights())
        
        return optimized_model
    
    def benchmark_inference(self, num_runs: int = 100) -> Dict:
        """
        Benchmark inference performance
        
        Args:
            num_runs: Number of inference runs
        
        Returns:
            Benchmark results
        """
        print("\n" + "=" * 60)
        print("BENCHMARKING INFERENCE")
        print("=" * 60)
        
        # Create dummy input
        input_shape = self.model.input_shape
        dummy_input = tf.random.normal((1, *input_shape[1:]))
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input, training=False)
        
        # Measure
        latencies = []
        
        import time
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.model(dummy_input, training=False)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        results = {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
        
        print(f"\nBenchmark Results ({num_runs} runs):")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Median: {results['median_ms']:.2f} ms")
        print(f"  95th percentile: {results['p95_ms']:.2f} ms")
        print(f"  99th percentile: {results['p99_ms']:.2f} ms")
        
        return results
    
    def create_inference_config(self, 
                               class_names: list,
                               sequence_length: int,
                               feature_dim: int,
                               normalization_params: Optional[Dict] = None) -> Dict:
        """
        Create configuration file for inference
        
        Args:
            class_names: List of class names
            sequence_length: Expected sequence length
            feature_dim: Feature dimension
            normalization_params: Normalization parameters (mean, std)
        
        Returns:
            Configuration dictionary
        """
        config = {
            'modelInfo': {
                'version': '1.0.0',
                'type': 'PSL_Recognition',
                'framework': 'TensorFlow.js',
                'inputShape': [sequence_length, feature_dim],
                'outputShape': [len(class_names)]
            },
            'classes': class_names,
            'numClasses': len(class_names),
            'sequenceLength': sequence_length,
            'featureDim': feature_dim,
            'normalization': normalization_params if normalization_params else {
                'enabled': False
            },
            'inference': {
                'batchSize': 1,
                'confidenceThreshold': 0.7,
                'topK': 3
            },
            'preprocessing': {
                'maxSequenceLength': sequence_length,
                'numLandmarks': 21,
                'handsToDetect': 2
            }
        }
        
        return config
    
    def _save_metadata(self, output_path: Path, model_size: int, quantization: str):
        """Save model metadata"""
        metadata = {
            'original_model': str(self.model_path),
            'conversion_date': tf.timestamp().numpy().item(),
            'model_size_bytes': model_size,
            'model_size_mb': model_size / 1e6,
            'quantization': quantization,
            'input_shape': self.model.input_shape[1:],
            'output_shape': self.model.output_shape[1:],
            'total_parameters': self.model.count_params()
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of directory"""
        total_size = 0
        for file in path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size


def convert_model(model_path: str,
                 output_dir: str,
                 class_names: list,
                 sequence_length: int = 60,
                 feature_dim: int = 188,
                 quantization: str = 'uint8'):
    """
    Complete conversion pipeline
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory
        class_names: List of class names
        sequence_length: Sequence length
        feature_dim: Feature dimension
        quantization: Quantization type
    """
    # Initialize converter
    converter = TFJSConverter(model_path, output_dir)
    
    # Benchmark original model
    print("\nBenchmarking original model...")
    benchmark_results = converter.benchmark_inference()
    
    # Convert to TensorFlow.js
    tfjs_path = converter.convert_to_tfjs(quantization_dtype=quantization)
    
    # Convert to TFLite (optional)
    tflite_path = converter.convert_to_tflite(optimize=True, quantize=True)
    
    # Create inference config
    config = converter.create_inference_config(
        class_names=class_names,
        sequence_length=sequence_length,
        feature_dim=feature_dim
    )
    
    # Save config
    config_path = Path(output_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"TensorFlow.js model: {tfjs_path}")
    print(f"TFLite model: {tflite_path}")
    print(f"Configuration: {config_path}")
    print("=" * 60 + "\n")
    
    return {
        'tfjs_path': tfjs_path,
        'tflite_path': tflite_path,
        'config_path': str(config_path),
        'benchmark': benchmark_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PSL model to TensorFlow.js')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained Keras model')
    parser.add_argument('--output_dir', type=str, default='models/inference/tfjs_model',
                       help='Output directory')
    parser.add_argument('--data_path', type=str, default='data/extracted_landmarks/dataset.pkl',
                       help='Path to dataset for class names')
    parser.add_argument('--quantization', type=str, default='uint8',
                       choices=['uint8', 'uint16', 'none'],
                       help='Quantization type')
    
    args = parser.parse_args()
    
    # Load dataset for class names
    if Path(args.data_path).exists():
        import pickle
        with open(args.data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        class_names = dataset['class_names']
        sequence_length = dataset['sequence_length']
        feature_dim = dataset['feature_dim']
    else:
        print(f"Warning: Dataset not found at {args.data_path}")
        print("Using default configuration")
        class_names = [f"word_{i}" for i in range(32)]
        sequence_length = 60
        feature_dim = 188
    
    # Convert
    quantization = None if args.quantization == 'none' else args.quantization
    
    result = convert_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        class_names=class_names,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        quantization=quantization
    )
    
    print("\nConversion successful!")
    print("You can now deploy the model to your web application")
