"""
Model Evaluation Suite for PSL Recognition
Comprehensive evaluation metrics and analysis
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import pandas as pd


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, class_names: List[str]):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate(self, 
                X_test: np.ndarray,
                y_test: np.ndarray,
                output_dir: str = None) -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            X_test: Test sequences
            y_test: Test labels (one-hot encoded)
            output_dir: Directory to save results
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60 + "\n")
        
        results = {}
        
        # Get predictions
        print("Generating predictions...")
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Basic metrics
        results['accuracy'] = self.calculate_accuracy(y_true, y_pred)
        results['top3_accuracy'] = self.calculate_topk_accuracy(y_test, y_pred_probs, k=3)
        results['top5_accuracy'] = self.calculate_topk_accuracy(y_test, y_pred_probs, k=5)
        
        # Per-class metrics
        results['per_class_metrics'] = self.per_class_analysis(y_true, y_pred)
        
        # Confusion matrix
        results['confusion_matrix'] = self.compute_confusion_matrix(y_true, y_pred)
        
        # Latency analysis
        results['latency'] = self.measure_latency(X_test)
        
        # Confidence analysis
        results['confidence_stats'] = self.analyze_confidence(y_pred_probs, y_true)
        
        # Error analysis
        results['error_analysis'] = self.analyze_errors(y_true, y_pred, y_pred_probs)
        
        # Print summary
        self._print_evaluation_summary(results)
        
        # Save results
        if output_dir:
            self._save_results(results, output_dir)
            self._plot_results(results, output_dir, y_true, y_pred, y_pred_probs)
        
        return results
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate overall accuracy"""
        return accuracy_score(y_true, y_pred)
    
    def calculate_topk_accuracy(self, 
                               y_true: np.ndarray,
                               y_pred_probs: np.ndarray,
                               k: int = 3) -> float:
        """
        Calculate top-k accuracy
        
        Args:
            y_true: True labels (one-hot)
            y_pred_probs: Prediction probabilities
            k: Top k predictions to consider
        
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
        y_true_labels = np.argmax(y_true, axis=1)
        
        correct = 0
        for true_label, top_k in zip(y_true_labels, top_k_preds):
            if true_label in top_k:
                correct += 1
        
        return correct / len(y_true_labels)
    
    def per_class_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Detailed per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with per-class metrics
        """
        # Get unique labels present in the dataset
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        labels_list = sorted(unique_labels.tolist())
        target_names_subset = [self.class_names[i] for i in labels_list]
        
        report = classification_report(
            y_true, y_pred,
            labels=labels_list,
            target_names=target_names_subset,
            output_dict=True,
            zero_division=0
        )
        
        # Identify problematic classes
        low_performing = {}
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and metrics.get('f1-score', 1.0) < 0.7:
                low_performing[class_name] = metrics
        
        return {
            'classification_report': report,
            'low_performing_classes': low_performing
        }
    
    def compute_confusion_matrix(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def measure_latency(self, 
                       X_test: np.ndarray,
                       num_samples: int = 100) -> Dict:
        """
        Measure inference latency
        
        Args:
            X_test: Test sequences
            num_samples: Number of samples to measure
        
        Returns:
            Dictionary with latency statistics
        """
        print("Measuring inference latency...")
        
        latencies = []
        num_samples = min(num_samples, len(X_test))
        
        # Warmup
        _ = self.model.predict(X_test[:1], verbose=0)
        
        # Measure
        for i in range(num_samples):
            sample = X_test[i:i+1]
            
            start = time.perf_counter()
            _ = self.model.predict(sample, verbose=0)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
    
    def analyze_confidence(self, 
                          y_pred_probs: np.ndarray,
                          y_true: np.ndarray) -> Dict:
        """
        Analyze prediction confidence
        
        Args:
            y_pred_probs: Prediction probabilities
            y_true: True labels
        
        Returns:
            Confidence statistics
        """
        max_probs = np.max(y_pred_probs, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Confidence for correct vs incorrect predictions
        correct_mask = (y_pred == y_true)
        correct_confidence = max_probs[correct_mask]
        incorrect_confidence = max_probs[~correct_mask]
        
        return {
            'overall': {
                'mean': float(np.mean(max_probs)),
                'median': float(np.median(max_probs)),
                'std': float(np.std(max_probs))
            },
            'correct_predictions': {
                'mean': float(np.mean(correct_confidence)) if len(correct_confidence) > 0 else 0,
                'median': float(np.median(correct_confidence)) if len(correct_confidence) > 0 else 0
            },
            'incorrect_predictions': {
                'mean': float(np.mean(incorrect_confidence)) if len(incorrect_confidence) > 0 else 0,
                'median': float(np.median(incorrect_confidence)) if len(incorrect_confidence) > 0 else 0
            }
        }
    
    def analyze_errors(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_pred_probs: np.ndarray) -> Dict:
        """
        Analyze prediction errors
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_probs: Prediction probabilities
        
        Returns:
            Error analysis
        """
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        error_details = []
        for idx in error_indices[:20]:  # Top 20 errors
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = y_pred_probs[idx, y_pred[idx]]
            
            error_details.append({
                'sample_idx': int(idx),
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': float(confidence)
            })
        
        # Most confused pairs
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=sorted(unique_labels))
        np.fill_diagonal(cm, 0)  # Remove correct predictions
        
        confused_pairs = []
        for i, true_label in enumerate(sorted(unique_labels)):
            for j, pred_label in enumerate(sorted(unique_labels)):
                if cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': self.class_names[true_label],
                        'predicted_class': self.class_names[pred_label],
                        'count': int(cm[i, j])
                    })
        
        # Sort by frequency
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.mean()),
            'sample_errors': error_details,
            'most_confused_pairs': confused_pairs[:10]
        }
    
    def _print_evaluation_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        
        print(f"\nLatency Statistics:")
        print(f"  Mean: {results['latency']['mean_ms']:.2f} ms")
        print(f"  Median: {results['latency']['median_ms']:.2f} ms")
        print(f"  95th percentile: {results['latency']['p95_ms']:.2f} ms")
        print(f"  99th percentile: {results['latency']['p99_ms']:.2f} ms")
        
        print(f"\nConfidence Statistics:")
        print(f"  Correct predictions: {results['confidence_stats']['correct_predictions']['mean']:.4f}")
        print(f"  Incorrect predictions: {results['confidence_stats']['incorrect_predictions']['mean']:.4f}")
        
        if results['per_class_metrics']['low_performing_classes']:
            print(f"\nLow-performing classes (F1 < 0.7):")
            for cls, metrics in results['per_class_metrics']['low_performing_classes'].items():
                print(f"  {cls}: F1={metrics['f1-score']:.4f}, "
                      f"Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}")
        
        print("\n" + "=" * 60)
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save evaluation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics (remove non-serializable items)
        save_results = {
            'accuracy': results['accuracy'],
            'top3_accuracy': results['top3_accuracy'],
            'top5_accuracy': results['top5_accuracy'],
            'latency': results['latency'],
            'confidence_stats': results['confidence_stats'],
            'error_analysis': results['error_analysis'],
            'classification_report': results['per_class_metrics']['classification_report']
        }
        
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Save confusion matrix
        np.save(output_dir / 'confusion_matrix.npy', results['confusion_matrix'])
        
        print(f"\nResults saved to: {output_dir}")
    
    def _plot_results(self, 
                     results: Dict,
                     output_dir: str,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_pred_probs: np.ndarray):
        """Plot evaluation results"""
        output_dir = Path(output_dir)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class metrics
        report = results['per_class_metrics']['classification_report']
        
        # Only include classes that are in the report
        metrics_df = pd.DataFrame([
            {
                'Class': cls,
                'Precision': report.get(cls, {}).get('precision', 0),
                'Recall': report.get(cls, {}).get('recall', 0),
                'F1-Score': report.get(cls, {}).get('f1-score', 0)
            }
            for cls in self.class_names if cls in report
        ])
        
        if len(metrics_df) == 0:
            print("Warning: No per-class metrics to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision')
        ax.bar(x, metrics_df['Recall'], width, label='Recall')
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Class'].tolist(), rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confidence distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        max_probs = np.max(y_pred_probs, axis=1)
        correct_mask = (y_pred == y_true)
        
        axes[0].hist(max_probs[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
        axes[0].hist(max_probs[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prediction Confidence Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confidence vs accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracy = []
        bin_centers = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
            if mask.sum() > 0:
                bin_accuracy.append((y_pred[mask] == y_true[mask]).mean())
                bin_centers.append((confidence_bins[i] + confidence_bins[i+1]) / 2)
        
        axes[1].plot(bin_centers, bin_accuracy, marker='o', linewidth=2, markersize=8)
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Calibration Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    # Example evaluation
    from tensorflow.keras.models import load_model
    from model_architecture import TCNBlock, TransformerBlock, PositionalEncoding
    
    model_path = "models/saved_models/tcn_transformer_20251203_095134/best_model.h5"
    data_path = "data/extracted_landmarks/dataset.pkl"
    
    if Path(model_path).exists() and Path(data_path).exists():
        # Load model with custom objects
        custom_objects = {
            'TCNBlock': TCNBlock,
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Load test data
        import pickle
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Use 20% as test set
        from sklearn.model_selection import train_test_split
        
        X = dataset['sequences']
        y = tf.keras.utils.to_categorical(dataset['labels'], len(dataset['class_names']))
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Evaluate
        evaluator = ModelEvaluator(model, dataset['class_names'])
        results = evaluator.evaluate(
            X_test, y_test,
            output_dir='models/saved_models/evaluation'
        )
    else:
        print("Model or data not found. Please train model first.")
