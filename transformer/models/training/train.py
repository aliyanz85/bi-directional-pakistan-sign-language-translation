"""
Training Pipeline for PSL Recognition
Includes callbacks, training loops, and model management
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from model_architecture import create_model, FocalLoss
from data_loader import PSLDataLoader


class MetricsLogger(Callback):
    """Custom callback for logging additional metrics"""
    
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/100 - Training...")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Calculate epoch duration
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Add timestamp
        logs['timestamp'] = datetime.now().isoformat()
        logs['epoch'] = epoch
        logs['duration'] = epoch_duration
        
        self.metrics_history.append(logs)
        
        # Save to JSON after each epoch
        with open(self.log_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Print detailed progress
        print(f"{'='*60}")
        print(f"Epoch {epoch + 1} Complete (took {epoch_duration:.2f}s)")
        print(f"  Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
        print(f"  Acc: {logs.get('accuracy', 0):.4f} | Val Acc: {logs.get('val_accuracy', 0):.4f}")
        print(f"  Top-3 Acc: {logs.get('top3_acc', 0):.4f} | Top-5 Acc: {logs.get('top5_acc', 0):.4f}")
        print(f"{'='*60}")


class TrainingPipeline:
    """Complete training pipeline for PSL recognition"""
    
    def __init__(self,
                 model_type: str = 'tcn_transformer',
                 num_classes: int = 32,
                 max_sequence_length: int = 60,
                 feature_dim: int = 188,
                 output_dir: str = 'models/saved_models'):
        """
        Initialize training pipeline
        
        Args:
            model_type: Type of model to train
            num_classes: Number of classes
            max_sequence_length: Sequence length
            feature_dim: Feature dimension
            output_dir: Directory to save models and logs
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.max_seq_len = max_sequence_length
        self.feature_dim = feature_dim
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        self.model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            max_sequence_length=max_sequence_length,
            feature_dim=feature_dim
        )
        
        self.history = None
        self.training_time = 0
    
    def compile_model(self,
                     loss='categorical_crossentropy',
                     optimizer='adamw',
                     learning_rate: float = 1e-3,
                     label_smoothing: float = 0.1,
                     use_focal_loss: bool = False):
        """
        Compile model with optimizer and loss
        
        Args:
            loss: Loss function
            optimizer: Optimizer name
            learning_rate: Initial learning rate
            label_smoothing: Label smoothing factor
            use_focal_loss: Whether to use focal loss
        """
        # Loss function
        if use_focal_loss:
            loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        elif label_smoothing > 0:
            loss_fn = keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            )
        else:
            loss_fn = loss
        
        # Optimizer
        if optimizer.lower() == 'adamw':
            opt = keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-4
            )
        elif optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Compile
        self.model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc'),
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')
            ]
        )
        
        print(f"\nModel compiled:")
        print(f"  Loss: {loss_fn if isinstance(loss_fn, str) else loss_fn.__class__.__name__}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Learning rate: {learning_rate}")
    
    def get_callbacks(self,
                     monitor: str = 'val_loss',
                     patience: int = 10,
                     min_delta: float = 0.001) -> List[Callback]:
        """
        Get training callbacks with AGGRESSIVE EARLY STOPPING
        
        Args:
            monitor: Metric to monitor
            patience: Patience for early stopping (REDUCED: 15->10)
            min_delta: Minimum change to qualify as improvement (INCREASED for stricter stopping)
        
        Returns:
            List of callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"{self.model_type}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Model checkpointing
            ModelCheckpoint(
                filepath=str(run_dir / 'best_model.h5'),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1
            ),
            
            # Early stopping - MORE AGGRESSIVE
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            
            # CSV logging
            CSVLogger(
                filename=str(run_dir / 'training_log.csv'),
                append=True
            ),
            
            # Custom metrics logger
            MetricsLogger(log_dir=str(run_dir))
        ]
        
        self.run_dir = run_dir
        
        return callbacks
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 16,
             class_weights: Dict = None,
             verbose: int = 0):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            class_weights: Class weights for imbalanced data
            verbose: Verbosity level (0 = silent, custom callback handles logging)
        
        Returns:
            Training history
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Model: {self.model_type}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("=" * 60 + "\n")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Start training
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total training time: {self.training_time / 60:.2f} minutes")
        print(f"Best val_loss: {min(self.history.history['val_loss']):.4f}")
        print(f"Best val_accuracy: {max(self.history.history['val_accuracy']):.4f}")
        print(f"Model saved to: {self.run_dir / 'best_model.h5'}")
        print("=" * 60 + "\n")
        
        # Save training summary
        self._save_training_summary()
        
        # Plot training history
        self._plot_training_history()
        
        return self.history
    
    def _save_training_summary(self):
        """Save training summary to JSON"""
        summary = {
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'sequence_length': self.max_seq_len,
            'feature_dim': self.feature_dim,
            'training_time_seconds': self.training_time,
            'total_epochs': len(self.history.history['loss']),
            'final_metrics': {
                'train_loss': float(self.history.history['loss'][-1]),
                'train_accuracy': float(self.history.history['accuracy'][-1]),
                'val_loss': float(self.history.history['val_loss'][-1]),
                'val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'best_val_loss': float(min(self.history.history['val_loss'])),
                'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
            },
            'model_parameters': self.model.count_params()
        }
        
        with open(self.run_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 Accuracy
        axes[1, 0].plot(self.history.history['top3_acc'], label='Train')
        axes[1, 0].plot(self.history.history['val_top3_acc'], label='Validation')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {self.run_dir / 'training_history.png'}")
    
    def cross_validate(self,
                      data_loader: PSLDataLoader,
                      n_splits: int = 5,
                      epochs: int = 50,
                      batch_size: int = 16) -> List[Dict]:
        """
        Perform k-fold cross-validation
        
        Args:
            data_loader: PSL data loader
            n_splits: Number of folds
            epochs: Epochs per fold
            batch_size: Batch size
        
        Returns:
            List of results for each fold
        """
        print("\n" + "=" * 60)
        print(f"STARTING {n_splits}-FOLD CROSS-VALIDATION")
        print("=" * 60 + "\n")
        
        splits = data_loader.get_kfold_splits(n_splits)
        fold_results = []
        
        X_all = data_loader.dataset['sequences']
        y_all = tf.keras.utils.to_categorical(
            data_loader.dataset['labels'],
            data_loader.num_classes
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n{'=' * 60}")
            print(f"FOLD {fold_idx + 1}/{n_splits}")
            print(f"{'=' * 60}\n")
            
            # Split data
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            
            # Reset model
            self.model = create_model(
                model_type=self.model_type,
                num_classes=self.num_classes,
                max_sequence_length=self.max_seq_len,
                feature_dim=self.feature_dim
            )
            self.compile_model()
            
            # Train
            history = self.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Store results
            fold_results.append({
                'fold': fold_idx + 1,
                'val_loss': min(history.history['val_loss']),
                'val_accuracy': max(history.history['val_accuracy']),
                'val_top3_acc': max(history.history['val_top3_acc'])
            })
        
        # Print summary
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        for result in fold_results:
            print(f"Fold {result['fold']}: "
                  f"Loss={result['val_loss']:.4f}, "
                  f"Acc={result['val_accuracy']:.4f}, "
                  f"Top3={result['val_top3_acc']:.4f}")
        
        avg_loss = np.mean([r['val_loss'] for r in fold_results])
        avg_acc = np.mean([r['val_accuracy'] for r in fold_results])
        avg_top3 = np.mean([r['val_top3_acc'] for r in fold_results])
        
        print(f"\nAverage: Loss={avg_loss:.4f}, "
              f"Acc={avg_acc:.4f}, Top3={avg_top3:.4f}")
        print("=" * 60 + "\n")
        
        return fold_results


if __name__ == "__main__":
    # Example training script
    print("PSL Recognition Training Pipeline")
    print("=" * 60)
    
    # This would be run after extracting landmarks
    # python train.py --data_path data/extracted_landmarks/dataset.pkl
    
    # Initialize data loader with FIXED SPLITS
    data_path = "data/extracted_landmarks/dataset.pkl"
    
    if Path(data_path).exists():
        loader = PSLDataLoader(
            data_path=data_path,
            batch_size=16,
            validation_split=0.15,  # 15% validation
            test_split=0.25,  # 25% test (larger for reliable evaluation)
            augmentation_factor=5  # Reduced from 10 to prevent overfitting
        )
        
        # Prepare data with proper 3-way split
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data()
        
        # Get class weights
        class_weights = loader.get_class_weights(y_train)
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(
            model_type='tcn_transformer',
            num_classes=loader.num_classes,
            feature_dim=loader.dataset['feature_dim']
        )
        
        # Compile model
        pipeline.compile_model(
            learning_rate=1e-3,
            label_smoothing=0.1
        )
        
        # Train
        history = pipeline.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=16,
            class_weights=class_weights
        )
        
        # CRITICAL: Evaluate on held-out test set
        print(f"\\n{'='*60}")
        print("EVALUATING ON HELD-OUT TEST SET")
        print(f"{'='*60}")
        
        # Load best model
        best_model_path = pipeline.run_dir / 'best_model.h5'
        from model_architecture import TCNBlock, TransformerBlock, PositionalEncoding
        
        custom_objects = {
            'TCNBlock': TCNBlock,
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }
        
        best_model = keras.models.load_model(best_model_path, custom_objects=custom_objects)
        
        # Evaluate on test set
        test_results = best_model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\\nTest Set Results (UNSEEN DATA):")
        print(f"  Test Loss: {test_results[0]:.4f}")
        print(f"  Test Accuracy: {test_results[1]:.4f}")
        print(f"  Test Top-3 Accuracy: {test_results[2]:.4f}")
        print(f"  Test Top-5 Accuracy: {test_results[3]:.4f}")
        
        # Save test results
        test_results_dict = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_top3_accuracy': float(test_results[2]),
            'test_top5_accuracy': float(test_results[3]),
            'test_samples': len(X_test),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
        
        with open(pipeline.run_dir / 'test_results.json', 'w') as f:
            json.dump(test_results_dict, f, indent=2)
        
        print(f"\\nTest results saved to: {pipeline.run_dir / 'test_results.json'}")
        
        # Run comprehensive evaluation
        from evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator(best_model, loader.class_names)
        eval_results = evaluator.evaluate(
            X_test, y_test,
            output_dir=str(Path('models/saved_models') / 'evaluation')
        )
        
        print(f"\\n{'='*60}")
        print("TRAINING COMPLETE - CHECK FOR OVERFITTING")
        print(f"{'='*60}")
        print(f"Training accuracy: {max(history.history['accuracy']):.4f}")
        print(f"Validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"Test accuracy: {test_results[1]:.4f}")
        print(f"\\nGap (Train - Test): {(max(history.history['accuracy']) - test_results[1])*100:.1f}%")
        print(f"\\nTarget: Gap should be <15% for acceptable generalization")
        print(f"{'='*60}\\n")
        
    else:
        print(f"Dataset not found at {data_path}")
        print("Please extract landmarks first using extract_landmarks.py")
