"""
Data Loader for PSL Recognition Training
Handles data loading, preprocessing, and batching
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List, Dict
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from augmentation import AugmentationPipeline


class PSLDataLoader:
    """Data loader for PSL recognition"""
    
    def __init__(self, 
                 data_path: str,
                 batch_size: int = 16,
                 validation_split: float = 0.15,
                 test_split: float = 0.25,
                 augmentation_factor: int = 5,
                 use_augmentation: bool = True):
        """
        Initialize data loader
        
        Args:
            data_path: Path to dataset pickle file
            batch_size: Batch size for training
            validation_split: Fraction of data for validation (from remaining after test)
            test_split: Fraction of data for test set
            augmentation_factor: How many augmented samples per original (REDUCED to prevent overfitting)
            use_augmentation: Whether to apply data augmentation (ONLY on training set)
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.augmentation_factor = augmentation_factor
        self.use_augmentation = use_augmentation
        
        # Load data
        self.dataset = self._load_dataset()
        self.class_names = self.dataset['class_names']
        self.num_classes = len(self.class_names)
        
        # Initialize augmentation pipeline
        if self.use_augmentation:
            self.augmentor = AugmentationPipeline(expansion_factor=augmentation_factor)
    
    def _load_dataset(self) -> Dict:
        """Load dataset from pickle file"""
        print(f"Loading dataset from {self.data_path}...")
        
        with open(self.data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Loaded {len(dataset['sequences'])} sequences")
        print(f"Number of classes: {len(dataset['class_names'])}")
        print(f"Sequence length: {dataset['sequence_length']}")
        print(f"Feature dimension: {dataset['feature_dim']}")
        
        return dataset
    
    def prepare_data(self, 
                    normalize: bool = True,
                    apply_augmentation: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for training with PROPER 3-way split to prevent data leakage
        
        CRITICAL FIX: Split FIRST into train/val/test, then augment ONLY training set
        This prevents test samples from being augmented versions of training samples
        
        Args:
            normalize: Whether to normalize features
            apply_augmentation: Whether to apply data augmentation (ONLY to training set)
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        sequences = self.dataset['sequences']
        labels = self.dataset['labels']
        
        print(f"\n{'='*60}")
        print("DATA SPLITTING (FIXED - NO DATA LEAKAGE)")
        print(f"{'='*60}")
        print(f"Total samples: {len(sequences)}")
        print(f"Total classes: {self.num_classes}")
        
        # Check if stratified split is possible (need at least 3 samples per class for 3-way split)
        unique, counts = np.unique(labels, return_counts=True)
        min_samples = counts.min()
        use_stratify = min_samples >= 3
        
        if not use_stratify:
            print(f"\nWarning: Some classes have <3 samples. Disabling stratified split.")
            print(f"Class distribution: {dict(zip(unique, counts))}")
        
        # STEP 1: Split into train+val and test (NO AUGMENTATION YET)
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels,
            test_size=self.test_split,
            stratify=labels if use_stratify else None,
            random_state=42
        )
        
        # STEP 2: Split train+val into train and val (STILL NO AUGMENTATION)
        val_size_adjusted = self.validation_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp if use_stratify else None,
            random_state=42
        )
        
        print(f"\nRaw data split (BEFORE augmentation):")
        print(f"  Training: {len(X_train)} samples ({len(X_train)/len(sequences)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(sequences)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(sequences)*100:.1f}%)")
        
        # Convert labels to one-hot AFTER splitting
        y_train_onehot = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, self.num_classes)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # STEP 3: Apply augmentation ONLY to training set (CRITICAL FIX)
        if apply_augmentation and self.use_augmentation:
            print(f"\n{'='*60}")
            print(f"AUGMENTATION (TRAINING SET ONLY - factor={self.augmentation_factor})")
            print(f"{'='*60}")
            
            X_train_list = list(X_train)
            y_train_list = list(y_train_onehot)
            
            X_train_aug, y_train_aug = self.augmentor.expand_dataset(
                X_train_list, y_train_list
            )
            
            # Pad sequences to same length
            max_len = self.dataset['sequence_length']
            X_train = self._pad_sequences(X_train_aug, max_len)
            y_train = np.array(y_train_aug)
            
            print(f"  Training samples AFTER augmentation: {len(X_train)}")
            print(f"  Validation samples (NO augmentation): {len(X_val)}")
            print(f"  Test samples (NO augmentation): {len(X_test)}")
        else:
            y_train = y_train_onehot
        
        # Assign val and test labels (always one-hot encoded)
        y_val = y_val_onehot
        y_test = y_test_onehot
        
        # STEP 4: Normalize (fit on training, apply to val and test)
        if normalize:
            print(f"\nNormalizing data (fit on training, apply to val/test)...")
            X_train, norm_params = self._normalize(X_train)
            X_val = self._apply_normalization(X_val, norm_params)
            X_test = self._apply_normalization(X_test, norm_params)
            
            # Save normalization parameters
            self.norm_params = norm_params
        
        print(f"\n{'='*60}")
        print("DATA PREPARATION COMPLETE")
        print(f"{'='*60}\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _pad_sequences(self, sequences: List[np.ndarray], 
                      max_len: int) -> np.ndarray:
        """Pad sequences to same length"""
        feature_dim = sequences[0].shape[1]
        padded = np.zeros((len(sequences), max_len, feature_dim))
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            padded[i, :seq_len] = seq[:seq_len]
        
        return padded
    
    def _normalize(self, sequences: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize sequences to zero mean and unit variance
        
        Args:
            sequences: Input sequences
        
        Returns:
            Normalized sequences and normalization parameters
        """
        # Reshape to (num_samples * seq_len, feature_dim)
        original_shape = sequences.shape
        flattened = sequences.reshape(-1, sequences.shape[-1])
        
        # Compute statistics
        mean = flattened.mean(axis=0)
        std = flattened.std(axis=0) + 1e-8
        
        # Normalize
        normalized = (flattened - mean) / std
        normalized = normalized.reshape(original_shape)
        
        params = {'mean': mean, 'std': std}
        
        return normalized, params
    
    def _apply_normalization(self, sequences: np.ndarray, 
                           params: Dict) -> np.ndarray:
        """Apply saved normalization parameters"""
        original_shape = sequences.shape
        flattened = sequences.reshape(-1, sequences.shape[-1])
        
        normalized = (flattened - params['mean']) / params['std']
        
        return normalized.reshape(original_shape)
    
    def create_tf_dataset(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         shuffle: bool = True,
                         augment: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset
        
        Args:
            X: Input sequences
            y: Labels
            shuffle: Whether to shuffle data
            augment: Whether to apply online augmentation
        
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            labels: Array of integer labels
        
        Returns:
            Dictionary mapping class index to weight
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get integer labels
        if len(labels.shape) > 1:
            labels = np.argmax(labels, axis=1)
        
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        
        class_weights = {i: w for i, w in enumerate(weights)}
        
        print("\nClass weights:")
        for cls_idx, weight in class_weights.items():
            cls_name = self.class_names[cls_idx]
            print(f"  {cls_name}: {weight:.3f}")
        
        return class_weights
    
    def get_kfold_splits(self, n_splits: int = 5) -> List[Tuple]:
        """
        Generate K-fold cross-validation splits
        
        Args:
            n_splits: Number of folds
        
        Returns:
            List of (train_idx, val_idx) tuples
        """
        sequences = self.dataset['sequences']
        labels = self.dataset['labels']
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        splits = []
        for train_idx, val_idx in skf.split(sequences, labels):
            splits.append((train_idx, val_idx))
        
        return splits
    
    def get_data_info(self) -> Dict:
        """Get information about the dataset"""
        sequences = self.dataset['sequences']
        labels = self.dataset['labels']
        
        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = {
            self.class_names[cls]: count 
            for cls, count in zip(unique, counts)
        }
        
        info = {
            'num_samples': len(sequences),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'sequence_length': self.dataset['sequence_length'],
            'feature_dim': self.dataset['feature_dim'],
            'class_distribution': class_distribution,
            'min_samples_per_class': counts.min(),
            'max_samples_per_class': counts.max(),
            'mean_samples_per_class': counts.mean(),
        }
        
        return info
    
    def print_data_summary(self):
        """Print summary of dataset"""
        info = self.get_data_info()
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total samples: {info['num_samples']}")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Sequence length: {info['sequence_length']}")
        print(f"Feature dimension: {info['feature_dim']}")
        print(f"\nSamples per class:")
        print(f"  Min: {info['min_samples_per_class']}")
        print(f"  Max: {info['max_samples_per_class']}")
        print(f"  Mean: {info['mean_samples_per_class']:.1f}")
        print(f"\nClass distribution:")
        for cls_name, count in sorted(info['class_distribution'].items()):
            print(f"  {cls_name}: {count}")
        print("=" * 60)


class MixupGenerator:
    """Generator for online Mixup augmentation"""
    
    def __init__(self, X, y, batch_size=16, alpha=0.2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.num_samples = len(X)
    
    def __call__(self):
        """Generate batches with Mixup"""
        indices = np.arange(self.num_samples)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, self.num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Sample lambda from Beta distribution
                lam = np.random.beta(self.alpha, self.alpha, len(batch_indices))
                
                # Sample random indices for mixing
                mix_indices = np.random.permutation(self.num_samples)[:len(batch_indices)]
                
                # Apply Mixup
                X_batch = np.zeros((len(batch_indices), *self.X.shape[1:]))
                y_batch = np.zeros((len(batch_indices), self.y.shape[1]))
                
                for i, (idx1, idx2, l) in enumerate(zip(batch_indices, mix_indices, lam)):
                    X_batch[i] = l * self.X[idx1] + (1 - l) * self.X[idx2]
                    y_batch[i] = l * self.y[idx1] + (1 - l) * self.y[idx2]
                
                yield X_batch, y_batch


if __name__ == "__main__":
    # Example usage
    data_path = "data/extracted_landmarks/dataset.pkl"
    
    if Path(data_path).exists():
        # Initialize data loader
        loader = PSLDataLoader(
            data_path=data_path,
            batch_size=16,
            augmentation_factor=10
        )
        
        # Print data summary
        loader.print_data_summary()
        
        # Prepare data
        X_train, X_val, y_train, y_val = loader.prepare_data()
        
        print(f"\nFinal data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  y_val: {y_val.shape}")
        
        # Create TensorFlow datasets
        train_dataset = loader.create_tf_dataset(X_train, y_train, shuffle=True)
        val_dataset = loader.create_tf_dataset(X_val, y_val, shuffle=False)
        
        print(f"\nTensorFlow datasets created successfully")
    else:
        print(f"Dataset not found at {data_path}")
        print("Please run extract_landmarks.py first to generate the dataset")
