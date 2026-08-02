"""
Data Augmentation Pipeline for PSL Recognition
Expands limited dataset through temporal and spatial transformations
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
import random
from scipy.interpolate import interp1d


class TemporalAugmentor:
    """Temporal augmentation techniques for video sequences"""
    
    def __init__(self):
        self.augmentation_methods = [
            'time_warp',
            'speed_variation',
            'frame_dropping',
            'temporal_jitter'
        ]
    
    def time_warp(self, sequence: np.ndarray, warp_factor: float = 0.10) -> np.ndarray:
        """
        Apply time warping to sequence (REDUCED intensity to prevent overfitting)
        
        Args:
            sequence: Input sequence of shape (T, F) where T is time, F is features
            warp_factor: Maximum warping factor (±10%, REDUCED from 15%)
        
        Returns:
            Warped sequence with same shape
        """
        seq_len = len(sequence)
        warp_amount = random.uniform(-warp_factor, warp_factor)
        new_len = int(seq_len * (1 + warp_amount))
        
        # Interpolate to new length then sample back to original length
        old_indices = np.linspace(0, seq_len - 1, seq_len)
        new_indices = np.linspace(0, seq_len - 1, new_len)
        
        warped = np.zeros((new_len, sequence.shape[1]))
        for feat_idx in range(sequence.shape[1]):
            interpolator = interp1d(old_indices, sequence[:, feat_idx], 
                                   kind='cubic', fill_value='extrapolate')
            warped[:, feat_idx] = interpolator(new_indices)
        
        # Resample to original length
        final_indices = np.linspace(0, new_len - 1, seq_len).astype(int)
        return warped[final_indices]
    
    def speed_variation(self, sequence: np.ndarray, 
                       speed_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Simulate different signing speeds (REDUCED range for realism)
        
        Args:
            sequence: Input sequence
            speed_range: Min and max speed multipliers (REDUCED: 0.8-1.2 → 0.9-1.1)
        
        Returns:
            Speed-varied sequence
        """
        speed = random.uniform(*speed_range)
        new_len = int(len(sequence) / speed)
        
        old_indices = np.linspace(0, len(sequence) - 1, len(sequence))
        new_indices = np.linspace(0, len(sequence) - 1, new_len)
        
        varied = np.zeros((new_len, sequence.shape[1]))
        for feat_idx in range(sequence.shape[1]):
            interpolator = interp1d(old_indices, sequence[:, feat_idx], 
                                   kind='linear', fill_value='extrapolate')
            varied[:, feat_idx] = interpolator(new_indices)
        
        return varied
    
    def frame_dropping(self, sequence: np.ndarray, 
                      drop_prob: float = 0.1) -> np.ndarray:
        """
        Randomly drop frames to simulate low FPS
        
        Args:
            sequence: Input sequence
            drop_prob: Probability of dropping each frame
        
        Returns:
            Sequence with dropped frames
        """
        keep_mask = np.random.random(len(sequence)) > drop_prob
        # Ensure at least half the frames remain
        if keep_mask.sum() < len(sequence) // 2:
            return sequence
        
        return sequence[keep_mask]
    
    def temporal_jitter(self, sequence: np.ndarray, 
                       jitter_std: float = 0.01) -> np.ndarray:
        """
        Add temporal jitter to features
        
        Args:
            sequence: Input sequence
            jitter_std: Standard deviation of jitter noise
        
        Returns:
            Jittered sequence
        """
        noise = np.random.normal(0, jitter_std, sequence.shape)
        return sequence + noise
    
    def augment(self, sequence: np.ndarray, 
                methods: List[str] = None) -> np.ndarray:
        """
        Apply random temporal augmentations
        
        Args:
            sequence: Input sequence
            methods: List of methods to apply (random if None)
        
        Returns:
            Augmented sequence
        """
        if methods is None:
            # Randomly select 1-2 augmentation methods
            num_methods = random.randint(1, 2)
            methods = random.sample(self.augmentation_methods, num_methods)
        
        augmented = sequence.copy()
        
        for method in methods:
            if method == 'time_warp':
                augmented = self.time_warp(augmented)
            elif method == 'speed_variation':
                augmented = self.speed_variation(augmented)
            elif method == 'frame_dropping':
                augmented = self.frame_dropping(augmented)
            elif method == 'temporal_jitter':
                augmented = self.temporal_jitter(augmented)
        
        return augmented


class SpatialAugmentor:
    """Spatial augmentation for hand landmarks"""
    
    def __init__(self, num_landmarks: int = 21):
        self.num_landmarks = num_landmarks
        self.feature_dim = num_landmarks * 3  # x, y, z coordinates
    
    def hand_rotation(self, landmarks: np.ndarray, 
                     angle_range: Tuple[float, float] = (-10, 10)) -> np.ndarray:
        """
        Rotate hand landmarks around center (REDUCED for realism)
        
        Args:
            landmarks: Flattened landmarks (63,) for single hand
            angle_range: Rotation angle range in degrees (REDUCED: ±20° → ±10°)
        
        Returns:
            Rotated landmarks
        """
        angle = np.radians(random.uniform(*angle_range))
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Reshape to (21, 3)
        landmarks_3d = landmarks.reshape(-1, 3)
        
        # Calculate center
        center = landmarks_3d.mean(axis=0)
        
        # Center, rotate, and uncenter
        centered = landmarks_3d - center
        rotated = centered @ rotation_matrix.T
        final = rotated + center
        
        return final.flatten()
    
    def scale_variation(self, landmarks: np.ndarray,
                       scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Apply random scaling to hand (REDUCED for realism)
        
        Args:
            landmarks: Flattened landmarks
            scale_range: Min and max scale factors (REDUCED: 0.85-1.15 → 0.9-1.1)
        
        Returns:
            Scaled landmarks
        """
        scale = random.uniform(*scale_range)
        landmarks_3d = landmarks.reshape(-1, 3)
        center = landmarks_3d.mean(axis=0)
        
        scaled = (landmarks_3d - center) * scale + center
        return scaled.flatten()
    
    def translation_shift(self, landmarks: np.ndarray,
                         shift_range: float = 0.05) -> np.ndarray:
        """
        Apply random translation
        
        Args:
            landmarks: Flattened landmarks
            shift_range: Maximum shift as fraction of frame
        
        Returns:
            Translated landmarks
        """
        landmarks_3d = landmarks.reshape(-1, 3)
        
        shift_x = random.uniform(-shift_range, shift_range)
        shift_y = random.uniform(-shift_range, shift_range)
        shift = np.array([shift_x, shift_y, 0])
        
        translated = landmarks_3d + shift
        return translated.flatten()
    
    def gaussian_noise(self, landmarks: np.ndarray,
                      noise_std: float = 0.005) -> np.ndarray:
        """
        Add Gaussian noise to landmarks
        
        Args:
            landmarks: Flattened landmarks
            noise_std: Standard deviation of noise
        
        Returns:
            Noisy landmarks
        """
        noise = np.random.normal(0, noise_std, landmarks.shape)
        return landmarks + noise
    
    def perspective_transform(self, landmarks: np.ndarray,
                            transform_strength: float = 0.1) -> np.ndarray:
        """
        Apply simple perspective-like transformation
        
        Args:
            landmarks: Flattened landmarks
            transform_strength: Strength of perspective effect
        
        Returns:
            Transformed landmarks
        """
        landmarks_3d = landmarks.reshape(-1, 3)
        
        # Apply depth-based scaling (simulate perspective)
        depth_factor = 1 + landmarks_3d[:, 2] * transform_strength
        landmarks_3d[:, :2] *= depth_factor[:, np.newaxis]
        
        return landmarks_3d.flatten()
    
    def augment_frame(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply random spatial augmentations to a single frame
        
        Args:
            landmarks: Flattened landmarks for one or two hands
        
        Returns:
            Augmented landmarks
        """
        augmented = landmarks.copy()
        
        # Randomly apply 2-3 augmentations
        augmentations = [
            (self.hand_rotation, 0.6),
            (self.scale_variation, 0.5),
            (self.translation_shift, 0.4),
            (self.gaussian_noise, 0.7),
            (self.perspective_transform, 0.3)
        ]
        
        for aug_func, prob in augmentations:
            if random.random() < prob:
                # Handle single or double hand
                if len(augmented) == 63:  # Single hand
                    augmented = aug_func(augmented)
                else:  # Double hand or extended features
                    # Apply to first 63 features (first hand)
                    augmented[:63] = aug_func(augmented[:63])
                    if len(augmented) >= 126:  # Second hand exists
                        augmented[63:126] = aug_func(augmented[63:126])
        
        return augmented


class MixupAugmentor:
    """Mixup and CutMix augmentation for sequences"""
    
    def mixup(self, seq1: np.ndarray, seq2: np.ndarray,
             label1: np.ndarray, label2: np.ndarray,
             alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Mixup augmentation
        
        Args:
            seq1, seq2: Two sequences of same shape
            label1, label2: Corresponding one-hot labels
            alpha: Beta distribution parameter
        
        Returns:
            Mixed sequence and label
        """
        lam = np.random.beta(alpha, alpha)
        
        mixed_seq = lam * seq1 + (1 - lam) * seq2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_seq, mixed_label
    
    def cutmix_temporal(self, seq1: np.ndarray, seq2: np.ndarray,
                       label1: np.ndarray, label2: np.ndarray,
                       alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal CutMix
        
        Args:
            seq1, seq2: Two sequences
            label1, label2: Corresponding labels
            alpha: Beta distribution parameter
        
        Returns:
            Mixed sequence and label
        """
        lam = np.random.beta(alpha, alpha)
        
        seq_len = len(seq1)
        cut_len = int(seq_len * (1 - lam))
        cut_start = random.randint(0, seq_len - cut_len)
        cut_end = cut_start + cut_len
        
        mixed_seq = seq1.copy()
        mixed_seq[cut_start:cut_end] = seq2[cut_start:cut_end]
        
        # Adjust labels based on mixing ratio
        actual_lam = 1 - (cut_len / seq_len)
        mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
        
        return mixed_seq, mixed_label


class AugmentationPipeline:
    """Complete augmentation pipeline"""
    
    def __init__(self, expansion_factor: int = 10):
        """
        Args:
            expansion_factor: How many augmented samples to create per original
        """
        self.expansion_factor = expansion_factor
        self.temporal_aug = TemporalAugmentor()
        self.spatial_aug = SpatialAugmentor()
        self.mixup_aug = MixupAugmentor()
    
    def augment_sequence(self, sequence: np.ndarray, 
                        use_mixup: bool = False,
                        mixup_sequence: np.ndarray = None) -> np.ndarray:
        """
        Apply full augmentation pipeline to a sequence
        
        Args:
            sequence: Input sequence of shape (T, F)
            use_mixup: Whether to apply mixup
            mixup_sequence: Second sequence for mixup
        
        Returns:
            Augmented sequence
        """
        # Temporal augmentation
        augmented = self.temporal_aug.augment(sequence)
        
        # Spatial augmentation on each frame
        for i in range(len(augmented)):
            augmented[i] = self.spatial_aug.augment_frame(augmented[i])
        
        return augmented
    
    def expand_dataset(self, sequences: List[np.ndarray], 
                      labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expand entire dataset through augmentation
        
        Args:
            sequences: List of original sequences
            labels: List of corresponding one-hot labels
        
        Returns:
            Expanded sequences and labels
        """
        augmented_sequences = []
        augmented_labels = []
        
        # Keep originals
        augmented_sequences.extend(sequences)
        augmented_labels.extend(labels)
        
        # Generate augmented versions
        for idx, (seq, label) in enumerate(zip(sequences, labels)):
            for _ in range(self.expansion_factor - 1):
                # Decide whether to use mixup (20% chance)
                if random.random() < 0.2 and len(sequences) > 1:
                    # Select random sequence with same label for mixup
                    same_class_indices = [i for i, l in enumerate(labels) 
                                         if np.array_equal(l, label)]
                    if len(same_class_indices) > 1:
                        mix_idx = random.choice([i for i in same_class_indices 
                                               if i != idx])
                        aug_seq, aug_label = self.mixup_aug.mixup(
                            seq, sequences[mix_idx], label, labels[mix_idx]
                        )
                    else:
                        aug_seq = self.augment_sequence(seq)
                        aug_label = label
                else:
                    aug_seq = self.augment_sequence(seq)
                    aug_label = label
                
                augmented_sequences.append(aug_seq)
                augmented_labels.append(aug_label)
        
        return augmented_sequences, augmented_labels


if __name__ == "__main__":
    # Example usage
    print("Data Augmentation Pipeline initialized")
    print("=" * 60)
    
    # Simulate a sequence: 30 frames, 95 features per frame
    dummy_sequence = np.random.randn(30, 95)
    dummy_label = np.zeros(32)
    dummy_label[0] = 1  # One-hot encoded
    
    pipeline = AugmentationPipeline(expansion_factor=10)
    
    # Test augmentation
    augmented = pipeline.augment_sequence(dummy_sequence)
    print(f"Original shape: {dummy_sequence.shape}")
    print(f"Augmented shape: {augmented.shape}")
    
    # Test dataset expansion
    sequences = [dummy_sequence] * 5
    labels = [dummy_label] * 5
    
    expanded_seqs, expanded_labels = pipeline.expand_dataset(sequences, labels)
    print(f"\nDataset expansion:")
    print(f"Original: {len(sequences)} sequences")
    print(f"Expanded: {len(expanded_seqs)} sequences")
    print(f"Expansion factor: {len(expanded_seqs) / len(sequences):.1f}x")
