"""
Data augmentation utilities (from enhanced training script).
Provides `augment_sequence` and `augment_dataset` to expand small datasets.
"""
import numpy as np


def augment_sequence(sequence, augmentation_type='random'):
    augmented = sequence.copy()
    if augmentation_type == 'noise' or augmentation_type == 'random':
        noise = np.random.normal(0, 0.02, sequence.shape)
        augmented = sequence + noise
    elif augmentation_type == 'scale' or augmentation_type == 'random':
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented = sequence * scale_factor
    elif augmentation_type == 'time_warp':
        original_length = len(sequence)
        warp_factor = np.random.uniform(0.9, 1.1)
        new_length = int(original_length * warp_factor)
        indices = np.linspace(0, original_length - 1, new_length)
        warped = np.array([sequence[int(i)] for i in indices])
        indices_back = np.linspace(0, len(warped) - 1, original_length)
        augmented = np.array([warped[int(i)] for i in indices_back])
    elif augmentation_type == 'rotation':
        angle = np.random.uniform(-0.1, 0.1)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        augmented = sequence.copy()
        for frame_idx in range(len(augmented)):
            for i in range(0, 132, 4):
                x, y = augmented[frame_idx, i], augmented[frame_idx, i+1]
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
            for i in range(132, 132+1404, 3):
                x, y = augmented[frame_idx, i], augmented[frame_idx, i+1]
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
            for i in range(132+1404, len(augmented[frame_idx]), 3):
                x, y = augmented[frame_idx, i], augmented[frame_idx, i+1]
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
    elif augmentation_type == 'dropout':
        dropout_mask = np.random.random(sequence.shape) > 0.1
        augmented = sequence * dropout_mask
    elif augmentation_type == 'shift':
        shift = np.random.randint(-3, 4)
        if shift > 0:
            augmented = np.concatenate([sequence[shift:], sequence[-shift:]])
        elif shift < 0:
            augmented = np.concatenate([sequence[:shift], sequence[:-shift]])
    return augmented


def augment_dataset(X, y, factor=10):
    augmented_X = []
    augmented_y = []
    strategies = ['noise', 'scale', 'time_warp', 'rotation', 'dropout', 'shift']
    for sequence, label in zip(X, y):
        augmented_X.append(sequence)
        augmented_y.append(label)
        for _ in range(factor - 1):
            strategy = np.random.choice(strategies)
            augmented_X.append(augment_sequence(sequence, strategy))
            augmented_y.append(label)
    return np.array(augmented_X), np.array(augmented_y)
