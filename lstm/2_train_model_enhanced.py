"""
Step 2 ENHANCED: Train LSTM model with AGGRESSIVE data augmentation
This version includes multiple augmentation strategies to handle small datasets
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
DATA_DIR = 'processed_data'
MODEL_DIR = 'models'
LOGS_DIR = 'logs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Hyperparameters - TUNED FOR SMALL DATASET
BATCH_SIZE = 4  # Smaller batch size for small dataset
EPOCHS = 200  # More epochs for better learning
LEARNING_RATE = 0.0005  # Lower learning rate for stability
PATIENCE = 30  # More patience for small dataset
AUGMENTATION_FACTOR = 10  # Generate 10x more data through augmentation

def load_data():
    """Load preprocessed data."""
    print("Loading data...")
    data = np.load(os.path.join(DATA_DIR, 'dataset.npz'))
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load class labels
    with open(os.path.join(DATA_DIR, 'classes.json'), 'r') as f:
        classes = json.load(f)
    
    print(f"✓ Original data loaded:")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Val:   {X_val.shape}")
    print(f"  - Test:  {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, classes

def augment_sequence(sequence, augmentation_type='random'):
    """
    Apply data augmentation to a single sequence.
    Multiple strategies to generate diverse training data.
    """
    augmented = sequence.copy()
    
    if augmentation_type == 'noise' or augmentation_type == 'random':
        # Add Gaussian noise (small amount)
        noise = np.random.normal(0, 0.02, sequence.shape)
        augmented = sequence + noise
    
    elif augmentation_type == 'scale' or augmentation_type == 'random':
        # Scale the sequence (simulate distance variation)
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented = sequence * scale_factor
    
    elif augmentation_type == 'time_warp':
        # Time warping - stretch or compress time
        original_length = len(sequence)
        warp_factor = np.random.uniform(0.9, 1.1)
        new_length = int(original_length * warp_factor)
        
        # Resample to new length then back to original
        indices = np.linspace(0, original_length - 1, new_length)
        warped = np.array([sequence[int(i)] for i in indices])
        
        # Resample back to original length
        indices_back = np.linspace(0, len(warped) - 1, original_length)
        augmented = np.array([warped[int(i)] for i in indices_back])
    
    elif augmentation_type == 'rotation':
        # Slight rotation simulation (modify x,y coordinates)
        angle = np.random.uniform(-0.1, 0.1)  # Small rotation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Apply rotation to x,y coordinates (not z or visibility)
        augmented = sequence.copy()
        for frame_idx in range(len(augmented)):
            # Pose: first 132 features (33 landmarks * 4)
            for i in range(0, 132, 4):
                x, y = augmented[frame_idx, i], augmented[frame_idx, i+1]
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
            
            # Face: next 1404 features (468 landmarks * 3)
            for i in range(132, 132+1404, 3):
                x, y = augmented[frame_idx, i], augmented[frame_idx, i+1]
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
            
            # Hands: remaining features
            for i in range(132+1404, len(augmented[frame_idx]), 3):
                x, y = augmented[frame_idx, i], augmented[frame_idx, i+1]
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
    
    elif augmentation_type == 'dropout':
        # Random feature dropout (simulate occlusion)
        dropout_mask = np.random.random(sequence.shape) > 0.1  # Keep 90%
        augmented = sequence * dropout_mask
    
    elif augmentation_type == 'shift':
        # Temporal shift - shift sequence in time
        shift = np.random.randint(-3, 4)
        if shift > 0:
            augmented = np.concatenate([sequence[shift:], sequence[-shift:]])
        elif shift < 0:
            augmented = np.concatenate([sequence[:shift], sequence[:-shift]])
    
    return augmented

def augment_dataset(X, y, factor=10):
    """
    Augment entire dataset by factor times using multiple strategies.
    """
    print(f"\n🔄 Augmenting dataset by {factor}x...")
    
    augmented_X = []
    augmented_y = []
    
    augmentation_strategies = ['noise', 'scale', 'time_warp', 'rotation', 'dropout', 'shift']
    
    for i, (sequence, label) in enumerate(zip(X, y)):
        # Keep original
        augmented_X.append(sequence)
        augmented_y.append(label)
        
        # Generate augmented versions
        for j in range(factor - 1):
            # Randomly choose augmentation strategy
            strategy = np.random.choice(augmentation_strategies)
            aug_seq = augment_sequence(sequence, strategy)
            augmented_X.append(aug_seq)
            augmented_y.append(label)
    
    X_aug = np.array(augmented_X)
    y_aug = np.array(augmented_y)
    
    print(f"✓ Augmentation complete:")
    print(f"  - Original samples: {len(X)}")
    print(f"  - Augmented samples: {len(X_aug)}")
    print(f"  - Increase: {len(X_aug) / len(X):.1f}x")
    
    return X_aug, y_aug

def build_model(sequence_length, feature_size, num_classes):
    """
    Build OPTIMIZED model for SMALL datasets with heavy regularization.
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(sequence_length, feature_size)),
        
        # Smaller architecture to prevent overfitting on small data
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, 
                       kernel_regularizer=keras.regularizers.l2(0.01),
                       dropout=0.4, recurrent_dropout=0.3)
        ),
        layers.BatchNormalization(),
        
        layers.Bidirectional(
            layers.LSTM(32, return_sequences=False,
                       kernel_regularizer=keras.regularizers.l2(0.01),
                       dropout=0.4, recurrent_dropout=0.3)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Dense layers with strong regularization
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        
        layers.Dense(16, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_callbacks(model_path, log_dir):
    """Create training callbacks optimized for small dataset."""
    
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    tensorboard = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    csv_logger = callbacks.CSVLogger(
        os.path.join(log_dir, 'training.csv')
    )
    
    return [checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]

def plot_history(history, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")

def train_model():
    """Main training function with aggressive augmentation."""
    print("="*60)
    print("ENHANCED TRAINING - OPTIMIZED FOR SMALL DATASET")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_data()
    
    # AUGMENT TRAINING DATA AGGRESSIVELY
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, factor=AUGMENTATION_FACTOR)
    
    # One-hot encode
    num_classes = len(classes)
    y_train_aug = to_categorical(y_train_aug, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    sequence_length = X_train_aug.shape[1]
    feature_size = X_train_aug.shape[2]
    
    print(f"\n📐 Model Configuration:")
    print(f"  - Sequence Length: {sequence_length}")
    print(f"  - Feature Size: {feature_size}")
    print(f"  - Number of Classes: {num_classes}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Max Epochs: {EPOCHS}")
    print(f"  - Augmented Training Samples: {len(X_train_aug)}")
    
    # Build model
    print("\n🏗️  Building optimized model...")
    model = build_model(sequence_length, feature_size, num_classes)
    
    # Compile with class weights for balanced learning
    class_weights = {0: 1.0, 1: 1.0}  # Equal weights for balanced classes
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\n💾 Total Parameters: {total_params:,}")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'sign_model_enhanced_{timestamp}.h5')
    log_dir = os.path.join(LOGS_DIR, f'enhanced_{timestamp}')
    
    # Create callbacks
    training_callbacks = create_callbacks(model_path, log_dir)
    
    print("\n🚀 Starting training with augmented data...")
    print("="*60)
    
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=training_callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Detailed predictions on test set
    test_predictions = model.predict(X_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(y_test, axis=1)
    
    print(f"\n🎯 Final Results:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"\n📋 Test Predictions:")
    for i, (true_class, pred_class, confidence) in enumerate(zip(test_true_classes, test_pred_classes, test_predictions)):
        true_label = classes[true_class]
        pred_label = classes[pred_class]
        conf = confidence[pred_class] * 100
        status = "✓" if true_class == pred_class else "✗"
        print(f"  {status} Sample {i+1}: True={true_label}, Predicted={pred_label} ({conf:.1f}%)")
    
    # Plot history
    plot_path = os.path.join(log_dir, 'training_history.png')
    plot_history(history, plot_path)
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'sign_language_model_final.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Save info
    info = {
        'timestamp': timestamp,
        'classes': classes,
        'sequence_length': int(sequence_length),
        'feature_size': int(feature_size),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'total_parameters': int(total_params),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'augmentation_factor': AUGMENTATION_FACTOR,
        'training_samples': int(len(X_train_aug))
    }
    
    with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Model info saved to {MODEL_DIR}/model_info.json")
    print("="*60)
    
    return model, history, test_accuracy

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model, history, accuracy = train_model()
    
    print(f"\n🎉 Enhanced training completed!")
    print(f"Final test accuracy: {accuracy*100:.2f}%")
    print(f"\n💡 Model has been trained with {AUGMENTATION_FACTOR}x data augmentation")
    print(f"This should significantly improve real-time detection!")
