"""
Step 2: Train LSTM model with best practices
Professional implementation with proper callbacks, monitoring, and model saving
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

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15  # Early stopping patience

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
    
    num_classes = len(classes)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"✓ Loaded data:")
    print(f"  - Train: {X_train.shape}, {y_train.shape}")
    print(f"  - Val:   {X_val.shape}, {y_val.shape}")
    print(f"  - Test:  {X_test.shape}, {y_test.shape}")
    print(f"  - Classes: {classes}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, classes

def build_model(sequence_length, feature_size, num_classes):
    """
    Build LSTM model with best practices:
    - Bidirectional LSTM for better temporal understanding
    - Dropout for regularization
    - Batch normalization for stable training
    - L2 regularization to prevent overfitting
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(sequence_length, feature_size)),
        
        # First Bidirectional LSTM layer
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, 
                       kernel_regularizer=keras.regularizers.l2(0.001))
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second Bidirectional LSTM layer
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True,
                       kernel_regularizer=keras.regularizers.l2(0.001))
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third LSTM layer (not bidirectional to reduce parameters)
        layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_callbacks(model_path, log_dir):
    """Create training callbacks for monitoring and optimization."""
    
    # Model checkpoint - save best model
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Early stopping - stop if no improvement
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # TensorBoard for visualization
    tensorboard = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # CSV logger
    csv_logger = callbacks.CSVLogger(
        os.path.join(log_dir, 'training.csv')
    )
    
    return [checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]

def plot_history(history, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
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
    """Main training function."""
    print("="*60)
    print("SIGN LANGUAGE MODEL TRAINING - PROFESSIONAL PIPELINE")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_data()
    
    sequence_length = X_train.shape[1]
    feature_size = X_train.shape[2]
    num_classes = len(classes)
    
    print(f"\n📐 Model Configuration:")
    print(f"  - Sequence Length: {sequence_length}")
    print(f"  - Feature Size: {feature_size}")
    print(f"  - Number of Classes: {num_classes}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Max Epochs: {EPOCHS}")
    
    # Build model
    print("\n🏗️  Building model...")
    model = build_model(sequence_length, feature_size, num_classes)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\n💾 Total Parameters: {total_params:,}")
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'sign_language_model_{timestamp}.h5')
    log_dir = os.path.join(LOGS_DIR, f'run_{timestamp}')
    
    # Create callbacks
    training_callbacks = create_callbacks(model_path, log_dir)
    
    # Train model
    print("\n🚀 Starting training...")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=training_callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n🎯 Final Results:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Plot training history
    plot_path = os.path.join(log_dir, 'training_history.png')
    plot_history(history, plot_path)
    
    # Save final model (in case best model callback didn't trigger)
    final_model_path = os.path.join(MODEL_DIR, 'sign_language_model_final.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Save training info
    info = {
        'timestamp': timestamp,
        'classes': classes,
        'sequence_length': int(sequence_length),
        'feature_size': int(feature_size),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'total_parameters': int(total_params),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    }
    
    with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Model info saved to {MODEL_DIR}/model_info.json")
    print("="*60)
    
    return model, history, test_accuracy

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model, history, accuracy = train_model()
    
    print(f"\n🎉 Training pipeline completed successfully!")
    print(f"Final test accuracy: {accuracy*100:.2f}%")
