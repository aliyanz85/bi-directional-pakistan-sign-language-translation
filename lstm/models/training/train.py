"""
Training entrypoint (adapted from existing repo scripts).
Saves best model and writes `model_info.json` to `models/`.
"""
import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical
from .model_architecture import build_lstm_model, create_callbacks, plot_history

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'processed_data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'models')
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15


def load_data():
    data = np.load(os.path.join(DATA_DIR, 'dataset.npz'))
    # Support both formats used in this repo
    if 'X_train' in data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        X = data['X']
        y = data['y']
        # fallback: no splits
        X_train, y_train = X, y
        X_val = X_test = X
        y_val = y_test = y

    with open(os.path.join(DATA_DIR, 'classes.json'), 'r') as f:
        classes = json.load(f)

    num_classes = len(classes)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test, classes


def train():
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_data()
    seq_len = X_train.shape[1]
    feat_size = X_train.shape[2]
    num_classes = len(classes)

    model = build_lstm_model(seq_len, feat_size, num_classes)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODEL_DIR, f'sign_language_model_{timestamp}.h5')
    log_dir = os.path.join(LOGS_DIR, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    cbs = create_callbacks(model_path, log_dir)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    model.save(os.path.join(MODEL_DIR, 'sign_language_model_final.h5'))

    info = {
        'timestamp': timestamp,
        'classes': classes,
        'sequence_length': int(seq_len),
        'feature_size': int(feat_size),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'total_parameters': int(model.count_params()),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history.get('val_accuracy', [0])))
    }
    with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    plot_history(history, os.path.join(log_dir, 'training_history.png'))
    print('Training finished. Model info written to models/model_info.json')


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    train()
