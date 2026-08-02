"""
Model definitions and training utilities.
Contains two model builders: `build_lstm_model` and `build_small_enhanced_model`.
Also provides `create_callbacks` and `plot_history` used by training scripts.
"""
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, callbacks


def build_lstm_model(sequence_length, feature_size, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_size)),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001))),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001))),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_small_enhanced_model(sequence_length, feature_size, num_classes):
    # Smaller model used in the enhanced training script
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_size)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01), dropout=0.4, recurrent_dropout=0.3)),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(32, return_sequences=False, kernel_regularizer=keras.regularizers.l2(0.01), dropout=0.4, recurrent_dropout=0.3)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_callbacks(model_path, log_dir, patience=15):
    chk = callbacks.ModelCheckpoint(filepath=model_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    tb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    csv = callbacks.CSVLogger(os.path.join(log_dir, 'training.csv')) if 'os' in globals() else callbacks.CSVLogger(log_dir + '/training.csv')
    return [chk, es, rl, tb, csv]


def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history.get('accuracy', []), label='Train Accuracy')
    ax1.plot(history.history.get('val_accuracy', []), label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history.get('loss', []), label='Train Loss')
    ax2.plot(history.history.get('val_loss', []), label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
