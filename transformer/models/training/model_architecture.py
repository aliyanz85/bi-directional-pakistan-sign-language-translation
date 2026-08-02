"""
Model Architecture: TCN + Transformer for PSL Recognition
Production-ready architecture with temporal modeling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, BatchNormalization,
    Add, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention, Activation
)
import numpy as np


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, max_sequence_length=60, **kwargs):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        # Generate position and dimension indices
        position = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
        i = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
        
        # Calculate angle rates and apply sin/cos
        angle_rads = self.get_angles(position, i, d_model)
        
        # Create indices for even (0, 2, 4...) and odd (1, 3, 5...)
        even_indices = tf.range(0, d_model, 2)
        odd_indices = tf.range(1, d_model, 2)
        
        # Apply sin and cos
        angle_rads_even = tf.gather(angle_rads, even_indices, axis=1)
        angle_rads_odd = tf.gather(angle_rads, odd_indices, axis=1)
        
        sines = tf.sin(angle_rads_even)
        cosines = tf.cos(angle_rads_odd)
        
        # Interleave sin and cos
        # Stack and reshape to interleave
        min_len = tf.minimum(tf.shape(sines)[1], tf.shape(cosines)[1])
        sines_trimmed = sines[:, :min_len]
        cosines_trimmed = cosines[:, :min_len]
        
        pos_encoding = tf.reshape(
            tf.stack([sines_trimmed, cosines_trimmed], axis=2),
            [seq_len, min_len * 2]
        )
        
        # Pad if necessary to match d_model
        padding_needed = d_model - tf.shape(pos_encoding)[1]
        pos_encoding = tf.pad(pos_encoding, [[0, 0], [0, padding_needed]])
        
        # Add batch dimension and cast
        pos_encoding = tf.cast(pos_encoding[tf.newaxis, :, :], inputs.dtype)
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])
        
        return inputs + pos_encoding
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_sequence_length": self.max_sequence_length})
        return config


class TCNBlock(layers.Layer):
    """Temporal Convolutional Network block with residual connection"""
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # Add L2 regularization to prevent overfitting
        self.conv1 = Conv1D(filters, kernel_size, padding='causal',
                           dilation_rate=dilation_rate,
                           kernel_regularizer=keras.regularizers.l2(0.01))
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)
        
        self.conv2 = Conv1D(filters, kernel_size, padding='causal',
                           dilation_rate=dilation_rate,
                           kernel_regularizer=keras.regularizers.l2(0.01))
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(dropout_rate)
        
        self.activation = Activation('relu')
        
    def build(self, input_shape):
        # Projection layer for residual if dimensions don't match
        if input_shape[-1] != self.filters:
            self.residual_conv = Conv1D(self.filters, 1, padding='same')
        else:
            self.residual_conv = lambda x: x
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # First convolution
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.dropout1(x, training=training)
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.dropout2(x, training=training)
        
        # Residual connection
        residual = self.residual_conv(inputs)
        return self.activation(Add()([x, residual]))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate
        })
        return config


class TransformerBlock(layers.Layer):
    """Transformer encoder block"""
    
    def __init__(self, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )
        self.dropout1 = Dropout(self.dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dense(self.d_model)
        ])
        self.dropout2 = Dropout(self.dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Multi-head attention with residual
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


class PSLRecognitionModel:
    """Main PSL recognition model builder"""
    
    def __init__(self, 
                 num_classes: int = 32,
                 max_sequence_length: int = 60,
                 feature_dim: int = 188):
        """
        Initialize model builder
        
        Args:
            num_classes: Number of sign language words
            max_sequence_length: Maximum sequence length
            feature_dim: Dimension of input features
        """
        self.num_classes = num_classes
        self.max_seq_len = max_sequence_length
        self.feature_dim = feature_dim
    
    def build_tcn_transformer(self,
                             tcn_filters: int = 64,
                             tcn_layers: int = 4,
                             transformer_blocks: int = 2,
                             num_heads: int = 2,
                             ff_dim: int = 128,
                             dropout_rate: float = 0.5) -> Model:
        """
        Build TCN + Transformer architecture with STRONG REGULARIZATION
        
        REDUCED CAPACITY to prevent overfitting on small dataset (249 samples)
        
        Args:
            tcn_filters: Number of filters in TCN layers (REDUCED: 128->64)
            tcn_layers: Number of TCN blocks
            transformer_blocks: Number of transformer encoder blocks
            num_heads: Number of attention heads (REDUCED: 4->2)
            ff_dim: Feed-forward dimension in transformer (REDUCED: 256->128)
            dropout_rate: Dropout rate (INCREASED: 0.3->0.5)
        
        Returns:
            Keras Model
        """
        inputs = Input(shape=(self.max_seq_len, self.feature_dim), name='input')
        
        # Initial projection with L2 regularization
        x = Dense(tcn_filters, kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)  # Add dropout after initial projection
        
        # Temporal Convolutional Blocks with increasing dilation
        for i in range(tcn_layers):
            dilation_rate = 2 ** i
            x = TCNBlock(
                filters=tcn_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate * 0.6  # 0.3 dropout in TCN
            )(x)
        
        # Add positional encoding for transformer
        x = PositionalEncoding(max_sequence_length=self.max_seq_len)(x)
        
        # Transformer encoder blocks
        for _ in range(transformer_blocks):
            x = TransformerBlock(
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate * 0.6  # 0.3 dropout in transformer
            )(x)
        
        # Global pooling with high dropout
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.6)(x)  # High dropout before classification
        
        # Classification head - REDUCED capacity
        x = Dense(128, activation='relu', 
                 kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(dropout_rate * 0.6)(x)
        
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='PSL_TCN_Transformer')
        
        return model
    
    def build_lightweight_model(self) -> Model:
        """
        Build lightweight model for faster inference
        Suitable for real-time browser deployment
        
        Returns:
            Lightweight Keras Model
        """
        inputs = Input(shape=(self.max_seq_len, self.feature_dim), name='input')
        
        # Depthwise separable convolutions for efficiency
        x = Dense(64)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Two TCN blocks with smaller filters
        for dilation in [1, 2]:
            x = TCNBlock(
                filters=64,
                kernel_size=3,
                dilation_rate=dilation,
                dropout_rate=0.2
            )(x)
        
        # Single transformer block
        x = PositionalEncoding(max_sequence_length=self.max_seq_len)(x)
        x = TransformerBlock(num_heads=2, ff_dim=128, dropout_rate=0.1)(x)
        
        # Pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='PSL_Lightweight')
        
        return model
    
    def build_lstm_baseline(self) -> Model:
        """
        Build LSTM baseline for comparison
        
        Returns:
            LSTM-based Keras Model
        """
        inputs = Input(shape=(self.max_seq_len, self.feature_dim), name='input')
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        
        # Classification head
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='PSL_LSTM_Baseline')
        
        return model


class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        weight = tf.pow(1 - y_pred, self.gamma)
        
        # Apply focal loss
        focal_loss = self.alpha * weight * ce
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


def create_model(model_type: str = 'tcn_transformer',
                num_classes: int = 32,
                max_sequence_length: int = 60,
                feature_dim: int = 188) -> Model:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('tcn_transformer', 'lightweight', 'lstm')
        num_classes: Number of classes
        max_sequence_length: Sequence length
        feature_dim: Feature dimension
    
    Returns:
        Compiled Keras model
    """
    builder = PSLRecognitionModel(num_classes, max_sequence_length, feature_dim)
    
    if model_type == 'tcn_transformer':
        model = builder.build_tcn_transformer()
    elif model_type == 'lightweight':
        model = builder.build_lightweight_model()
    elif model_type == 'lstm':
        model = builder.build_lstm_baseline()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating PSL Recognition Models...")
    print("=" * 60)
    
    # TCN + Transformer model
    print("\n1. TCN + Transformer Model")
    model = create_model('tcn_transformer', num_classes=32)
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Test with dummy input
    dummy_input = tf.random.normal((1, 60, 188))
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be ~1.0): {tf.reduce_sum(output).numpy():.4f}")
    
    # Lightweight model
    print("\n" + "=" * 60)
    print("2. Lightweight Model")
    lightweight_model = create_model('lightweight', num_classes=32)
    print(f"Total parameters: {lightweight_model.count_params():,}")
    
    # LSTM baseline
    print("\n" + "=" * 60)
    print("3. LSTM Baseline Model")
    lstm_model = create_model('lstm', num_classes=32)
    print(f"Total parameters: {lstm_model.count_params():,}")
    
    print("\n" + "=" * 60)
    print("Model creation successful!")
