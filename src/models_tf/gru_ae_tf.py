# models_tf/gru_ae_tf.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_gru_autoencoder(
    input_dim: int,
    seq_len: int,
    hidden: int = 64,
    layers_n: int = 1,
    dropout: float = 0.0,
    lr: float = 1e-3,
    loss: str = "mse",
):
    """
    Full-sequence GRU Autoencoder (Keras).
    Encoder: stacked GRU -> latent h
    Decoder: RepeatVector(seq_len) + stacked GRU -> TimeDistributed(Dense(input_dim))
    """
    assert layers_n >= 1, "layers_n must be >= 1"

    inp = layers.Input(shape=(seq_len, input_dim))  # (B, T, D)

    # Encoder
    x = inp
    for _ in range(max(0, layers_n - 1)):
        x = layers.GRU(hidden, return_sequences=True, dropout=dropout)(x)
    h = layers.GRU(hidden, return_sequences=False, dropout=dropout)(x)  # (B, H)

    # Decoder
    y = layers.RepeatVector(seq_len)(h)  # (B, T, H)
    for _ in range(layers_n):
        y = layers.GRU(hidden, return_sequences=True, dropout=dropout)(y)

    out = layers.TimeDistributed(layers.Dense(input_dim))(y)  # (B, T, D)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    return model
