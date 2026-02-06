# src/models_tf/transformer_ae_tf.py
# Transformer autoencoder compatible with Keras 3 (no raw tf.* on KerasTensors)

from keras import layers as L, Model, Input, optimizers, losses, ops
import numpy as np
from keras.initializers import Constant


class SinePositionalEncoding(L.Layer):
    """Precomputed sinusoidal positional encoding added to inputs."""
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)

    def build(self, input_shape):
        # Build float32 PE matrix (T, C) once, as a non-trainable weight.
        pos = np.arange(self.seq_len, dtype="float32")[:, None]           # (T, 1)
        i   = np.arange(self.d_model, dtype="float32")[None, :]           # (1, C)
        angle_rates = 1.0 / np.power(10000.0, (2 * np.floor(i / 2.0)) / self.d_model)
        angle_rads  = pos * angle_rates

        pe = np.zeros((self.seq_len, self.d_model), dtype="float32")
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pe = self.add_weight(
            name="pe",
            shape=pe.shape,
            initializer=Constant(pe),
            trainable=False,
        )

    def call(self, x):
        # x: (B, T, C) → add PE (1, T, C) with matching dtype
        T = x.shape[1]                 # static from config (seq_len)
        pe = self.pe[:T, :]            # (T, C)
        pe = ops.expand_dims(pe, 0)    # (1, T, C)
        pe = ops.cast(pe, x.dtype)
        return x + pe


def _encoder_block(x, d_model: int, heads: int, ff_mult: int, dropout: float):
    # Pre-norm attention
    y = L.LayerNormalization(epsilon=1e-6)(x)
    attn = L.MultiHeadAttention(
        num_heads=heads,
        key_dim=d_model // heads,   # per-head dim
        dropout=dropout,
    )(y, y)
    x = L.Add()([x, attn])

    # Pre-norm feed-forward
    y = L.LayerNormalization(epsilon=1e-6)(x)
    y = L.Dense(ff_mult * d_model, activation="gelu")(y)
    y = L.Dropout(dropout)(y)
    y = L.Dense(d_model)(y)
    x = L.Add()([x, y])
    return x


def build_transformer_autoencoder(
    input_dim: int,
    seq_len: int,
    d_model: int = 64,
    layers_n: int = 4,
    heads: int = 4,
    ff_mult: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
    loss: str = "mse",
):
    assert d_model % heads == 0, "d_model must be divisible by heads"

    inp = Input(shape=(seq_len, input_dim))
    x = L.Dense(d_model)(inp)                                   # input projection
    x = SinePositionalEncoding(seq_len, d_model, name="pos_enc")(x)

    for _ in range(layers_n):
        x = _encoder_block(x, d_model, heads, ff_mult, dropout)

    out = L.TimeDistributed(L.Dense(input_dim))(x)              # reconstruction head
    model = Model(inp, out, name="transformer_autoencoder")
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses.get(loss))
    return model
