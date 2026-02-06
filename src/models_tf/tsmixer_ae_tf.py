# src/models_tf/tsmixer_ae_tf.py
# Keras-3 safe TSMixer autoencoder (no raw tf.* on KerasTensors)

from keras import layers as L, Model, Input, optimizers

def _mixer_block(x, *, seq_len: int, hidden: int, time_mlp: int, channel_mlp: int, dropout: float):
    # x: (B, T, C)

    # --- Token mixing (across time) ---
    y = L.LayerNormalization(epsilon=1e-6)(x)
    y = L.Permute((2, 1))(y)                    # (B, C, T)
    y = L.Dense(time_mlp, activation="gelu")(y) # mix along T
    y = L.Dropout(dropout)(y)
    y = L.Dense(seq_len)(y)                     # back to T (use static seq_len)
    y = L.Permute((2, 1))(y)                    # (B, T, C)
    x = L.Add()([x, y])

    # --- Channel mixing (across features) ---
    y = L.LayerNormalization(epsilon=1e-6)(x)
    y = L.Dense(channel_mlp, activation="gelu")(y)
    y = L.Dropout(dropout)(y)
    y = L.Dense(hidden)(y)                      # back to channel width (static hidden)
    x = L.Add()([x, y])
    return x

def build_tsmixer_autoencoder(
    input_dim: int,
    seq_len: int,
    hidden: int = 64,
    layers_n: int = 4,
    time_mlp: int = 128,
    channel_mlp: int = 128,
    dropout: float = 0.1,
    lr: float = 1e-3,
    loss: str = "mse",
):
    # Input: (T, C)
    inp = Input(shape=(seq_len, input_dim))

    # Optional channel lift to common width
    x = L.Dense(hidden)(inp)

    for _ in range(layers_n):
        x = _mixer_block(
            x,
            seq_len=seq_len,
            hidden=hidden,
            time_mlp=time_mlp,
            channel_mlp=channel_mlp,
            dropout=dropout,
        )

    # Reconstruction head per time step
    out = L.TimeDistributed(L.Dense(input_dim))(x)

    model = Model(inp, out, name="tsmixer_autoencoder")
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss)
    return model
