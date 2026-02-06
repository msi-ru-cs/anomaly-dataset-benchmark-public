import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input, optimizers

def _res_block(x, filters, kernel_size, dilation, dropout):
    x_in = x
    y = L.Conv1D(filters, kernel_size,
                 padding="causal", dilation_rate=dilation,
                 activation="relu")(x)
    y = L.Dropout(dropout)(y)
    y = L.LayerNormalization()(y)
    # match channels for residual
    if x_in.shape[-1] != filters:
        x_in = L.Conv1D(filters, 1, padding="same")(x_in)
    y = L.Add()([y, x_in])
    return y

def build_tcn_autoencoder(input_dim, seq_len,
                          hidden=64, layers_n=6, kernel_size=3,
                          dilation_base=2, dropout=0.1,
                          lr=1e-3, loss="mse"):
    inp = Input(shape=(seq_len, input_dim))

    # encoder (causal, dilated)
    x = inp
    dilations = [dilation_base ** i for i in range(layers_n)]
    for d in dilations:
        x = _res_block(x, hidden, kernel_size, d, dropout)

    # bottleneck (keep length)
    x = L.Conv1D(hidden, 1, padding="same", activation="relu")(x)

    # decoder (mirror dilations)
    for d in reversed(dilations):
        x = _res_block(x, hidden, kernel_size, d, dropout)

    out = L.TimeDistributed(L.Dense(input_dim))(x)

    model = Model(inp, out, name="tcn_autoencoder")
    model.compile(optimizer=optimizers.Adam(lr), loss=loss)
    return model
