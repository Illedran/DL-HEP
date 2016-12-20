from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Activation, Input

def create_dense_autoencoder(input_dimensions, latent_dimensions, layers, activation='sigmoid',
                             optimizer='adadelta', loss='mse', batch_normalization=False):
    input_layer = Input(shape=(input_dimensions,))
    x = input_layer
    depth = len(layers)
    for layer_dims in layers:
        x = Dense(layer_dims)(x)
        if activation is not None:
            x = Activation(activation)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
    encoded = Dense(latent_dimensions)(x)
    x = encoded
    for layer_dims in reversed(layers):
        x = Dense(layer_dims)(x)
        if activation is not None:
            x = Activation(activation)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
    decoded = Dense(input_dimensions)(x)

    autoencoder = Model(input=input_layer, output=decoded)

    encoder = Model(input=input_layer, output=encoded)

    encoded_input = Input(shape=(latent_dimensions,))
    x = encoded_input
    for decoding_layer in autoencoder.layers[len(autoencoder.layers)//2+1:]:
        x = decoding_layer(x)
    decoder = Model(input=encoded_input, output=x)

    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder, encoder, decoder
