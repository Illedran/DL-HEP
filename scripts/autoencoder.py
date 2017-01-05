from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Input

class Autoencoder(Model):

    def __init__(self, input_dimensions, latent_dimensions, layers, activation='sigmoid',
                    optimizer='adadelta', loss='mse', batch_normalization=False):
        input_layer = Input(shape=(input_dimensions,))
        x = input_layer
        for layer_dims in layers:
            x = Dense(layer_dims)(x)
            if activation is not None:
                if type(activation) is str:
                    x = Activation(activation)(x)
                else:
                    x = Activation(activation())(x)
            if batch_normalization:
                x = BatchNormalization()(x)
        encoded = Dense(latent_dimensions)(x)
        x = encoded
        for layer_dims in reversed(layers):
            x = Dense(layer_dims)(x)
            if type(activation) is str:
                x = Activation(activation)(x)
            else:
                x = Activation(activation())(x)
            if batch_normalization:
                x = BatchNormalization()(x)
        decoded = Dense(input_dimensions)(x)

        super().__init__(input=input_layer, output=decoded)

        # Encoder sub-model
        self.encoder = Model(input=input_layer, output=encoded)

        # Decoder sub-model
        encoded_input = Input(shape=(latent_dimensions,))
        x = encoded_input
        for decoding_layer in self.layers[len(self.layers) // 2 + 1:]:
            x = decoding_layer(x)

        self.decoder = Model(input=encoded_input, output=x)

        self.compile(optimizer=optimizer, loss=loss)

        self.activation = activation
        self.batch_normalization = batch_normalization

