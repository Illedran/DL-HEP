from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Input
import numpy as np

# Automatically split into 4 subclasses based on PRI_jet_num
# [0, 1, 2, 3]
PRI_jet_num_classes = [0, 1, 2, 3]


class Autoencoder(Model):
    def __init__(self, input_dimensions, latent_dimensions, layers, activation='sigmoid',
                 batch_normalization=False, **kwargs):
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

        self.activation = activation
        self.batch_normalization = batch_normalization


# class BaggingAutoencoder():
#     def __init__(self, train_data, latent_dimensions, layers, **kwargs):
#         self.column_mask = {jet_class: np.logical_not(
#             train_data[train_data.PRI_jet_num == jet_class].isnull().sum() ==
#             len(train_data[train_data.PRI_jet_num == jet_class])
#         ) for jet_class in [0, 1, 2, 3]}
#
#         self.classes = {}
#         for jet_class in PRI_jet_num_classes:
#             self.column_mask[jet_class].PRI_jet_num = False
#             self.classes[jet_class] = Autoencoder(input_dimensions=self.column_mask[jet_class].sum(),
#                                                        latent_dimensions=latent_dimensions,
#                                                        layers=layers, **kwargs)
#
#     def fit(self, x, **kwargs):
#         for jet_class in PRI_jet_num_classes:
#             self.classes[jet_class].fit(x[x.PRI_jet_num == jet_class, self.column_mask[jet_class]], **kwargs)
#
#     def predict(self, x, **kwargs):
#         for jet_class in PRI_jet_num_classes:
#             self.classes[jet_class].predict(x[x.PRI_jet_num == jet_class, self.column_mask[jet_class]], **kwargs)