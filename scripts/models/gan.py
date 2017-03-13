from keras.models import Sequential
from keras.layers import BatchNormalization, Activation, Input, Dense, Dropout
from keras.regularizers import l1, l1l2

def generator_model(random_samples, sample_dimension, layers, activation=None,
                    reg=lambda: l1(1e-5), batch_norm_mode=0):
    model = Sequential()
    model.add(Dense(random_samples, input_shape=(random_samples,)))
    for layer_dims in layers:
        model.add(Dense(layer_dims, W_regularizer=reg()))
        if batch_norm_mode is not None:
            model.add(BatchNormalization(mode=batch_norm_mode))
        if activation is not None:
            if type(activation) is str:
                model.add(Activation(activation))
            else:
                model.add(Activation(activation()))
    model.add(Dense(sample_dimension))
    return model

def discriminator_model(sample_dimension, layers, activation=None,
                        reg=lambda: l1l2(1e-5, 1e-5), dropout=0.25, batch_norm_mode=0):
    model = Sequential()
    model.add(Dense(sample_dimension, input_shape=(sample_dimension,)))
    for layer_dims in layers:
        model.add(Dense(layer_dims, W_regularizer=reg()))
        if batch_norm_mode is not None:
            model.add(BatchNormalization(mode=batch_norm_mode))
        if activation is not None:
            if type(activation) is str:
                model.add(Activation(activation))
            else:
                model.add(Activation(activation()))
        if dropout is not None and dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model
