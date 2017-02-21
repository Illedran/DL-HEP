from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Activation, Input, Dense, Dropout, Lambda, Flatten
import tensorflow as tf
import tensorflow.contrib.distributions as dist
import keras.backend as K
import numpy as np
from keras.optimizers import Nadam
from keras.regularizers import l1, l1l2

class GAN(Model):
    def __init__(self, random_samples, sample_dimension, gen_layers, disc_layers, disc_dropout=0.25, hidden_layer_type=Dense, layer_args={},
                 activation=None, batch_normalization=False):
        self.random_samples = random_samples
        self.gen_input = Input(shape=(random_samples,))
        x = self.gen_input
        for layer_dims in gen_layers:
            x = hidden_layer_type(layer_dims, **layer_args)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                if type(activation) is str:
                    x = Activation(activation)(x)
                else:
                    x = Activation(activation())(x)
        self.generated = Dense(sample_dimension)(x)

        self.generator = Model(input=self.gen_input, output=self.generated)
        self.generator.compile(loss='binary_crossentropy', optimizer=Nadam())

        self.disc_input = Input(shape=(sample_dimension,))
        x = self.disc_input
        for layer_dims in disc_layers:
            x = hidden_layer_type(layer_dims, **layer_args)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                if type(activation) is str:
                    x = Activation(activation)(x)
                else:
                    x = Activation(activation())(x)
            if disc_dropout is not None and disc_dropout > 0:
                x = Dropout(disc_dropout)(x)
        self.disc_V = Dense(1, activation='sigmoid')(x)

        self.discriminator = Model(self.disc_input, self.disc_V)
        d_optim = Nadam()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

        self.gan_input = Input(shape=(random_samples,))
        x = self.generator(self.gan_input)
        self.gan_V = self.discriminator(x)
        super().__init__(input=self.gan_input, output=self.gan_V)

        g_optim = Nadam()

        self.compile(loss='binary_crossentropy', optimizer=g_optim)

        self.activation = activation
        self.batch_normalization = batch_normalization

    # def set_trainable(self, value):

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
