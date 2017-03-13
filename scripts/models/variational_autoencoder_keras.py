from keras.models import Model
from keras.layers import BatchNormalization, Activation, Input, Dense, Dropout, Lambda
import tensorflow as tf
import tensorflow.contrib.distributions as dist
import keras.backend as K


class VariationalAutoencoder(Model):
    def __init__(self, input_dimensions, latent_dimensions, layers, hidden_layer_type=Dense, layer_args={},
                 batch_size=64,
                 activation=None, batch_normalization=False, dropout=0.5):
        has_dropout = dropout is not None and dropout > 0
        input_layer = Input(shape=(input_dimensions,))
        x = input_layer
        if has_dropout:
            x = Dropout(dropout)(x)
        for layer_dims in layers:
            x = hidden_layer_type(layer_dims, **layer_args)(x)
            if activation is not None:
                if type(activation) is str:
                    x = Activation(activation)(x)
                else:
                    x = Activation(activation())(x)
            if batch_normalization:
                x = BatchNormalization()(x)

        def sampling(args):
            z_mu, z_ls2 = args
            epsilon = tf.random_normal(shape=(batch_size, latent_dimensions), mean=0., stddev=1.0)
            return z_mu + K.sqrt(K.exp(z_ls2)) * epsilon

        z_mu = Dense(latent_dimensions)(x)
        z_ls2 = Dense(latent_dimensions)(x)
        z = Lambda(sampling)([z_mu, z_ls2])
        x = z
        for layer_dims in reversed(layers):
            x = hidden_layer_type(layer_dims, **layer_args)(x)
            if activation is not None:
                if type(activation) is str:
                    x = Activation(activation)(x)
                else:
                    x = Activation(activation())(x)
            if batch_normalization:
                x = BatchNormalization()(x)
        x_mu = Dense(latent_dimensions)(x)
        x_ls2 = Dense(latent_dimensions)(x)

        decoded = Lambda(sampling)([x_mu, x_ls2])
        super().__init__(input=input_layer, output=decoded)

        def xent_loss(X_true, X_predict):
            return K.mean(K.binary_crossentropy(X_true, X_predict), axis=-1)

        def kl_loss(X_true, X_predict):
            latent_prior = dist.MultivariateNormalDiag([0.] * latent_dimensions, [1.] * latent_dimensions)
            approximate_posterior = dist.MultivariateNormalDiag(z_mu, K.sqrt(K.exp(z_ls2)))
            return {'kl_loss': K.mean(dist.kl(latent_prior, approximate_posterior))}

        def vae_loss(X_true, X_predict):
            xent_loss = K.sum(0.5 * x_ls2 + (tf.square(x - x_mu) / (2.0 * tf.exp(x_ls2))), 1)
            latent_prior = dist.MultivariateNormalDiag([0.] * latent_dimensions, [1.] * latent_dimensions)
            approximate_posterior = dist.MultivariateNormalDiag(z_mu, K.sqrt(K.exp(z_ls2)))
            kl_loss = dist.kl(latent_prior, approximate_posterior)
            return xent_loss + kl_loss

        self.compile(loss=vae_loss, optimizer='adam')

        self.activation = activation
        self.batch_normalization = batch_normalization
