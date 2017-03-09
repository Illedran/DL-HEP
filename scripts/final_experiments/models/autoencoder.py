import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, optimize_loss, layer_norm, batch_norm, dropout

class Autoencoder():
    def __init__(self, input_dimensions, latent_dimensions, layers, layer_params=None):

        if layer_params is None:
            layer_params = {
                'activation_fn': tf.nn.relu6,
                #'normalizer_fn': layer_norm
            }

        self.input_layer = tf.placeholder(tf.float32, [None, input_dimensions])
        self.dropout = tf.placeholder(tf.float32)
        self.step = tf.Variable(0, trainable=False)

        x = self.input_layer
        for layer_dims in layers:
            x = dropout(x, 1-self.dropout)
            x = fully_connected(x, layer_dims, **layer_params)

        x = tf.nn.dropout(x, 1-self.dropout)
        self.encoded = fully_connected(x, latent_dimensions, activation_fn=None)

        x = self.encoded
        for layer_dims in reversed(layers):
            x = dropout(x, 1-self.dropout)
            x = fully_connected(x, layer_dims, **layer_params)

        x = tf.nn.dropout(x, 1-self.dropout)
        self.decoded = fully_connected(x, input_dimensions, activation_fn=None)

        self.optimizer = tf.train.AdamOptimizer()
        self.sloss = tf.reduce_mean(tf.square(self.input_layer-self.decoded), axis=1)
        self.loss = tf.reduce_mean(self.sloss)

        self.model = optimize_loss(self.loss, global_step=self.step, learning_rate=1e-4, optimizer='Adam')


class VariationalAutoencoder():
    def __init__(self, input_dimensions, latent_dimensions, layers, layer_params=None):

        if layer_params is None:
            layer_params = {
                'activation_fn': tf.nn.relu6,
                #'normalizer_fn': layer_norm
            }

        self.input_layer = tf.placeholder(tf.float32, [None, input_dimensions])
        self.dropout = tf.placeholder(tf.float32)
        batch_size = tf.shape(self.input_layer)[0]
        self.step = tf.Variable(0, trainable=False)

        x = self.input_layer
        for layer_dims in layers:
            x = dropout(x, 1 - self.dropout)
            x = fully_connected(x, layer_dims, **layer_params)

        x = tf.nn.dropout(x, 1-self.dropout)

        self.z_mu = fully_connected(x, latent_dimensions, activation_fn=None,
                                    # normalizer_fn=layer_norm
                                    )
        self.z_ls2 = fully_connected(x, latent_dimensions, activation_fn=None,
                                     # normalizer_fn=layer_norm
                                     )

        self.samples = tf.random_normal([batch_size, latent_dimensions], mean=0., stddev=1., dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_ls2)) * self.samples

        x = self.z
        for layer_dims in reversed(layers):
            x = dropout(x, 1 - self.dropout)
            x = fully_connected(x, layer_dims, **layer_params)

        x = tf.nn.dropout(x, 1-self.dropout)

        self.x_mu = fully_connected(x, input_dimensions, activation_fn=None,
                                    # normalizer_fn=layer_norm
                                    )
        self.x_ls2 = fully_connected(x, input_dimensions, activation_fn=None,
                                     # normalizer_fn=layer_norm
                                     )

        self.reconstr_loss = tf.reduce_sum(
            0.5 * self.x_ls2 + (tf.square(self.input_layer - self.x_mu) / (2.0 * tf.exp(self.x_ls2))), 1)

        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_ls2 - tf.square(self.z_mu) - tf.exp(self.z_ls2), 1)
        self.sloss = self.reconstr_loss + self.latent_loss
        self.loss = tf.reduce_mean(self.sloss)


        self.model = optimize_loss(self.loss, global_step=self.step, learning_rate=1e-4, optimizer='Adam')
