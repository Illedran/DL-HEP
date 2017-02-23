import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, optimize_loss, layer_norm, batch_norm

class Autoencoder():
    def __init__(self, input_dimensions, latent_dimensions, layers, layer_params=None):

        if layer_params is None:
            layer_params = {
                'activation_fn': tf.nn.relu6,
                # 'normalizer_fn': layer_norm
            }

        self.input_layer = tf.placeholder(tf.float32, [None, input_dimensions])
        self.step = tf.Variable(0, trainable=False)

        x = self.input_layer
        for layer_dims in layers:
            x = fully_connected(x, layer_dims, **layer_params)

        self.encoded = fully_connected(x, latent_dimensions, activation_fn=None)

        x = self.encoded
        for layer_dims in reversed(layers):
            x = fully_connected(x, layer_dims, **layer_params)

        self.decoded = fully_connected(x, input_dimensions, activation_fn=None)

        self.optimizer = tf.train.AdamOptimizer()
        self.loss = tf.losses.mean_squared_error(self.input_layer, self.decoded)

        def decay_lr(lr, step):
            return tf.train.exponential_decay(lr, step, 10000, 0.99)

        self.model = optimize_loss(self.loss, global_step=self.step, learning_rate=0.0001,
                                   learning_rate_decay_fn=decay_lr, optimizer='Adam')


class VariationalAutoencoder():
    def __init__(self, input_dimensions, latent_dimensions, layers, activation=tf.nn.elu, learning_rate=0.001,
                 batch_size=512):

        self.input_layer = tf.placeholder(tf.float32, [batch_size, input_dimensions])
        self.step = tf.Variable(0, trainable=False)

        x = self.input_layer
        for layer_dims in layers:
            x = fully_connected(x, layer_dims, activation_fn=activation,
                                # normalizer_fn=layer_norm
                                )

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
            x = fully_connected(x, layer_dims, activation_fn=activation,
                                # normalizer_fn=layer_norm
                                )

        self.x_mu = fully_connected(x, input_dimensions, activation_fn=None,
                                    # normalizer_fn=layer_norm
                                    )
        self.x_ls2 = fully_connected(x, input_dimensions, activation_fn=None,
                                     # normalizer_fn=layer_norm
                                     )

        self.reconstr_loss = tf.reduce_sum(
            0.5 * self.x_ls2 + (tf.square(self.input_layer - self.x_mu) / (2.0 * tf.exp(self.x_ls2))), 1)

        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_ls2 - tf.square(self.z_mu) - tf.exp(self.z_ls2), 1)
        self.loss = tf.reduce_mean(self.reconstr_loss + self.latent_loss)

        def decay_lr(lr, step):
            return tf.train.exponential_decay(lr, step, 5000, 0.94)

        self.model = optimize_loss(self.loss, global_step=self.step, learning_rate=learning_rate, optimizer='Adam',
                                   clip_gradients=10., learning_rate_decay_fn=decay_lr)
