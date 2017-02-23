import numpy as np
import pandas as pd
from scripts.final_experiments.models.autoencoder import Autoencoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm
from collections import OrderedDict
import json
import os
from scripts.final_experiments.utils import batch_generator


def AutoencoderTrain(params):
    X = pd.read_hdf("scripts/final_experiments/data/atlas-higgs_{}.hdf".format(params['jet_num']), "X").values.astype(
        np.float32)
    y = pd.read_hdf("scripts/final_experiments/data/atlas-higgs_{}.hdf".format(params['jet_num']), "y")
    w = pd.read_hdf("scripts/final_experiments/data/atlas-higgs_{}.hdf".format(params['jet_num']), "w").values.astype(
        np.float32)

    X_b = X[y == 0]
    X_s = X[y == 1]
    w_b = w[y == 0]
    w_s = w[y == 1]

    rows, cols = X_b.shape

    ss = StandardScaler()

    ss.fit(X_b)

    X = ss.transform(X)
    X_b = ss.transform(X_b)
    X_s = ss.transform(X_s)

    ae = Autoencoder(cols, params['latent_dimensions'], params['layers_dimensions'])

    init = tf.global_variables_initializer()
    bar_postfix_data = OrderedDict()
    history = OrderedDict()
    history['train_loss'] = []
    history['val_loss'] = []
    history['signal_loss'] = []

    idxes = np.arange(X_b.shape[0])
    np.random.shuffle(idxes)
    fraction = int(len(idxes) * (1 - params['validation_fraction']))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(params['epochs']):
            with tqdm(desc="Epoch {0:04d}".format(epoch + 1), total=len(X_b[idxes[:fraction]]), ncols=200) as bar:
                for batch in batch_generator(np.random.permutation(X_b[idxes[:fraction]]), params['batch_size']):
                    b_size = len(batch)
                    _, loss = sess.run([ae.model, ae.loss], feed_dict={ae.input_layer: batch})
                    # Update data and bar
                    bar.update(b_size)
                train_loss = sess.run(ae.loss, feed_dict={ae.input_layer: X_b[idxes[:fraction]]})
                bar_postfix_data['train_loss'] = train_loss
                history['train_loss'].append(float(train_loss))  # np.float32 are not json serializable

                val_loss = sess.run(ae.loss, feed_dict={ae.input_layer: X_b[idxes[fraction:]]})
                bar_postfix_data['val_loss'] = val_loss
                history['val_loss'].append(float(val_loss))

                signal_loss = sess.run(ae.loss, feed_dict={ae.input_layer: X_s})
                bar_postfix_data['signal_loss'] = signal_loss
                history['signal_loss'].append(float(signal_loss))

                bar.set_postfix(bar_postfix_data)
            if epoch % 10 == 0 or epoch == params['epochs'] - 1:
                data_path = "ae_jn{0}_z{1}_{2}".format(params['jet_num'], params['latent_dimensions'],
                                                       'x'.join(map(str, params['layers_dimensions'])))
                directory = 'scripts/final_experiments/results/{0}'.format(data_path)
                os.makedirs(directory, exist_ok=True)
                saver.save(sess, directory + "/{0}.ckpt".format(data_path))
                json.dump(history, open(directory + "/history.json".format(data_path), 'w'), indent=4)


def main():
    params = {
        'latent_dimensions': 1,
        'layers_dimensions': [50, 25],
        'batch_size': 256,
        'validation_fraction': 0.2,
        'epochs': 2000,
        'jet_num': 0
    }

    for jet_num in [0, 1, 2, 3]:
        for latent_dimensions in [1, 2, 4, 8]:
            for layers_dimensions in [[50, 25], [100, 50, 25], [250, 100, 50, 25]]:
                params['jet_num'] = jet_num
                params['latent_dimensions'] = latent_dimensions
                params['layers_dimensions'] = layers_dimensions
                AutoencoderTrain(params)

if __name__ == '__main__':
    main()