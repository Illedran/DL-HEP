import numpy as np
import pandas as pd
from scripts.final_experiments.models.autoencoder import Autoencoder, VariationalAutoencoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm
from collections import OrderedDict
import json
import os
from scripts.final_experiments.utils import batch_generator


def AutoencoderTrain(params):
    X_train = pd.read_hdf(
        "scripts/final_experiments/data/{0}_train_{1}.hdf".format(params['name_prefix'], params['jet_num']),
        "X").values.astype(np.float32)
    y_train = pd.read_hdf(
        "scripts/final_experiments/data/{0}_train_{1}.hdf".format(params['name_prefix'], params['jet_num']), "y")
    X_val = pd.read_hdf("scripts/final_experiments/data/{0}_val_{1}.hdf".format(params['name_prefix'], params['jet_num']),
                        "X").values.astype(np.float32)
    y_val = pd.read_hdf("scripts/final_experiments/data/{0}_val_{1}.hdf".format(params['name_prefix'], params['jet_num']),
                        "y")

    X_b = X_train[y_train == 0]
    X_b = X_b
    rows, cols = X_b.shape

    ss = StandardScaler()

    ss.fit(X_b)

    X_b = ss.transform(X_b)
    X_b_val = ss.transform(X_val[y_val == 0])

    tf.reset_default_graph()
    if params['model_type'] == 'ae':
        model = Autoencoder(cols, 2, [50, 25])
    elif params['model_type'] == 'vae':
        model = VariationalAutoencoder(cols, 2, [50, 25])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    bar_postfix_data = OrderedDict()
    history = OrderedDict()
    history['train_loss'] = []
    history['val_loss'] = []

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(params['epochs']):
            with tqdm(desc="Epoch {0:04d}".format(epoch + 1), total=rows, ncols=200) as bar:
                for batch in batch_generator(np.random.permutation(X_b), params['batch_size']):
                    b_size = len(batch)
                    _, loss = sess.run([model.model, model.loss], feed_dict={model.input_layer: batch})
                    # Update data and bar
                    bar.update(b_size)
                train_loss = sess.run(model.loss, feed_dict={model.input_layer: X_b})
                bar_postfix_data['train_loss'] = train_loss
                history['train_loss'].append(float(train_loss))  # np.float32 are not json serializable
                bar.set_postfix(bar_postfix_data)
                val_loss = sess.run(model.loss, feed_dict={model.input_layer: X_b_val})
                bar_postfix_data['val_loss'] = val_loss
                history['val_loss'].append(float(val_loss))  # np.float32 are not json serializable
                bar.set_postfix(bar_postfix_data)

            if epoch % 10 == 0 or epoch == params['epochs'] - 1:
                data_path = "{0}_{2}_jn{1}".format(params['name_prefix'], params['jet_num'], params['model_type'])
                directory = 'scripts/final_experiments/results/{0}'.format(data_path)
                os.makedirs(directory, exist_ok=True)
                saver.save(sess, directory + "/{0}.ckpt".format(data_path))
                json.dump(history, open(directory + "/history.json".format(data_path), 'w'), indent=4)

def main():
    params = {
        'batch_size': 256,
        'epochs': 2000,
    }

    for model_type in ['ae', 'vae']:
        for name_prefix in ["atlas-higgs_esperiment1", "atlas-higgs_esperiment2"]:
            for jet_num in [0, 1, 2, 3]:
                params['jet_num'] = jet_num
                params['name_prefix'] = name_prefix
                params['model_type'] = model_type
                AutoencoderTrain(params)

if __name__ == '__main__':
    main()


# sess = tf.Session()
# saver = tf.train.Saver()
# ae = VariationalAutoencoder(14, 2, [50, 25])
# saver.restore(sess, "scripts/final_experiments/results/atlas-higgs_esperiment2_jn0/atlas-higgs_esperiment2_jn0.ckpt")