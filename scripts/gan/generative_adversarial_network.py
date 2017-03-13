import os
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
import keras.backend as K
import tensorflow as tf

from scripts.models.gan import generator_model, discriminator_model

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers.advanced_activations import ELU, LeakyReLU
import numpy as np
from keras.optimizers import Nadam, Adam


def example_gan(adversarial_optimizer, path, X, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy', params = {}):
    csvpath = os.path.join(path, "history.csv")
    if os.path.exists(csvpath):
        print("Already exists: {}".format(csvpath))
        return

    print("Training: {}".format(csvpath))
    # gan (x - > yfake, yreal), z generated on GPU
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    # print summary of models
    generator.summary()
    discriminator.summary()
    gan.summary()

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)

    # train model
    y = targets(X.shape[0])
    history = model.fit(x=X, y=y, nb_epoch=params['epochs'], batch_size=params['batch_size'])

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)

    # save models
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))

def GANTrain(params):
    X_train = pd.read_hdf(
        "data/atlas-higgs_experiment{0}_train_{1}.hdf".format(params['experiment'], params['jet_num']), "X") \
        .values.astype(np.float32)
    y_train = pd.read_hdf(
        "data/atlas-higgs_experiment{0}_train_{1}.hdf".format(params['experiment'], params['jet_num']), "y")

    X_test = pd.read_hdf(
        "data/atlas-higgs_experiment{0}_test_{1}.hdf".format(params['experiment'], params['jet_num']), "X") \
        .values.astype(np.float32)
    # y_test = pd.read_hdf(
    #     "data/atlas-higgs_experiment{0}_test_{1}.hdf".format(params['experiment'], params['jet_num']), "y")

    X_b = X_train[y_train == 0]


    ss = StandardScaler()

    ss.fit(X_b)

    X_b = ss.transform(X_b)
    X_test = ss.transform(X_test)


    rows, cols = X_b.shape
    ss = MinMaxScaler()

    ss.fit(X_b)
    ss.fit(X_test)


    latent_dim = 2
    # generator (z -> x)
    generator = generator_model(latent_dim, cols, layers=[25, 50], activation=LeakyReLU)
    # discriminator (x -> y)
    discriminator = discriminator_model(cols, layers=[30, 10, 5], activation=LeakyReLU, dropout=0.5)
    example_gan(AdversarialOptimizerSimultaneous(), "scripts/results/altas-higgs_gan_esperiment{0}_jn{1}".format(params['experiment'], params['jet_num']),
                X_b,
                opt_g=Adam(1e-4),
                opt_d=Adam(1e-4),
                nb_epoch=500, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim, params=params)


def main():
    params = {
        'epochs': 500,
        'batch_size': 512,
    }

    for experiment in [1, 2]:
        for jet_num in [0, 1, 2, 3]:
            params['jet_num'] = jet_num
            params['experiment'] = experiment
            GANTrain(params)

if __name__ == "__main__":
    main()
