import os
from keras.callbacks import TensorBoard
from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras_adversarial.unrolled_optimizer import UnrolledAdversarialOptimizer
import keras.backend as K

from scripts.models.gan import generator_model, discriminator_model

import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import ELU, LeakyReLU
import numpy as np
from keras.optimizers import Nadam, Adam


def example_gan(adversarial_optimizer, path, X, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy'):
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
    history = model.fit(x=X, y=y, nb_epoch=nb_epoch, batch_size=32)

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)

    # save models
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))


def main():
    X_train = pd.read_hdf("data/atlas-higgs-train_0.hdf", "X").values.astype(np.float32)
    y_train = pd.read_hdf("data/atlas-higgs-train_0.hdf", "y")
    w_train = pd.read_hdf("data/atlas-higgs-train_0.hdf", "w").values.astype(np.float32)

    X_test = pd.read_hdf("data/atlas-higgs-public-leaderboard_0.hdf", "X").values.astype(np.float32)
    y_test = pd.read_hdf("data/atlas-higgs-public-leaderboard_0.hdf", "y")
    w_test = pd.read_hdf("data/atlas-higgs-public-leaderboard_0.hdf", "w").values.astype(np.float32)

    X_train_b = X_train[y_train == 0]
    X_train_s = X_train[y_train == 1]
    w_train_b = w_train[y_train == 0]
    w_train_s = w_train[y_train == 1]

    X_test_b = X_test[y_test == 0]
    X_test_s = X_test[y_test == 1]
    w_test_b = w_test[y_test == 0]
    w_test_s = w_test[y_test == 1]

    rows, cols = X_train_b.shape
    ss = StandardScaler()

    ss.fit(X_train_b)

    X_train = ss.transform(X_train)
    X_train_b = ss.transform(X_train_b)
    X_train_s = ss.transform(X_train_s)

    X_test = ss.transform(X_test)
    X_test_b = ss.transform(X_test_b)
    X_test_s = ss.transform(X_test_s)

    latent_dim = 2
    # x \in R^{28x28}

    # generator (z -> x)
    generator = generator_model(latent_dim, cols, layers=[4, 8, 12], activation=LeakyReLU)
    # discriminator (x -> y)
    discriminator = discriminator_model(cols, layers=[256, 128, 64], activation=LeakyReLU, dropout=0.5)
    example_gan(AdversarialOptimizerSimultaneous(), "scripts/gan/output", X_train_b,
                opt_g=Adam(1e-4, decay=1e-4),
                opt_d=Adam(1e-3, decay=1e-4),
                nb_epoch=500, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)


if __name__ == "__main__":
    main()
