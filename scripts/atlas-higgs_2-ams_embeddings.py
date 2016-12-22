import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from scripts.utils import parse_dataset
from scripts.autoencoder import create_dense_autoencoder
from keras.layers.advanced_activations import ELU
import os
from tqdm import tqdm

# No import errors
if 'DISPLAY' in os.environ:
    DISPLAY = True
    import matplotlib.pyplot as plt
else:
    import matplotlib

    matplotlib.use('Agg')
    DISPLAY = False
    import matplotlib.pyplot as plt


def show_graph(name):
    if DISPLAY:
        plt.show()
    else:
        plt.savefig(name)


def ams(y_test, w_test, reconstruction_error, thresh):
    y_predict = reconstruction_error <= thresh
    s = w_test[np.logical_and(y_test, y_predict)].sum()
    b = w_test[np.logical_and(np.logical_not(y_test), y_predict)].sum()
    b_reg = 10

    return np.sqrt(2 * ((s + b + b_reg) * np.log(s / (b + b_reg) + 1) - s))


np.set_printoptions(suppress=True)

nb_epoch = 250
latent_dimensions = 2
layers = [64, 32, 16, 8, 4]
activation = ELU
loss = 'mse'
batch_normalization = True

# Train data
X_train, y_train, _, _ = parse_dataset(cls='t')
X_train = X_train[y_train == 0.]
X_test, y_test, w_test, _ = parse_dataset(cls='b')

imputer = Imputer()
sscaler = StandardScaler()
preproc = Pipeline([('imputer', imputer), ('scaler', sscaler)])

rows, cols = X_train.shape
preproc.fit(X_train)
X_train = preproc.transform(X_train)

# Preparing model
ae, encoder, decoder = create_dense_autoencoder(input_dimensions=cols, latent_dimensions=latent_dimensions,
                                                layers=layers, activation=activation, loss=loss,
                                                batch_normalization=batch_normalization)

ae.fit(X_train, X_train, nb_epoch=nb_epoch, batch_size=2 ** int(np.log2(rows) * 0.6))

X_test = preproc.transform(X_test)

embedding_signal = encoder.predict(X_test[y_test == 1.])
embedding_background = encoder.predict(X_test[y_test == 0.])
plt.scatter(embedding_signal[:, 0], embedding_signal[:, 1], c='green', alpha=0.4)
plt.scatter(embedding_background[:, 0], embedding_background[:, 1], c='red', alpha=0.4)
show_graph("results/embeddings_scatter.png")

# embeddings = encoder.predict(X_test)
# x_axis = np.sort(embeddings[::10])
# ams_values = [ams(y_test, w_test, embeddings, err) for err in tqdm(x_axis)]
# plt.plot(x_axis, ams_values)
# show_graph("results/ams_plot_embeddings.png")
