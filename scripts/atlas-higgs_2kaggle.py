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

nb_epoch = 100
latent_dimensions = 1
layers = [20, 10, 5, 3]
activation = ELU
loss = 'mse'
batch_normalization = True

# Train data
X_train_total, y_train_total, _, _ = parse_dataset(cls='t')
X_train_total = X_train_total[y_train_total == 0.]
X_test_total, y_test_total, w_test_total, _ = parse_dataset(cls='b')

# sscaler = StandardScaler()
# preproc = Pipeline([('scaler', sscaler)])
#
# # plot AMS for each internal class
# nan_classes = np.unique(np.isnan(X_train_total).sum(axis=1))
# for nan_class in nan_classes:
#     print("Class {}".format(nan_class))
#     X_train = X_train_total[np.isnan(X_train_total).sum(axis=1) == nan_class]
#     X_train = X_train[:, np.isnan(X_train).any(axis=0) == False]
#     rows, cols = X_train.shape
#
#     X_train = preproc.fit_transform(X_train)
#
#     # Preparing model
#     ae, encoder, decoder = create_dense_autoencoder(input_dimensions=cols, latent_dimensions=latent_dimensions,
#                                                     layers=layers, activation=activation, loss=loss,
#                                                     batch_normalization=batch_normalization)
#
#     ae.fit(X_train, X_train, nb_epoch=nb_epoch, batch_size=2 ** int(np.log2(rows) * 0.5))
#
#     nan_mask_test = np.isnan(X_test_total).sum(axis=1) == nan_class
#     X_test = X_test_total[nan_mask_test]
#     X_test = preproc.transform(X_test[:, np.isnan(X_test).any(axis=0) == False])
#     y_test = y_test_total[nan_mask_test]
#     w_test = w_test_total[nan_mask_test]
#
#     X_predict = ae.predict(X_test)
#     reconstruction_error = np.linalg.norm(X_predict - X_test, axis=1)
#     x_axis = np.sort(reconstruction_error)
#     ams_values = [ams(y_test, w_test, reconstruction_error, err) for err in tqdm(x_axis)]
#
#     # embedding_signal = encoder.predict(X_test[y_test == 1.])
#     # embedding_background = encoder.predict(X_test[y_test == 0.])
#     # plt.scatter(embedding_signal[:, 0], embedding_signal[:, 1], c='green', alpha=0.4)
#     # plt.scatter(embedding_background[:, 0], embedding_background[:, 1], c='red', alpha=0.4)
#     # show_graph("results/embedding_scatter.png")
#     plt.plot(x_axis, ams_values, label='{}'.format(nan_class))
#
# # Plot AMS for all classes, with imputing
# print("All")
imputer = Imputer()
sscaler = StandardScaler()
preproc = Pipeline([('imputer', imputer), ('scaler', sscaler)])

X_train = X_train_total
rows, cols = X_train.shape

preproc.fit(X_train)
X_train = preproc.transform(X_train)

# Preparing model
ae, encoder, decoder = create_dense_autoencoder(input_dimensions=cols, latent_dimensions=latent_dimensions,
                                                layers=layers, activation=activation, loss=loss,
                                                batch_normalization=batch_normalization)

ae.fit(X_train, X_train, nb_epoch=nb_epoch, batch_size=2 ** int(np.log2(rows) * 0.6))

X_test = preproc.transform(X_test_total)
y_test = y_test_total
w_test = w_test_total

X_predict = ae.predict(X_test)
reconstruction_error = np.linalg.norm(X_predict - X_test, axis=1)
x_axis = np.sort(reconstruction_error[::10])
ams_values = [ams(y_test, w_test, reconstruction_error, err) for err in tqdm(x_axis)]

plt.plot(x_axis, ams_values, label='all')

plt.legend()
show_graph("results/ams_plot.png")

# embedding_signal = encoder.predict(X_test[y_test == 1.]).flatten()
# embedding_background = encoder.predict(X_test[y_test == 0.]).flatten()
# plt.scatter(embedding_signal[:, 0], embedding_signal[:, 1], c='green', alpha=0.4)
# plt.scatter(embedding_background[:, 0], embedding_background[:, 1], c='red', alpha=0.4)
# plt.plot(np.sort(embedding_signal), c='green')
# plt.plot(np.sort(embedding_background), c='red')
show_graph("results/embedding_scatter.png")


embeddings = encoder.predict(X_test).flatten()
x_axis = np.sort(embeddings[::10])
ams_values = [ams(y_test, w_test, embeddings, err) for err in tqdm(x_axis)]
plt.plot(x_axis, ams_values)
show_graph("results/ams_plot_embeddings.png")
