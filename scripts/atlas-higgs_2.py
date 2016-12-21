import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from scripts.utils import parse_dataset, get_train_test_data
from scripts.autoencoder import create_dense_autoencoder
import os
from keras.layers.advanced_activations import ELU
from tqdm import tqdm

# No import errors
if 'DISPLAY' in os.environ:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def ams(w_test, reconstruction_error, thresh):
    y_predict = reconstruction_error <= thresh
    s = w_test[np.logical_and(y_test, y_predict)].sum()
    b = w_test[np.logical_or(np.logical_not(y_test), y_predict)].sum()
    b_reg = 10

    return np.sqrt(2*((s+b+b_reg)*np.log(s/(b+b_reg)+1)-s))

np.set_printoptions(suppress=True)

batch_size = 1024
nb_epoch = 5

# Train data
X_train, y_train, w, ids = parse_dataset(cls='t')
rows, cols = X.shape
X_train, y_train, w_train, X_test, y_test, w_test = get_train_test_data(X, y, w)

# Preprocessing: imputing of NaN values and normalization
imputer = Imputer()
sscaler = StandardScaler()

preproc = Pipeline([('imputer', imputer), ('scaler', sscaler)])
preproc.fit(X_train)
X_train = preproc.transform(X_train)
X_test = preproc.transform(X_test)

# Preparing model
ae, encoder, decoder = create_dense_autoencoder(input_dimensions=cols, latent_dimensions=2, layers=[16, 8, 4],
                                                activation=ELU, loss='mse', batch_normalization=True)

ae.fit(X_train, X_train, nb_epoch=nb_epoch, batch_size=batch_size)

X_predict = ae.predict(X_test)
reconstruction_error = np.sort(np.linalg.norm(X_predict - X_test, axis=1))

ams_values = []
for err in tqdm(reconstruction_error):
    ams_values.append(ams(w_test, reconstruction_error, err))
# embedding_signal = encoder.predict(X_test[y_test == 1.])
# embedding_background = encoder.predict(X_test[y_test == 0.])
# plt.scatter(embedding_signal[:, 0], embedding_signal[:, 1], c='green', alpha=0.4)
# plt.scatter(embedding_background[:, 0], embedding_background[:, 1], c='red', alpha=0.4)

plt.plot(reconstruction_error, ams_values)
plt.show()
#plt.savefig("../results/embedding_scatter.png")