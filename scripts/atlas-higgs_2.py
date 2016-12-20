import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from utils import parse_dataset, get_train_test_data, save_stats
from autoencoder import create_dense_autoencoder
import os

# No import errors
if 'DISPLAY' in os.environ:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

configuration = {
    'dataset_num': 1,
    'nb_epoch': 250,
    'batch_size': 32
}

X, y = parse_dataset(configuration['dataset_num'])
rows, cols = X.shape

X_train, y_train, X_test, y_test = get_train_test_data(X, y)

# Preprocessing: imputing of NaN values and normalization
sscaler = StandardScaler()

preproc = Pipeline([('scaler', sscaler)])
preproc.fit(X_train)
X_train = preproc.transform(X_train)
X_test = preproc.transform(X_test)

# Preparing model
ae, encoder, decoder = create_dense_autoencoder(input_dimensions=cols, latent_dimensions=2, layers=[16, 8, 4],
                                                activation='softplus', loss='mse')

ae.fit(X_train, X_train, nb_epoch=configuration['nb_epoch'], batch_size=configuration['batch_size'])

embedding_background = encoder.predict(X_test[y_test == 0.])
embedding_signal = encoder.predict(X_test[y_test == 1.])
plt.scatter(embedding_background[:, 0], embedding_background[:, 1], c='red', alpha=0.4)
plt.scatter(embedding_signal[:, 0], embedding_signal[:, 1], c='green', alpha=0.4)
plt.savefig("../results/embedding_scatter.png")