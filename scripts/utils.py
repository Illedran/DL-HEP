import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def parse_dataset(nan_class=None):
    with open("data/atlas-higgs_nan-classes.txt") as f:
        classes = map(int, f.read().split(','))
    if nan_class == None:
        with open("data/atlas-higgs_nan-classes.txt") as f:
            classes = f.read().split(',')
            datasets = {}
            for cls in classes:
                atlas_data = pd.read_hdf("data/atlas-higgs_{}.hdf".format(cls), "atlas_data").values.astype(np.float32)
                X = atlas_data[:, :-1]
                y = atlas_data[:, -1]
                datasets[cls] = (X, y)
            return datasets
    elif nan_class in classes:
        atlas_data = pd.read_hdf("data/atlas-higgs_{}.hdf".format(nan_class), "atlas_data").values.astype(np.float32)
        X = atlas_data[:, :-1]
        y = atlas_data[:, -1]
        return X, y
    else:
        return None

def get_train_test_data(X, y, random_state=1337):
    X_b = X[y == 0.]  # Non-anomaly
    X_s = X[y == 1.]  # Anomaly
    y_b = y[y == 0.]  # Non-anomaly
    y_s = y[y == 1.]  # Anomaly

    # Train on 80% of non-anomaly data
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, train_size=0.8, random_state=random_state)

    X_train, y_train = shuffle(X_b_train, y_b_train, random_state=random_state)
    X_test, y_test = shuffle(np.vstack([X_b_test, X_s]), np.hstack([y_b_test, y_s]), random_state=random_state)

    return (X_train, y_train, X_test, y_test)