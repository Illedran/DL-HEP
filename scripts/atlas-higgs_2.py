import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.wrappers.scikit_learn import KerasRegressor
from utils import parse_dataset, get_train_test_data, save_stats

np.set_printoptions(suppress=True)


configuration = {
    'dataset_num' : 1,
    'starting_dim': 50,
    'encoded_dim': 5,
    'layers_num': 5,
    'nb_epoch': 1,
    'batch_normalization': True
}

activation = ELU
step = (configuration['starting_dim'] - configuration['encoded_dim']) // configuration['encoded_dim']

def autoencoder():
    autoencoder = Sequential()
    autoencoder.add(Dense(configuration['starting_dim'], input_dim=cols))
    autoencoder.add(activation())
    if configuration['batch_normalization']:
        autoencoder.add(BatchNormalization())
    for i in range(configuration['starting_dim'] - step, configuration['encoded_dim'], -step):
        autoencoder.add(Dense(i))
        autoencoder.add(activation())
        if configuration['batch_normalization']:
            autoencoder.add(BatchNormalization())
    for i in range(configuration['encoded_dim'], configuration['starting_dim'] + step, step):
        autoencoder.add(Dense(i))
        autoencoder.add(activation())
        if configuration['batch_normalization']:
            autoencoder.add(BatchNormalization())
    autoencoder.add(Dense(cols))
    autoencoder.compile(optimizer='adadelta', loss='mse')
    autoencoder.summary()
    return autoencoder

X, y = parse_dataset(configuration['dataset_num'])
rows, cols = X.shape

X_train, y_train, X_test, y_test = get_train_test_data(X, y)

# Preprocessing: imputing of NaN values and normalization
sscaler = RobustScaler()

preproc = Pipeline([('scaler', sscaler)])
preproc.fit(X_train)
X_train = preproc.transform(X_train)
X_test = preproc.transform(X_test)

# Preparing model
autoencoder = KerasRegressor(build_fn=autoencoder, nb_epoch=configuration['nb_epoch'], shuffle=True, batch_size=rows // 100)

autoencoder.fit(X_train, X_train)
X_predict = autoencoder.predict(X_test)

difference = np.linalg.norm(X_predict - X_test, axis=1)

print("Saving stats...")
save_stats(configuration, difference, y_test)
