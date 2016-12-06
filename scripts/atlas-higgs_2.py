import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.activations import *
from utils import parse_dataset, get_train_test_data, save_stats
from keras import backend as K

np.set_printoptions(suppress=True)

configuration = {
    'dataset_num': 1,
    'starting_dim': 197,
    'encoded_dim': 17,
    'layers_num': 15,
    'nb_epoch': 500,
    'batch_normalization': True,
    'activation': sigmoid,
    'batch_size': 32
}
step = (configuration['starting_dim'] - configuration['encoded_dim']) // configuration['layers_num']

X, y = parse_dataset(configuration['dataset_num'])
rows, cols = X.shape

X_train, y_train, X_test, y_test = get_train_test_data(X, y)

# Preprocessing: imputing of NaN values and normalization
sscaler = MinMaxScaler()

preproc = Pipeline([('scaler', sscaler)])
preproc.fit(X_train)
X_train = preproc.transform(X_train)
X_test = preproc.transform(X_test)

# Preparing model

autoencoder = Sequential()
autoencoder.add(Dense(configuration['starting_dim'], input_dim=cols))
if configuration['activation'] is not None:
    autoencoder.add(Activation(configuration['activation']))
if configuration['batch_normalization']:
    autoencoder.add(BatchNormalization())
for i in range(configuration['starting_dim'] - step, configuration['encoded_dim'], -step):
    autoencoder.add(Dense(i))
    if configuration['activation'] is not None:
        autoencoder.add(Activation(configuration['activation']))
    if configuration['batch_normalization']:
        autoencoder.add(BatchNormalization())
for i in range(configuration['encoded_dim'], configuration['starting_dim'] + step, step):
    autoencoder.add(Dense(i))
    if configuration['activation'] is not None:
        autoencoder.add(Activation(configuration['activation']))
    if configuration['batch_normalization']:
        autoencoder.add(BatchNormalization())
autoencoder.add(Dense(cols))
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.summary()

autoencoder.fit(X_train, X_train, nb_epoch=configuration['nb_epoch'], shuffle=True,
                batch_size=configuration['batch_size'], validation_split=0.2)
X_predict = autoencoder.predict(X_test, batch_size=configuration['batch_size'], verbose=1)

difference = np.mean(np.square(X_predict - X_test), axis=1)

print("Saving stats...")
save_stats(configuration, difference, y_test)
