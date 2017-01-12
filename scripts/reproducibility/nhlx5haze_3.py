from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from scripts.utils import parse_dataset

from keras.models import Sequential
from keras.layers import Dense

ensemble = []

type_size = 36

X_train, y_train, _ = parse_dataset('t')
X_test, y_test, w_test = parse_dataset('b')

rows, cols = X_train.shape

imputer = Imputer()
sscaler = StandardScaler()
preproc = Pipeline([('imputer', imputer), ('scaler', sscaler)])

X_train = preproc.fit_transform(X_train)
X_test = preproc.transform(X_test)

for _ in range(type_size):
    model = Sequential()
    model.add(Dense(50, input_dim=cols, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    model.fit(X_train, y_train, nb_epoch=100, batch_size=10)

    ensemble.append(model)

for _ in range(type_size):
    model = Sequential()
    model.add(Dense(50, input_dim=cols, activation='sigmoid'))
    model.add(Dense(25, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    model.fit(X_train, y_train, nb_epoch=100, batch_size=10)

    ensemble.append(model)

for _ in range(type_size):
    model = Sequential()
    model.add(Dense(50, input_dim=cols, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(25, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    model.fit(X_train, y_train, nb_epoch=100, batch_size=10)

    ensemble.append(model)