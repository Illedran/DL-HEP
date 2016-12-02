import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# Dataset direct link: http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz
a = pd.read_csv("atlas-higgs-challenge-2014-v2.csv")
a = a.drop(['EventId', 'KaggleSet', 'KaggleWeight', 'Weight'], axis=1)
a.loc[a.Label == 'b', 'Label'] = 0
a.loc[a.Label == 's', 'Label'] = 1
a[a == -999.] = np.nan
nan_cols = []
for col in a:
    if not a[col].notnull().sum()==len(a):
        nan_cols.append(col)
a = a.drop(nan_cols, axis=1)
# a=a.dropna()

data = a.values.astype(np.float32)
X = data[:, :-1]
y = data[:, -1]

rows, cols = X.shape
encoded_dims = 5


def autoencoder():
    autoencoder = Sequential()
    autoencoder.add(Dense(10, input_dim=cols))
    autoencoder.add(Dense(5))

    autoencoder.add(Dense(10))
    autoencoder.add(Dense(cols))
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    autoencoder.summary()
    return autoencoder

# Preparing dataset
X_b = X[y == 0.]  # Non-anomaly
X_s = X[y == 1.]  # Anomaly
y_b = y[y == 0.]  # Non-anomaly
y_s = y[y == 1.]  # Anomaly

# Train on 80% of non-anomaly data
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, train_size=0.8, random_state=1337)

X_train = X_b_train
y_train = y_b_train
X_test = np.vstack([X_b_test, X_s])
y_test = np.hstack([y_b_test, y_s])


# Preprocessing: imputing of NaN values and normalization
imputer = Imputer()
sscaler = StandardScaler()

preproc = Pipeline([('imputer', imputer), ('scaler', sscaler)])
preproc.fit(X_train)
X_train = preproc.transform(X_train)
X_test = preproc.transform(X_test)

# Preparing model
autoencoder = KerasRegressor(build_fn=autoencoder, nb_epoch=100, shuffle=True, batch_size=8192)

autoencoder.fit(X_train, X_train)
X_predict = autoencoder.predict(X_test)

difference = np.linalg.norm(X_predict - X_test, axis=1)

thresholds = 100
thresh = []
rocauc = []
f1 = []
prec = []
rec = []
acc = []
for n in np.linspace(0, 8, num=thresholds):#difference[y_test==0.].mean()/2, difference[y_test==1.].mean()*2, num=thresholds):
    y_predict = (difference >= n).astype(np.bool)
    y_test = y_test.astype(np.bool)
    thresh.append(n)
    rocauc.append(roc_auc_score(y_test, y_predict))
    f1.append(f1_score(y_test, y_predict))
    prec.append(precision_score(y_test, y_predict))
    rec.append(recall_score(y_test, y_predict))
    acc.append(accuracy_score(y_test, y_predict))
    anom_n = y_predict.sum()
    print(n, anom_n, len(y_predict)-anom_n)

plt.plot(thresh, rocauc, label="ROC AUC")
plt.plot(thresh, f1, label="F1")
plt.plot(thresh, prec, label="Precision")
plt.plot(thresh, rec, label="Recall")
plt.plot(thresh, acc, label="Accuracy")
plt.legend()
plt.show()
