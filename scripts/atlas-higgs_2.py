import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
from utils import parse_dataset, get_train_test_data

np.set_printoptions(suppress=True)
dataset_num = 1

X, y = parse_dataset(dataset_num)
rows, cols = X.shape


starting_size = 50
encoded_dim = 5
layers = 5
nb_epoch = 5000
activation = ELU
batch_normalization = True
step = (starting_size - encoded_dim) // layers
def autoencoder():
    autoencoder = Sequential()
    autoencoder.add(Dense(starting_size, input_dim=cols))
    autoencoder.add(activation())
    if batch_normalization:
        autoencoder.add(BatchNormalization())
    for i in range(starting_size-step, encoded_dim, -step):
        autoencoder.add(Dense(i))
        autoencoder.add(activation())
        if batch_normalization:
            autoencoder.add(BatchNormalization())
    for i in range(encoded_dim, starting_size+step, step):
        autoencoder.add(Dense(i))
        autoencoder.add(activation())
        if batch_normalization:
            autoencoder.add(BatchNormalization())
    autoencoder.add(Dense(cols))
    autoencoder.compile(optimizer='adadelta', loss='mse')
    autoencoder.summary()
    return autoencoder

X_train, y_train, X_test, y_test = get_train_test_data(X, y)

# Preprocessing: imputing of NaN values and normalization
sscaler = RobustScaler()

preproc = Pipeline([('scaler', sscaler)])
preproc.fit(X_train)
X_train = preproc.transform(X_train)
X_test = preproc.transform(X_test)

# Preparing model
autoencoder = KerasRegressor(build_fn=autoencoder, nb_epoch=nb_epoch, shuffle=True, batch_size=rows//100)

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
corr = []
# np.linspace(difference.min(), difference.max(), num=thresholds)
# np.linspace(difference[y_test==0.].mean()-difference[y_test==0.].var()*2, difference[y_test==1.].mean()+difference[y_test==1.].var()*2, num=thresholds)
for n in np.linspace(difference[y_test==0.].mean()-difference[y_test==0.].var()*2, difference[y_test==1.].mean()+difference[y_test==1.].var()*2, num=thresholds):
    y_predict = (difference >= n).astype(np.bool)
    y_test = y_test.astype(np.bool)
    thresh.append(n)
    rocauc.append(roc_auc_score(y_test, y_predict))
    f1.append(f1_score(y_test, y_predict))
    prec.append(precision_score(y_test, y_predict))
    rec.append(recall_score(y_test, y_predict))
    acc.append(accuracy_score(y_test, y_predict))
    corr.append(matthews_corrcoef(y_test, y_predict))
    anom_n = y_predict.sum()
    #print(n, anom_n, len(y_predict) - anom_n)

plt.plot(thresh, rocauc, label="ROC AUC")
plt.plot(thresh, f1, label="F1")
plt.plot(thresh, prec, label="Precision")
plt.plot(thresh, rec, label="Recall")
plt.plot(thresh, acc, label="Accuracy")
plt.plot(thresh, corr, label="Corr")
plt.legend()
plt.xlim([difference[y_test==0.].mean()-difference[y_test==0.].var()*2, difference[y_test==1.].mean()+difference[y_test==1.].var()*2])
plt.ylim([0,1])
plt.savefig("../results/{0}-{1}-{2}-{3}-{4}-{5}.png".format(dataset_num, starting_size, encoded_dim, layers, nb_epoch, 'bn' if batch_normalization else 'no'))