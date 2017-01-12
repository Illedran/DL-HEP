import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
if 'DISPLAY' in os.environ:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, mean_squared_error
from tqdm import tqdm

# Datasets can be:
# 0,1,7,8,10,11,t,b,u,v
def parse_dataset(cls=None):
    atlas_data = pd.read_hdf("data/atlas-higgs_{}.hdf".format(cls), "atlas_data")
    w = atlas_data.KaggleWeight.values.astype(np.float32)
    y = atlas_data.Label.values.astype(np.float32)
    X = atlas_data.drop(['Label', #'KaggleWeight',
                         'Weight'], axis=1).values.astype(np.float32)
    return X, y, w

def get_train_test_data(X, y, w, random_state=1337, train_size=0.8):
    X_b = X[y == 0.]  # Non-anomaly
    X_s = X[y == 1.]  # Anomaly
    y_b = y[y == 0.]  # Non-anomaly
    y_s = y[y == 1.]  # Anomaly
    w_b = w[y == 0.]  # Non-anomaly
    w_s = w[y == 1.]  # Anomaly


    X_b_train, X_b_test, y_b_train, y_b_test, w_b_train, w_b_test = \
        train_test_split(X_b, y_b, w_b, train_size=train_size, random_state=random_state)
    X_train, y_train, w_train = shuffle(X_b_train, y_b_train, w_b_train, random_state=random_state)
    X_test, y_test, w_test = shuffle(np.vstack([X_b_test, X_s]), np.hstack([y_b_test, y_s]), np.hstack([w_b_test, w_s]), random_state=random_state)
    return (X_train, y_train, w_train, X_test, y_test, w_test)


def save_stats(configuration, difference, y_test):
    metrics = {
        'thresh': {
            'data': [],
            'label': 'Threshold',
            'plotted': False
        },
        'rocauc': {
            'data': [],
            'label': 'ROC AUC',
            'plotted': True
        },
        'f1': {
            'data': [],
            'label': 'F1',
            'plotted': True
        },
        'prec': {
            'data': [],
            'label': 'Precision',
            'plotted': True
        },
        'rec': {
            'data': [],
            'label': 'Recall',
            'plotted': True
        },
        'acc': {
            'data': [],
            'label': 'Accuracy',
            'plotted': True
        },
        'corr': {
            'data': [],
            'label': 'Correlation',
            'plotted': True
        },
        'num_anomalies': {
            'data': [],
            'label': 'Number of anomalies',
            'plotted': False
        },
        'num_regular': {
            'data': [],
            'label': 'Number of regular data',
            'plotted': False
        }
    }
    min_x = difference.min() # max(0, difference[y_test == 0.].mean() - difference[y_test == 0.].var())
    max_x = difference.max() # difference[y_test == 1.].mean() + difference[y_test == 1.].var()
    y_real = y_test.astype(np.bool)

    for n in tqdm(np.sort(difference)):
        y_predict = (difference > n).astype(np.bool)
        metrics['thresh']['data'].append(n)
        metrics['rocauc']['data'].append(roc_auc_score(y_real, y_predict))
        metrics['f1']['data'].append(f1_score(y_real, y_predict))
        metrics['prec']['data'].append(precision_score(y_real, y_predict))
        metrics['rec']['data'].append(recall_score(y_real, y_predict))
        metrics['acc']['data'].append(accuracy_score(y_real, y_predict))
        metrics['corr']['data'].append(matthews_corrcoef(y_real, y_predict))
        metrics['num_anomalies']['data'].append(y_predict.sum())
        metrics['num_regular']['data'].append(len(y_predict) - y_predict.sum())

    for metric in metrics:
        if metrics[metric]['plotted']:
            plt.plot(metrics['thresh']['data'], metrics[metric]['data'], label=metrics[metric]['label'])

    plt.legend()
    plt.xlim([min_x, max_x])
    plt.ylim([0, 1])

    out_file = "{0}-{1}-{2}-{3}-{4}-{5}".format(configuration['dataset_num'],
                                                configuration['starting_dim'],
                                                configuration['encoded_dim'],
                                                configuration['layers_num'],
                                                configuration['nb_epoch'],
                                                'bn' if configuration['batch_normalization'] else 'no')
    if not os.path.exists("../results/" + out_file):
        os.mkdir("../results/" + out_file)
    plt.savefig("../results/" + out_file + "/metrics_plots.png")

    with open("../results/" + out_file + "/metadata.txt", 'w') as f:
        f.write("Test set data (anomalies, non_anomalies): %d, %d\n" % (
            len(y_test[y_test == 1.]), len(y_test[y_test == 0.])))
        f.write("Reconstruction error mean (anomalies, non_anomalies): %8f, %8f\n" % (
            difference[y_test == 1.].mean(), difference[y_test == 0.].mean()))
        f.write("Reconstruction error var (anomalies, non_anomalies): %8f, %8f\n" % (
            difference[y_test == 1.].var(), difference[y_test == 0.].var()))

    pd.DataFrame(data=np.stack([metrics[metric]['data'] for metric in metrics]).T,
                 columns=[metrics[metric]['label'] for metric in metrics]).to_hdf(
        "../results/" + out_file + "/metrics.hdf", "metrics", mode='w', complib='zlib', complevel=9)

def ams(y_predict, y_true, weights):
    s = weights[np.logical_and(y_true, y_predict)].sum()
    b = weights[np.logical_and(np.logical_not(y_true), y_predict)].sum()
    b_reg = 10

    return np.sqrt(2 * ((s + b + b_reg) * np.log(s / (b + b_reg) + 1) - s))
