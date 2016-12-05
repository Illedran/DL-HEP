import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
import os


def parse_dataset(nan_class=None):
    with open("../data/atlas-higgs_nan-classes.txt") as f:
        classes = map(int, f.read().split(','))
    if nan_class == None:
        with open("../data/atlas-higgs_nan-classes.txt") as f:
            classes = f.read().split(',')
            datasets = {}
            for cls in classes:
                atlas_data = pd.read_hdf("../data/atlas-higgs_{}.hdf".format(cls), "atlas_data").values.astype(
                    np.float32)
                X = atlas_data[:, :-1]
                y = atlas_data[:, -1]
                datasets[cls] = (X, y)
            return datasets
    elif nan_class in classes:
        atlas_data = pd.read_hdf("../data/atlas-higgs_{}.hdf".format(nan_class), "atlas_data").values.astype(np.float32)
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


def save_stats(configuration, difference, y_test):
    thresholds = 250
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
    # np.linspace(difference.min(), difference.max(), num=thresholds)
    # np.linspace(difference[y_test==0.].mean()-difference[y_test==0.].var()*2, difference[y_test==1.].mean()+difference[y_test==1.].var()*2, num=thresholds)
    min_x = max(0, difference[y_test == 0.].mean() - difference[y_test == 0.].var() * 5)
    max_x = difference[y_test == 1.].mean() + difference[y_test == 1.].var() * 5
    for n in np.linspace(min_x, max_x, num=thresholds):
        y_predict = (difference >= n).astype(np.bool)
        y_test = y_test.astype(np.bool)
        metrics['thresh']['data'].append(n)
        metrics['rocauc']['data'].append(roc_auc_score(y_test, y_predict))
        metrics['f1']['data'].append(f1_score(y_test, y_predict))
        metrics['prec']['data'].append(precision_score(y_test, y_predict))
        metrics['rec']['data'].append(recall_score(y_test, y_predict))
        metrics['acc']['data'].append(accuracy_score(y_test, y_predict))
        metrics['corr']['data'].append(matthews_corrcoef(y_test, y_predict))
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
        f.write("Reconstruction error mean (anomalies, non_anomalies): %5f, %5f\n" % (
            difference[y_test == 0.].mean(), difference[y_test == 1.].mean()))
        f.write("Reconstruction error var (anomalies, non_anomalies): %5f, %5f\n" % (
            difference[y_test == 0.].var(), difference[y_test == 1.].var()))

    pd.DataFrame(data=np.stack([metrics[metric]['data'] for metric in metrics]).T,
                 columns=[metrics[metric]['label'] for metric in metrics]).to_hdf(
        "../results/" + out_file + "/metrics.hdf", "metrics", mode='w', complib='zlib', complevel=9)
