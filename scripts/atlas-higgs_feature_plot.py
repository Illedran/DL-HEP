import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from scripts.utils import parse_dataset
from scripts.autoencoder import create_dense_autoencoder
from keras.layers.advanced_activations import ELU
import os
from tqdm import tqdm

# No import errors
if 'DISPLAY' in os.environ:
    DISPLAY = True
    import matplotlib.pyplot as plt
else:
    import matplotlib

    matplotlib.use('Agg')
    DISPLAY = False
    import matplotlib.pyplot as plt


def show_graph(name):
    if DISPLAY:
        plt.show()
    else:
        plt.savefig(name)


atlas_higgs = pd.read_hdf("data/atlas-higgs_t.hdf")
labels = atlas_higgs.Label
atlas_higgs = atlas_higgs.drop(['EventId', 'KaggleWeight', 'Weight', 'KaggleSet', 'Label'], axis=1)

for idx, col in enumerate(atlas_higgs.columns):
    print(col)
    ax = plt.subplot(5, 6, idx + 1)
    plt.hist(atlas_higgs[col][labels == 1].dropna(), color='green', alpha=0.6, normed=True)
    plt.hist(atlas_higgs[col][labels == 0].dropna(), color='red', alpha=0.6, normed=True)
    ax.set_title(col)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
