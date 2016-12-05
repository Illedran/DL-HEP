import pandas as pd
import numpy as np

# Dataset direct link: http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz
atlas_data = pd.read_csv("data/atlas-higgs-challenge-2014-v2.csv")
atlas_data.drop(['EventId', 'KaggleSet', 'KaggleWeight', 'Weight'], axis=1, inplace=True)
atlas_data['Label'].replace(['s', 'b'], [1, 0], inplace=True)
atlas_data.replace(-999., np.nan, inplace=True)

data_classes = atlas_data.isnull().sum(axis=1)

datasets = {}
for cls in data_classes.unique():
    datasets[cls] = atlas_data[data_classes == cls].dropna(axis=1)

for key in datasets:
    datasets[key].to_hdf("data/atlas-higgs_{}.hdf".format(key), "atlas_data", mode='w', complib='zlib', complevel=9)

with open('data/atlas-higgs_nan-classes.txt', 'w') as f:
    f.write(','.join(map(str, datasets.keys())))