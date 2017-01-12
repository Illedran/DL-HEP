import pandas as pd
import numpy as np
from tqdm import tqdm

# Dataset direct link: http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz
atlas_data = pd.read_csv("data/atlas-higgs-challenge-2014-v2.csv")
atlas_data['Label'].replace(['s', 'b'], [1, 0], inplace=True)
atlas_data.replace(-999., np.nan, inplace=True)

# Full HDF dataset
print("Saving full HDF dataset...")
atlas_data.to_hdf("data/atlas-higgs.hdf", "atlas_data", mode='w', complib='zlib', complevel=9)

kaggle_sets = atlas_data.KaggleSet.unique()
jet_nums = atlas_data.PRI_jet_num.unique()
atlas_data = atlas_data.drop(['EventId', 'PRI_tau_phi', 'PRI_lep_phi', 'PRI_met_phi',
                              'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], axis=1)

datasets = {}
# Kaggle sets: as separated in the original Kaggle competition
# t, b, v, u
# t: used for training
# b: used for public leaderboard
# v: used for private leaderboard
# u: unused
for k_set in tqdm(kaggle_sets):
    dataset_k = atlas_data[atlas_data.KaggleSet == k_set].drop(['KaggleSet'], axis=1)
    for jnum in tqdm(jet_nums):
        class_name = "{0}-{1}".format(k_set, jnum)
        dataset_k_num = dataset_k[dataset_k.PRI_jet_num == jnum] \
            .drop(['PRI_jet_num'], axis=1) \
            .drop(dataset_k.columns[dataset_k[dataset_k.PRI_jet_num == jnum].isnull().sum() == len(
            dataset_k[dataset_k.PRI_jet_num == jnum])], axis=1)
        if jnum == 0:
            dataset_k_num = dataset_k_num.drop(['PRI_jet_all_pt'], axis=1)
        datasets[class_name] = dataset_k_num

    datasets[k_set] = dataset_k

print("\nGenerating datasets...")
for key in tqdm(datasets):
    datasets[key].to_hdf("data/atlas-higgs_{}.hdf".format(key), "atlas_data", mode='w', complib='zlib', complevel=9)

