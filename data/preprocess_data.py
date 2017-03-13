import pandas as pd
import numpy as np
from tqdm import tqdm

for experiment in [1, 2]:
    data = pd.read_csv("data/atlas-higgs-challenge-2014-v2.csv")

    jet_nums = data.PRI_jet_num.unique()
    data = data.drop(['EventId', 'Weight'], axis=1)
    data['Label'].replace(['s', 'b'], [1, 0], inplace=True)
    data.replace(-999., np.nan, inplace=True)

    # ”t”:training, ”b”:public leaderboard, ”v”:private leaderboard, ”u”:unused
    name_prefix = "atlas-higgs_esperiment{}".format(experiment)
    test_data = data[data.KaggleSet == 'b']
    if experiment == 1:
        data = data[(data.KaggleSet == 't')]
    elif experiment == 2:
        data = data[(data.KaggleSet == 't') | (data.KaggleSet == 'v')]

    data = data.drop(['KaggleSet'], axis=1)
    test_data = test_data.drop(['KaggleSet'], axis=1)

    datasets = {}

    # This complicated for does the following:
    # For each value of PRI_jet_num, remove from the dataset all columns which are all NaN and
    # remove the PRI_jet_num column
    for jnum in tqdm(jet_nums):
        class_name = "{}".format(jnum)
        dataset_k_num = data[data.PRI_jet_num == jnum] \
            .drop(['PRI_jet_num'], axis=1) \
            .drop(data.columns[data[data.PRI_jet_num == jnum].isnull().sum() == len(
            data[data.PRI_jet_num == jnum])], axis=1)
        if jnum == 0:
            dataset_k_num = dataset_k_num.drop(['PRI_jet_all_pt'], axis=1)
        datasets[class_name] = dataset_k_num

    # This drops all columns that have 'phi' in their name and not 'DER'
    #
    for dataset in datasets:
        for column in datasets[dataset].columns:
            if 'phi' in column and 'DER' not in column:
                datasets[dataset] = datasets[dataset].drop([column], axis=1)
        datasets[dataset].fillna(data.DER_mass_MMC.mean(), inplace=True)

    for key in datasets:
        y = datasets[key].Label
        w = datasets[key].KaggleWeight
        X = datasets[key].drop(['Label', 'KaggleWeight'], axis=1)

        X.to_hdf("scripts/final_experiments/data/{0}_train_{1}.hdf".format(name_prefix, key), "X", mode='a',
                 complib='zlib', complevel=9)
        y.to_hdf("scripts/final_experiments/data/{0}_train_{1}.hdf".format(name_prefix, key), "y", mode='a',
                 complib='zlib', complevel=9)
        w.to_hdf("scripts/final_experiments/data/{0}_train_{1}.hdf".format(name_prefix, key), "w", mode='a',
                 complib='zlib', complevel=9)

    datasets = {}
    for jnum in tqdm(jet_nums):
        class_name = "{}".format(jnum)
        dataset_k_num = test_data[test_data.PRI_jet_num == jnum] \
            .drop(['PRI_jet_num'], axis=1) \
            .drop(test_data.columns[test_data[test_data.PRI_jet_num == jnum].isnull().sum() == len(
            test_data[test_data.PRI_jet_num == jnum])], axis=1)
        if jnum == 0:
            dataset_k_num = dataset_k_num.drop(['PRI_jet_all_pt'], axis=1)
        datasets[class_name] = dataset_k_num

    # Drop phi cols?
    for dataset in datasets:
        for column in datasets[dataset].columns:
            if 'phi' in column and 'DER' not in column:
                datasets[dataset] = datasets[dataset].drop([column], axis=1)
        datasets[dataset].fillna(data.DER_mass_MMC.mean(), inplace=True)

    for key in datasets:
        y = datasets[key].Label
        w = datasets[key].KaggleWeight
        X = datasets[key].drop(['Label', 'KaggleWeight'], axis=1)

        X.to_hdf("scripts/final_experiments/data/{0}_test_{1}.hdf".format(name_prefix, key), "X", mode='a',
                 complib='zlib', complevel=9)
        y.to_hdf("scripts/final_experiments/data/{0}_test_{1}.hdf".format(name_prefix, key), "y", mode='a',
                 complib='zlib', complevel=9)
        w.to_hdf("scripts/final_experiments/data/{0}_test_{1}.hdf".format(name_prefix, key), "w", mode='a',
                 complib='zlib', complevel=9)