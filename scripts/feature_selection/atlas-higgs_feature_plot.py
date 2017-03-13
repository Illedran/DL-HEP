import pandas as pd
from scripts.utils import show_graph
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from collections import OrderedDict
# No import errors


atlas_higgs = pd.read_csv("data/atlas-higgs-challenge-2014-vs2.csv")
atlas_higgs = atlas_higgs.drop(['EventId', 'KaggleWeight', 'KaggleSet'], axis=1)

# This dict is here in order to limit the axis in the histogram in order to have meaningful figures
axis_size = OrderedDict([
    ('DER_mass_MMC', ((0, 250), (0, 0.025))),
    ('DER_mass_transverse_met_lep', ((0, 150), (0, 0.030))),
    ('DER_mass_vis', ((0, 200), (0, 0.0275))),
    ('DER_pt_h', ((0, 400), (0, 0.03))),
    ('DER_deltaeta_jet_jet', ((0, 9), (0, 0.45))),
    ('DER_mass_jet_jet', ((0, 2000), (0, 0.005))),
    ('DER_prodeta_jet_jet', ((-15, 10), (0, 0.5))),
    ('DER_deltar_tau_lep', ((0, 5), (0, 2.5))),
    ('DER_pt_tot', ((0, 100), (0, 0.14))),
    ('DER_sum_pt', ((0, 600), (0, 0.025))),
    ('DER_pt_ratio_lep_tau', ((0, 10), (0, 1.0))),
    ('DER_met_phi_centrality', ((-1.5, 1.5), (0, 8))),
    ('DER_lep_eta_centrality', ((0,1), (0, 10))),
    ('PRI_tau_pt', ((0, 140), (0, 0.04))),
    ('PRI_tau_eta', ((-3, 3), (0, 0.4))),
    ('PRI_tau_phi', ((-np.pi, np.pi), (0, 0.2))),
    ('PRI_lep_pt', ((0, 150), (0, 0.045))),
    ('PRI_lep_eta', ((-3, 3), (0, 0.4))),
    ('PRI_lep_phi', ((-np.pi, np.pi), (0, 0.2))),
    ('PRI_met', ((0, 200), (0, 0.05))),
    ('PRI_met_phi', ((-np.pi, np.pi), (0, 0.2))),
    ('PRI_met_sumet', ((0, 800), (0, 0.009))),
    ('PRI_jet_leading_pt', ((0, 250), (0, 0.03))),
    ('PRI_jet_leading_eta', ((-5, 5), (0, 0.25))),
    ('PRI_jet_leading_phi', ((-np.pi, np.pi), (0, 0.2))),
    ('PRI_jet_subleading_pt', ((0, 200), (0, 0.05))),
    ('PRI_jet_subleading_eta', ((-5, 5), (0, 0.25))),
    ('PRI_jet_subleading_phi', ((-np.pi, np.pi), (0, 0.2))),
    ('PRI_jet_all_pt', ((0, 500), (0, 0.04))),
])

bins = {col: np.histogram(atlas_higgs[col].dropna(), bins='auto')[1] for col in atlas_higgs.columns}
for jnum in atlas_higgs.PRI_jet_num.unique():
    # Drop PRI_jet_num column
    jnum_class = atlas_higgs[atlas_higgs.PRI_jet_num == jnum].drop(['PRI_jet_num'], axis=1)
    # Drop columns that are ALL NaN
    jnum_class = jnum_class.drop(jnum_class.columns[jnum_class.isnull().sum() == len(jnum_class)], axis=1)
    # Additionally, for class with 0 jets, drop PRI_jet_all
    if jnum == 0:
        jnum_class = jnum_class.drop(['PRI_jet_all_pt'], axis=1)
    labels = jnum_class.Label
    jnum_class = jnum_class.drop(['Label'], axis=1)
    fig = plt.figure(dpi=200)
    num_rows = 6
    num_cols = 5
    gridspec.GridSpec(num_rows, num_cols)
    for idx, col in enumerate(axis_size.keys()):
        if col in jnum_class.columns:
            ax = plt.subplot2grid((num_rows, num_cols), (idx // 5, idx % 5))
            b = jnum_class[labels == 0].dropna()
            s = jnum_class[labels == 1].dropna()
            back = plt.hist(b[col], bins=bins[col], weights=b['Weight'], color='#6DB6FF', alpha=0.5, histtype='stepfilled', label='Background', normed=1)
            sig = plt.hist(s[col], bins=bins[col], weights=s['Weight'], color='#920000', alpha=0.5, histtype='stepfilled', label='Signal', normed=1)
            if col in axis_size:
                # print(col, axis_size[col][0], axis_size[col][1])
                ax.set_xlim(axis_size[col][0])
                ax.set_ylim(axis_size[col][1])
            ax.tick_params(axis='x', which='minor', bottom='on', direction='out')
            ax.set_title(r'\texttt{%s}' % col.replace('_', '\_'))

    fig.set_size_inches(15, 10)
    fig.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9, bottom=0.12, top=0.88)
    fig.suptitle(r'\texttt{PRI\_jet\_num} $= %d$' % jnum, fontsize=24)
    plt.legend(bbox_to_anchor=(0.85, 0.199), bbox_transform=plt.gcf().transFigure)
    show_graph("histo_{}".format(jnum), save_to="scripts/feature_selection/plots", format='png', dpi=fig.dpi)