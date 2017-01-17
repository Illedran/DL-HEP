import pandas as pd
import os

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
    plt.hist(atlas_higgs[col][labels == 0].dropna(), bins=20, color='#D55E00', alpha=0.6, normed=True)
    plt.hist(atlas_higgs[col][labels == 1].dropna(), bins=20, color='#0072B2', alpha=0.6, normed=True)
    ax.set_title(col)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
