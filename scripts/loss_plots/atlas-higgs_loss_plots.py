from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from scripts.utils import parse_dataset
from scripts.autoencoder import Autoencoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.utils import show_graph
import pickle

# Background: #6DB6FF
# Signal: #920000
for loss in ['mse', 'cosine_proximity']:
    for z in [1,2,4,8]:
        histories = []
        for jet_class in [0, 1, 2, 3]:
            X_train, y_train, _ = parse_dataset(cls='t-{}'.format(jet_class))
            X_train = X_train[y_train == 0.]

            X_test, y_test, w_test = parse_dataset(cls='b-{}'.format(jet_class))

            rows, cols = X_train.shape

            ae = Autoencoder(input_dimensions=cols, latent_dimensions=z, layers=[50, 20])
            ae.compile(loss=loss, optimizer='adam')

            imputer = Imputer()
            sscaler = StandardScaler()
            preproc = Pipeline([('imputer', imputer), ('scaler', sscaler)])

            preproc.fit(X_train)
            X_train = preproc.transform(X_train)
            X_test = preproc.transform(X_test)

            histories.append(ae.fit(X_train, X_train, nb_epoch=250, batch_size=32, validation_data=(X_test, X_test)))
            # 0: 74421 training, 40163 validation di cui 10114 segnale
            # 1: 49834 training, 30791 validation di cui 10924 segnale
            # 2: 24645 training, 20266 validation di cui 10267 segnale
            # 3: 15433 training, 8780 validation di cui 2720 segnale

            histories = pickle.load(open("scripts/loss_plots/histories_{0}_{1}.pickle".format(loss, z), 'rb'))
            fig = plt.figure(dpi=600)
            num_rows = 2
            num_cols = 2
            gridspec.GridSpec(num_rows, num_cols)
            for idx, hist in enumerate(histories):
                ax = plt.subplot2grid((num_rows, num_cols), (idx // num_rows, idx % num_cols))
                train = plt.plot(hist.epoch, hist.history['loss'], color='#6DB6FF', label='Training loss')
                validation = plt.plot(hist.epoch, hist.history['val_loss'], color='#920000', label='Test loss')
                ax.tick_params(axis='x', which='minor', bottom='on', direction='out')
                ax.set_xlim([0, 250])
                ax.set_ylim([0.0, 1.0001])
                ax.set_title("PRI_jet_num = {}".format(idx))

            fig.set_size_inches(16, 12)
            fig.subplots_adjust(wspace=0.1, hspace=0.13, left=0.1, right=0.9, bottom=0.08, top=0.92)
            fig.suptitle("latent_dimensions = {}".format(z), fontsize=32)
            plt.legend(bbox_to_anchor=(0.90625, 0.99), bbox_transform=plt.gcf().transFigure, fancybox=True, shadow=True)
            show_graph("loss_{0}_{1}".format(loss, z), save_to="scripts/loss_plots", format='svg', dpi=fig.dpi)

            # pickle.dump(histories, open("histories_{}.pickle".format(z), 'wb'))
