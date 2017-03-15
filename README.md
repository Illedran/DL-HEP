# DL-HEP

Note: all python scripts should be run from the root directory of this repository.

In order to start training the model, the data must first be fetched and put in the `data` directory. 
Afterwards, generate the subsets for training and test by running:
`python data/preprocess_data.py`
This script will also split each dataset based on their number of the `PRI_jet_num` feature.

Inside `scripts/models` the following scripts are available:
* `autoencoder_tf.py` and `autoencoder_keras.py`: Instances an autoencoder using the respective
deep learning framework. Note that the TensorFlow version was used in this work and this is reflected in the train scripts.
* `variational_autoencoder_tf.py` and `variational_autoencoder_keras.py`. Instances a variational autoencoder.
* `gan.py`. Contains code for instancing the generator and discriminator models of a GAN.
These are actually combined with the `keras_adversarial` python package that can be found here on GitHub (it's also a git submodule in this repository).

These models are instanced in their respective training scripts in:
* `models/autoencoder`
* `models/variational_autoencoder`
* `models/gan`

Additional code is available in the `scripts/feature_selection` to plot distributions for each feature.