import numpy as np
from keras import models
from keras.datasets import mnist
from keras.layers.core import Dense
from autoencoder_layers import DependentDense
from helpers import show_image, tile_raster_images, show_representations


def plot_weights(weights):
    w_c = weights.sum(axis=1)  # 50x1
    weights = weights / w_c.reshape((weights.shape[0], 1))
    IMG = tile_raster_images(
            X=weights,
            img_shape=(28, 28), tile_shape=(10, 5),
            tile_spacing=(1, 1))
    show_image(IMG)


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    return (X_train, y_train), (X_test, y_test)


def build_model(encoder_dim=50, bottleneck_dim=20):
    encoder = Dense(encoder_dim, input_dim=28 * 28, activation='tanh')
    bottleneck = Dense(bottleneck_dim, activation='tanh')
    bottleneck_2 = DependentDense(encoder_dim, bottleneck, activation='tanh')
    decoder = DependentDense(28 * 28, encoder, activation='tanh')
    model = models.Sequential()
    model.add(encoder)
    model.add(bottleneck)
    model.add(bottleneck_2)
    model.add(decoder)
    return model, encoder


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    model, encoder = build_model()
    if not False:
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(X_train, X_train, nb_epoch=200, batch_size=512, validation_split=0.2, )
        model.save_weights('./fcc.neuro', overwrite=True)
    else:
        model.load_weights('./fcc.neuro')
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    show_representations(model, X_test)
    plot_weights(encoder.get_weights()[0].T)
