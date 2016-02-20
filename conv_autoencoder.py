import numpy as np
from keras import models
from keras.datasets import mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from autoencoder.autoencoder_layers import DependentDense, SumLayer, Deconvolution2D
from autoencoder.helpers import show_image, show_representations


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    return (X_train, y_train), (X_test, y_test)


def build_model(nb_filters=32, nb_pool=2, nb_conv=3):
    model = models.Sequential()
    c = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', input_shape=(1, 28, 28))
    model.add(c)
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    d =Dense(30)
    model.add(d)
    model.add(Activation('tanh'))
    model.add(DependentDense(nb_filters * 14 * 14, d))
    model.add(Reshape((nb_filters, 14, 14)))
    model.add(UpSampling2D(size=(nb_pool, nb_pool)))
    model.add(Activation('tanh'))
    model.add(Deconvolution2D(c, border_mode='same'))
    return model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    model = build_model()
    if False:
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.summary()
        model.fit(X_train, X_train, nb_epoch=12, batch_size=512, validation_split=0.2, )
        model.save_weights('./conv.neuro', overwrite=True)
    else:
        model.load_weights('./conv.neuro')
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    show_representations(model, X_test)
