import os
import pickle
import zipfile

import numpy as np
from keras import models
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.models import Graph

from autoencoder_layers import DependentDense, Deconvolution2D, DePool2D
from helpers import tile_raster_images, show_image, keras2rgb


def load_cars(split=0.8):
    # Vehicle images are courtecy of German Aerospace Center (DLR)
    # Remote Sensing Technology Institute, Photogrammetry and Image Analysis
    # http://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5431/9230_read-42467/
    if not os._exists('./data/cars.pkl'):
        print('Extracting cars dataset')
        with zipfile.ZipFile('./data/cars.pkl.zip', "r") as z:
            z.extractall("./data/")

    with open('./data/cars.pkl', 'rb') as ff:
        (X_data, y_data) = pickle.load(ff)
    X_data = X_data.reshape(X_data.shape[0], 3, 32, 32)
    l = int(split * X_data.shape[0])
    X_train = X_data[:l]
    X_test = X_data[l:]

    return X_train, X_test


def build_model(nb_filters=32, nb_pool=2, nb_conv=3):
    C_1 = 64
    C_2 = 32
    C_3 = 16
    c = Convolution2D(C_1, nb_conv, nb_conv, border_mode='same', input_shape=(3, 32, 32))
    mp = MaxPooling2D(pool_size=(nb_pool, nb_pool))
    c2 = Convolution2D(C_2, nb_conv, nb_conv, border_mode='same', input_shape=(3, 32, 32))
    mp2 = MaxPooling2D(pool_size=(nb_pool, nb_pool))
    d = Dense(100)
    encoder = get_encoder(c, c2, d, mp, mp2)
    decoder = get_decoder(C_1, C_2, C_3, c, c2, d, mp, mp2, nb_pool)

    graph = Graph()
    graph.add_input(name='input', input_shape=(3, 32, 32))
    graph.add_node(encoder, name='encoder', input='input')
    graph.add_node(decoder, name='decoder', input='encoder')
    graph.add_output(name='autoencoder_feedback', input='decoder')
    graph.compile('rmsprop', {'autoencoder_feedback': 'mean_squared_error'})

    return graph


def get_decoder(C_1, C_2, C_3, c, c2, d, mp, mp2, nb_pool):
    decoder = models.Sequential()

    decoder.add(DependentDense(d.input_shape[1], d, input_shape=(d.output_shape[1],)))
    decoder.add(Activation('tanh'))
    decoder.add(Reshape((C_2, 8, 8)))
    # ====================================================
    # decoder.add(DePool2D(mp3, size=(nb_pool, nb_pool)))
    # decoder.add(Deconvolution2D(c3, nb_out_channels=C_2, border_mode='same'))
    # decoder.add(Activation('tanh'))
    # ====================================================
    decoder.add(DePool2D(mp2, size=(nb_pool, nb_pool)))
    decoder.add(Deconvolution2D(c2, nb_out_channels=C_1, border_mode='same'))
    decoder.add(Activation('tanh'))
    # ====================================================
    decoder.add(DePool2D(mp, size=(nb_pool, nb_pool)))
    decoder.add(Deconvolution2D(c, nb_out_channels=3, border_mode='same'))
    decoder.add(Activation('tanh'))
    # ====================================================
    return decoder


def get_encoder(c, c2, d, mp, mp2):
    encoder = models.Sequential()
    # ====================================================
    encoder.add(c)
    encoder.add(Activation('tanh'))
    encoder.add(mp)
    # ====================================================
    encoder.add(Dropout(0.25))
    # ====================================================
    encoder.add(c2)
    encoder.add(Activation('tanh'))
    encoder.add(mp2)
    # ====================================================
    # model.add(c3)
    # model.add(Activation('tanh'))
    # model.add(mp3)
    # ====================================================
    encoder.add(Dropout(0.25))
    # ====================================================
    encoder.add(Flatten())
    encoder.add(d)
    encoder.add(Activation('tanh'))
    return encoder


if __name__ == '__main__':
    X_train, X_test = load_cars()
    model = build_model()
    if not False:
        model.summary()
        model.fit({'input': X_train, 'autoencoder_feedback': X_train}, nb_epoch=100, batch_size=64,
                  validation_split=0.2,
                  callbacks=[EarlyStopping(patience=12)])
        model.save_weights('./cars.neuro', overwrite=True)
    else:
        model.load_weights('./cars.neuro')

    l = model.predict({'input': X_test[:25, ...]})
    representations = np.clip(l['autoencoder_feedback'], 0, 1)

    _r = tile_raster_images(
            X=keras2rgb(representations),
            img_shape=(32, 32, 3), tile_shape=(5, 5),
            tile_spacing=(1, 1))

    _o = tile_raster_images(
            X=keras2rgb(X_test),
            img_shape=(32, 32, 3), tile_shape=(5, 5),
            tile_spacing=(1, 1))

    show_image([(_o, 'Source'), (_r, 'Representations')])
