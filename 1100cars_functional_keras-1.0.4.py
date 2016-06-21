import os
import sys
import pickle
import zipfile

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Input
from keras.models import Sequential,Model

from autoencoder_layers_keras_1_0 import DependentDense, Deconvolution2D, DePool2D
from helpers import tile_raster_images, show_image, keras2rgb


def load_cars(split=0.8, python_version=3):
    # Vehicle images are courtecy of German Aerospace Center (DLR)
    # Remote Sensing Technology Institute, Photogrammetry and Image Analysis
    # http://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5431/9230_read-42467/
    if not os.path.exists('./data/cars.pkl'):
        print('Extracting cars dataset')
        with zipfile.ZipFile('./data/cars.pkl.zip', "r") as z:
            z.extractall("./data/")
    
    # if we're using python 3
    if python_version == 3:
        with open('./data/cars.pkl', 'rb') as ff:
            (X_data, y_data) = pickle.load(ff)
    else:
        with open('./data/cars_py2.pkl', 'rb') as ff:
            (X_data, y_data) = pickle.load(ff)
    
    X_data = X_data.reshape(X_data.shape[0], 3, 32, 32)
    l = int(split * X_data.shape[0])
    X_train = X_data[:l]
    X_test = X_data[l:]

    return X_train, X_test


def build_model_functional(nb_pool=2, nb_conv=3):
    channels = 3
    C_1 = 64
    C_2 = 32
    C_3 = 16
    code_length=100
    
    input_layer = Input(name='input_layer', 
                        shape=(channels,32,32))
    
    c1 = Convolution2D(C_1, nb_conv, nb_conv, 
                      border_mode='same',
                      name='c1',
                      activation='tanh')(input_layer)
    mp1 = MaxPooling2D(pool_size=(nb_pool, nb_pool),
                      name='mp1')(c1)
    c2 = Convolution2D(C_2, nb_conv, nb_conv, 
                       border_mode='same',
                       name='c2',
                       activation='tanh')(mp1)
    mp2 = MaxPooling2D(pool_size=(nb_pool, nb_pool),
                       name='mp2')(c2)
    c3 = Convolution2D(C_3, nb_conv, nb_conv, 
                       border_mode='same',
                       name='c3',
                       activation='tanh')(mp2)
    mp3 = MaxPooling2D(pool_size=(nb_pool, nb_pool),
                       name='mp3')(c3)
    flat3 = Flatten(name='flat3')(mp3)
    code = Dense(output_dim=code_length, 
              name='code',
              activation='tanh')(flat3)
    
    code_prime = DependentDense(output_dim=C_3*4*4, 
                        master_layer=code,
                        name='code_prime')(code)
    code_reshaped = Reshape(target_shape=(C_3, 4, 4),
                         name='code_reshaped')(code_prime)
    dep3 = DePool2D(master_layer=mp3, 
                    size=(nb_pool, nb_pool),
                    name='dep3')(code_reshaped)
    dec3 = Deconvolution2D(master_layer=c3, 
                           #nb_out_channels=C_2, 
                           border_mode='same',
                           name='dec3',
                           activation='tanh')(dep3)
    dep2 = DePool2D(master_layer=mp2, 
                    size=(nb_pool, nb_pool),
                    name='dep2')(dec3)
    dec2 = Deconvolution2D(master_layer=c2, 
                           #nb_out_channels=C_1, 
                           border_mode='same',
                           name='dec2',
                           activation='tanh')(dep2)
    dep1 = DePool2D(master_layer=mp1, 
                    size=(nb_pool, nb_pool),
                    name='dep1')(dec2)
    dec1 = Deconvolution2D(master_layer=c1, 
                           #nb_out_channels=3, 
                           border_mode='same',
                           name='dec1',
                           activation='tanh')(dep1)          
    
    model = Model(input_layer,dec1)
    model.compile('adam', loss='mean_squared_error')
    #model.compile('rmsprop', loss='mean_squared_error')

    return model

if __name__ == '__main__':
    X_train, X_test = load_cars(python_version=sys.version_info.major)
    model = build_model_functional()
    if not False:
        model.summary()
        model.fit(X_train, X_train, nb_epoch=100, batch_size=64,
                  validation_split=0.2,
                  callbacks=[EarlyStopping(patience=12)])
        model.save_weights('./cars.neuro', overwrite=True)
    else:
        model.load_weights('./cars.neuro')

    l = model.predict(X_test[:25, ...])
    representations = np.clip(l, 0, 1)

    _r = tile_raster_images(
            X=keras2rgb(representations),
            img_shape=(32, 32, 3), tile_shape=(5, 5),
            tile_spacing=(1, 1))

    _o = tile_raster_images(
            X=keras2rgb(X_test),
            img_shape=(32, 32, 3), tile_shape=(5, 5),
            tile_spacing=(1, 1))

    show_image([(_o, 'Source'), (_r, 'Representations')])
