import pylab
# print('*****')
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu0')
# print('*****')

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.objectives import binary_crossentropy
from keras.regularizers import l2
from theano.ifelse import ifelse
from theano.scalar import float32, float64

from dirt_track.klass_map import show_stats, ROAD_CLASS, DIRT_CLASS, show_for_dataset, HOUSE_CLASS, SNOW_CLASS, \
    TREE_CLASS, TOTAL_CLASSES, MODEL_NAME, GEO_NAME, L_CHANNEL_ONLY, MASKS, NAMES
from theano_test.helpers import path_pycharm_env

# path_pycharm_env()
import os

import theano

import theano.tensor as T
import matplotlib.pyplot  as plt
import pickle
import numpy as np
from keras import backend as K
import datetime

from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

model = Sequential()
model.add(Convolution2D(200, 2, 2, border_mode='same', input_shape=(1,20,20)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
d = Dense(1500, W_regularizer=l2(1e-3), activation='relu')
model.add(d)
model.add(Dropout(0.5))
model.add(Dense(1))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
c = theano.function([d.get_input(train=False)], d.get_output(train=False))
o = c(np.random.random((1,20000)).astype('float32'))
print(d.input_shape)
