from keras.models import Sequential, Graph
from keras.layers.core import Dense, Flatten, Reshape, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.optimizers import SGD
from keras import models
from autoencoder_layers import DependentDense, Deconvolution2D, DePool2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from helpers import keras_arr2rgb_arr, make_mosaic

from keras.datasets import cifar10, mnist
from keras.utils import np_utils

nb_classes = 10
img_rows = 28
img_cols = 28

def build_model():
    nb_filters = 32
    cv_size = 3
    nb_pool = 2

    nb_mid_dense = 128
    nb_encode = 30

    mn_size = 12
    act = 'tanh'

    model_in = Sequential()

    # =========      ENCODER     ========================
    c1 = Convolution2D(nb_filters, cv_size, cv_size, border_mode='valid', input_shape=(1, img_rows, img_cols))
    model_in.add(c1)
    model_in.add(Activation(act))

    c2 = Convolution2D(nb_filters, cv_size, cv_size, border_mode='valid')
    model_in.add(c2)
    model_in.add(Activation(act))

    mp1 = MaxPooling2D(pool_size=(nb_pool, nb_pool))
    model_in.add(mp1)

    model_in.add(Flatten())

    d1 = Dense(nb_mid_dense)
    model_in.add(d1)
    model_in.add(Activation(act))

    print('model in')
    model_in.summary()

    graph = Graph()
    graph.add_input(name='input', input_shape=(1, img_rows, img_cols))
    graph.add_node(model_in, name='model_in', input='input')

    # =========      BOTTLENECK     ======================
    d2 = Dense(nb_encode)
    graph.add_node(d2, name = 'bottle_in', input='model_in')
    graph.add_node(Activation(act), name = 'act1', input='bottle_in')

    # =========      BOTTLENECK^-1   =====================
    graph.add_node(DependentDense(nb_mid_dense,d2), name = 'bottle_out', input = 'act1')
    graph.add_node(Activation(act), name = 'act2', input = 'bottle_out')

    # =========      DECODER     =========================
    model_out = models.Sequential()
    model_out.add(DependentDense(nb_filters * mn_size * mn_size, d1, input_shape=(nb_mid_dense,)))
    model_out.add(Activation(act))

    model_out.add(Reshape((nb_filters, mn_size, mn_size)))

    model_out.add(DePool2D(mp1, size=(nb_pool, nb_pool)))

    model_out.add(Deconvolution2D(c2, nb_out_channels = nb_filters, border_mode='same'))
    model_out.add(Activation(act))
    model_out.add(ZeroPadding2D(padding=(1, 1)))

    model_out.add(Deconvolution2D(c1, nb_out_channels =1, border_mode='same'))
    model_out.add(Activation(act))
    model_out.add(ZeroPadding2D(padding=(1, 1)))

    print('model out')
    model_out.summary()

    # =========      SOFTMAX_OUT2     =====================
    graph.add_node(model_out, name='model_out', input='act2')
    graph.add_output(name='output1', input='model_out')

    graph.add_node(Dense(nb_classes, activation='softmax', init='uniform'), name='dense1', input='model_in')

    graph.add_output(name='output2', input='dense1', merge_mode='sum')

    graph.summary()

    graph.compile('rmsprop', {'output1':'mse','output2':'categorical_crossentropy'})

    return graph

def train(model, nb_epoch = 10):
    batch_size = 32

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    model.fit({'input':X_train, 'output1':X_train, 'output2':Y_train},
                  batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle = True)

    print('')
    model.save_weights('ae_mnist.neuro', overwrite=True)

def test_model(model):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_test = X_test.astype('float32')

    X_test /= 255.0

    out = model.predict({'input':X_test})

    y_pred = out['output2']

    y = np_utils.probas_to_classes(y_pred)

    print('accuracy:', np_utils.accuracy(y, y_test))

    X_test = keras_arr2rgb_arr(X_test[:144])
    X_rec = keras_arr2rgb_arr(out['output1'])
    X_rec = X_rec[:144]

    for i in range(143):
        X_rec[i,:,:,:] = np.clip(X_rec[i,:,:,:], 0, 1)

    show1 = make_mosaic(X_test,12,12)
    show2 = make_mosaic(X_rec,12,12)

    plt.figure()
    plt.subplot(121)
    plt.imshow(show1); plt.title('Sample');
    plt.subplot(122)
    plt.imshow(show2); plt.title('Reconstruction');
    plt.show()

if __name__ == '__main__':

    model = build_model()
    train(model, 1)
    test_model(model)
