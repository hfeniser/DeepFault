from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
import sys

import matplotlib.pyplot as plt
import numpy as np

def load_data(one_hot=True):
    """
    Load MNIST data
    :param one_hot:
    :return:
    """
    #Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Preprocess dataset
    #Normalization and reshaping of input.
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        #For output, it is important to change number to one-hot vector.
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test


def load_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into model
    model.load_weights(model_name + '.h5')

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def get_layer_outs(model, class_specific_test_set):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    # Testing
    layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

    return layer_outs


def get_python_version():
    if (sys.version_info > (3, 0)):
        # Python 3 code in this block
        return 3
    else:
        # Python 2 code in this block
        return 2


def show_image(vector):
    img = vector
    plt.imshow(img)
    plt.show()