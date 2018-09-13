from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
import sys
from sklearn.metrics import classification_report, confusion_matrix
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import h5py
from datetime import datetime
from os import path, makedirs
import traceback


def load_MNIST():
    path = "/scratch/sg778/DeepEntrust/tutorial/datasets/mnist.npz"
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_data(one_hot=True):
    """
    Load MNIST data
    :param one_hot:
    :return:
    """
    #Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # (X_train, y_train), (X_test, y_test) = load_MNIST()

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
        Y_train = np_utils.to_categorical(y_train, num_classes=10)
        Y_test = np_utils.to_categorical(y_test, num_classes=10)

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

    print("Model structure loaded from ", model_name)
    return model


def get_layer_outs_old(model, class_specific_test_set):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    # Testing
    layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

    return layer_outs


# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
def get_layer_outs(model, test_input):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layer_outs = [func([test_input]) for func in functors]
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


def calculate_prediction_metrics(Y_test, Y_pred, score):
    """
    Calculate classification report and confusion matrix
    :param Y_test:
    :param Y_pred:
    :param score:
    :return:
    """
    #Find test and prediction classes
    Y_test_class = np.argmax(Y_test, axis=1)
    Y_pred_class = np.argmax(Y_pred, axis=1)

    classifications = np.absolute(Y_test_class - Y_pred_class)

    correct_classifications = []
    incorrect_classifications = []
    for i in range(1, len(classifications)):
        if (classifications[i] == 0):
            correct_classifications.append(i)
        else:
            incorrect_classifications.append(i)


    # Accuracy of the predicted values
    print(classification_report(Y_test_class, Y_pred_class))
    print(confusion_matrix(Y_test_class, Y_pred_class))

    acc = sum([np.argmax(Y_test[i]) == np.argmax(Y_pred[i]) for i in range(len(Y_test))]) / len(Y_test)
    v1 = ceil(acc*10000)/10000
    v2 = ceil(score[1]*10000)/10000
    correct_accuracy_calculation =  v1 == v2
    try:
        if not correct_accuracy_calculation:
            raise Exception("Accuracy results don't match to score")
    except Exception as error:
        print("Caught this error: " + repr(error))


def get_dummy_dominants(model, dominants):
    import random
    # dominant = {x: random.sample(range(model.layers[x].output_shape[1]), 2) for x in range(1, len(model.layers))}
    dominant = {x: random.sample(range(0, 10), len(dominants[x])) for x in range(1, len(dominants)+1)}
    return dominant


def save_perturbed_test(x_perturbed, y_perturbed, filename):
    # save X
    with h5py.File(filename + '_perturbations_x.h5', 'w') as hf:
        hf.create_dataset("x_perturbed", data=x_perturbed)

    #save Y
    with h5py.File(filename + '_perturbations_y.h5', 'w') as hf:
        hf.create_dataset("y_perturbed", data=y_perturbed)

    return


def load_perturbed_test(filename):
    # read X
    with h5py.File(filename + '_perturbations_x.h5', 'r') as hf:
        x_perturbed = hf["x_perturbed"][:]

    # read Y
    with h5py.File(filename + '_perturbations_y.h5', 'r') as hf:
        y_perturbed = hf["y_perturbed"][:]

    return x_perturbed, y_perturbed


def save_perturbed_test_groups(x_perturbed, y_perturbed, filename, group_index):
    # save X
    filename = filename + '_perturbations.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        group.create_dataset("x_perturbed", data=x_perturbed)
        group.create_dataset("y_perturbed", data=y_perturbed)

    print("Classifications saved in ", filename)

    return


def load_perturbed_test_groups(filename, group_index):
    with h5py.File(filename + '_perturbations.h5', 'r') as hf:
        group = hf.get('group' + str(group_index))
        x_perturbed = group.get('x_perturbed').value
        y_perturbed = group.get('y_perturbed').value

        return x_perturbed, y_perturbed


def create_experiment_dir(experiment_path, model_name):
    # define experiment name
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    experiment_name = path.join(experiment_path, model_name + "_" + timestamp)

    if not path.exists(experiment_path):
        makedirs(experiment_path)

    if not path.exists(path.join(experiment_path, experiment_name)):
        makedirs(path.join(experiment_path, experiment_name))

    return experiment_name, timestamp


def save_classifications(correct_classifications, misclassifications, filename, group_index):
    filename = filename + '_classifications.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        group.create_dataset("correct_classifications", data=correct_classifications)
        group.create_dataset("misclassifications", data=misclassifications)

    print("Classifications saved in ", filename)
    return


def load_classifications(filename, group_index):
    filename = filename + '_classifications.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            correct_classifications = group.get('correct_classifications').value
            misclassifications = group.get('misclassifications').value

            print("Classifications loaded from ", filename)
            return correct_classifications, misclassifications
    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)


def save_layer_outs(layer_outs, filename, group_index):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_"+str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def load_layer_outs(filename, group_index):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                layer_outs.append(group.get('layer_outs_'+str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Layer outs loaded from ", filename)
        return layer_outs


def save_dominant_neurons(dominant_neurons, filename, group_index):
    filename = filename + '_dominant_neurons.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        for i in range(len(dominant_neurons)):
            group.create_dataset("dominant_neurons"+str(i), data=dominant_neurons[i])

    print("Dominant neurons saved in ", filename)
    return


def load_dominant_neurons(filename, group_index):
    filename = filename + '_dominant_neurons.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            dominant_neurons = []
            while True:
                dominant_neurons.append(group.get('dominant_neurons' + str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Dominant neurons  loaded from ", filename)
        return dominant_neurons
