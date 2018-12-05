from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation
from train_mnist_nn import __save_trained_model
from utils import load_model, load_classifications, save_classifications
from utils import get_trainable_layers, get_layer_outs, save_layer_outs
from test_nn import test_model

import numpy as np
import matplotlib.pyplot as plt



num_hidden = 3
num_neuron = 4
model_name = 'syn_test_model_' + str(num_hidden) + "_" + str(num_neuron)
experiment_path = 'experiment_results'
group_index = 1

cov = [[2,0], [0,2]]

def shuffle_arrays(arr1, arr2):
    perm = np.random.permutation(len(arr1))

    shuffled_arr1 = arr1[perm]
    shuffled_arr2 = arr2[perm]

    return shuffled_arr1, shuffled_arr2

def synthesize_data(data_size):

    a_1 = np.random.multivariate_normal([1,1], cov, data_size) # mu = (1,1)
    a_2 = np.random.multivariate_normal([1,9], cov, data_size) # mu = (1,9)
    a_3 = np.random.multivariate_normal([5,5], cov, data_size) # mu = (5,5)
    a_4 = np.random.multivariate_normal([9,1], cov, data_size) # mu = (9,1)
    a_5 = np.random.multivariate_normal([9,9], cov, data_size) # mu = (9,9)

    b_1 = np.random.multivariate_normal([1,5], cov, data_size) # mu = (1,5)
    b_2 = np.random.multivariate_normal([5,1], cov, data_size) # mu = (5,1)
    b_3 = np.random.multivariate_normal([5,9], cov, data_size) # mu = (5,9)
    b_4 = np.random.multivariate_normal([9,5], cov, data_size) # mu = (9,5)

    X_train = np.concatenate((a_1, a_2, a_3, a_4, a_5, b_1, b_2, b_3, b_4), axis=0)

    class_a = np.array([[0,1]])
    class_b = np.array([[1,0]])

    y_train = np.concatenate((np.repeat(class_a, 5*data_size, axis=0),
                              np.repeat(class_b, 4*data_size, axis=0)), axis=0)

    X_train, y_train = shuffle_arrays(X_train, y_train)
    return X_train, y_train



X_train, y_train = synthesize_data(18000)
X_test, y_test   = synthesize_data(1800)

try:
    model = load_model('neural_networks/' + model_name)
except:
    model = Sequential()
    model.add(Dense(4,input_shape=(2,)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(4, use_bias=False))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(4, use_bias=False))
    model.add(LeakyReLU(alpha=.01))

    model.add(Dense(2, activation='softmax', use_bias=False))

    model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)


    __save_trained_model(model, num_hidden, num_neuron, 'syn_test_model')


trainable_layers = get_trainable_layers(model)

a1_test = np.random.multivariate_normal([1,1], cov, 100)
a2_test = np.random.multivariate_normal([1,9], cov, 100)
b1_test = np.random.multivariate_normal([1,5], cov, 100)

class_a = np.array([[0,1]])
y = np.repeat(class_a, 100, axis=0)

correct_classifications, misclassifications, layer_outs, predictions = \
test_model(model, a2_test, y)

hout = layer_outs[trainable_layers[-1]][0]
print(np.mean(hout[correct_classifications], axis=0))
#print(np.mean(hout[misclassifications], axis=0))

exit()

filename = experiment_path + '/' + model_name
try:
    correct_classifications, misclassifications = load_classifications(filename, group_index)
    layer_outs = load_layer_outs(filename, group_index)
except:
    correct_classifications, misclassifications, layer_outs, predictions = test_model(model, X_val, Y_val)
    save_classifications(correct_classifications, misclassifications, filename, group_index)
    save_layer_outs(layer_outs, filename, group_index)
