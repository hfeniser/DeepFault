from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from collections import defaultdict
from random import shuffle
import numpy as np
import argparse

def calculate_mustafa_cov(falsified_dict, num_pssbl_class):
    num_pssbl_shift = num_pssbl_class * (num_pssbl_class - 1)
    total_falsify = 0
    for k, v in falsified_dict.iteritems():
        total_falsify += len(v)

    return float(total_falsify) / num_pssbl_shift


def eval_model(model, filtered_ind=[], call='before'):

    fw = open('mutation_report', 'a')
    fw.write('******************\n')
    if call == 'before':
        fw.write('*     BEFORE     *\n')
    else:
        fw.write('*     AFTER      *\n')
    fw.write('******************\n\n')
    #Provide a seed for reproducability
    np.random.seed(7)

    #Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Preprocess dataset
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    predictions = model.predict(X_test)

    #Indices of inputs that will be filtered.
    ind_to_filter = []
    #A dictionary where keys are correct class labels and values are wrong
    #class assignments of falsified inputs
    falsified_dict = defaultdict(list)
    num_corrected = 0 #Number of corrected predictions after mutation
    num_falsified = 0 #Number of falsified predicitons after mutation
    cnt_glob = 0
    for pred, crrct in zip(predictions, Y_test):
        predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
        true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]
        if cnt_glob in filtered_ind:
            if true_class == predicted_class:
                num_corrected += 1
            cnt_glob += 1
            continue
        if true_class != predicted_class:
            num_falsified += 1
            falsified_dict[true_class].append(predicted_class)
            ind_to_filter.append(cnt_glob)

        cnt_glob += 1

    for k, v in falsified_dict.iteritems():
        falsified_dict[k] = list(set(v))

    fw.write('Number of wrong guesses: ' + str(num_falsified) + '/' +
             str(len(predictions) - len(filtered_ind)) + '\n')

    print 'Number of wrong guesses: ' + str(num_falsified) + '/' + str(len(predictions) - len(filtered_ind))

    if call == 'after':
        fw.write('Number of corrected guesses after mutation: ' +
             str(num_corrected) + '\n')

        num_pssbl_class = len(Y_test[0])
        fw.write('DeepMutation Coverage of filtered test inputs: ' +
                 str(float(len(falsified_dict)) / num_pssbl_class) + '\n')
        fw.write('Mustafa Coverage of filtered test inputs: ' +
                 str(calculate_mustafa_cov(falsified_dict, num_pssbl_class)) +
                '\n')

    fw.write('\n')
    fw.write('Dictionary showing true class to wrong class: \n')
    fw.write(str(falsified_dict))
    fw.write('\n\n\n')

    return ind_to_filter

#Coor is coordinate of a neuron which is a 2-tuple where the first element is
#layer number second number is neuron number
def mutation_schema(model, mu_type, coor=None, coor2=None):

    fw = open('mutation_report', 'w')
    fw.write('===DNN MUTATION REPORT===\n\n')
    fw.close()

    filtered_ind = eval_model(model)

    #Neuron effect blocking
    if mu_type == 'neb':
        #index 0 is hidden weights, index 1 is bias weights
        old_weights = model.layers[coor[0]].get_weights()
        old_weights[0][coor[1]] = [0] * len(old_weights[0][coor[1]])
        #Extra mutations
        old_weights[0][1] = [0] * len(old_weights[0][coor[1]])
        old_weights[0][2] = [0] * len(old_weights[0][coor[1]])

        model.layers[coor[0]].set_weights(old_weights)

    #Neuron activation inverse
    elif mu_type == 'nai':
        old_weights = model.layers[coor[0]-2].get_weights()
        for ow in old_weights[0]:
            ow[coor[1]] = -ow[coor[1]]

        old_weights[1][coor[1]] = -old_weights[1][coor[1]]
        model.layers[coor[0]-2].set_weights(old_weights)

    #Given neuron's previous layer weight shuffling
    elif mu_type == 'ws':
        old_weights = model.layers[coor[0]-2].get_weights()
        shuffle_list = []
        for ow in old_weights[0]:
            shuffle_list.append(ow[coor[1]])

        shuffle(shuffle_list)
        for ow, sl in zip(old_weights[0], shuffle_list):
            ow[coor[1]] = sl

        model.layers[coor[0]-2].set_weights(old_weights)

    #Neuron switch within a layer
    elif mu_type == 'ns':
        #get n1 weights
        n1_next_weights = model.layers[coor[0]].get_weights()
        n1_prev_weights = model.layers[coor[0]-2].get_weights()
        #get n2 weights
        n2_next_weights = model.layers[coor2[0]].get_weights()
        n2_prev_weights = model.layers[coor2[0]-2].get_weights()

        #n1next <-- n2next
        n1_next_weights[0][coor[1]] = n2_next_weights[0][coor2[1]]
        model.layers[coor[0]].set_weights(n1_next_weights)

        #n2next <-- n1next
        n2_next_weights[0][coor2[1]] = n1_next_weights[0][coor[1]]
        model.layers[coor[0]].set_weights(n2_next_weights)

        #n2prev <-- n1prev
        n1_cnnctd_prev_weights = []
        for n1pw in n1_prev_weights[0]:
            n1_cnnctd_prev_weights.append(n1pw[coor[1]])
        for n2pw, n1cpw in zip(n2_prev_weights, n1_cnnctd_prev_weights):
            n2pw[coor2[1]] = n1cpw
        model.layers[coor2[0]-2].set_weights(n2_prev_weights)

        #n1prev <-- n2prev
        n2_cnnctd_prev_weights = []
        for n2pw in n2_prev_weights[0]:
            n2_cnnctd_prev_weights.append(n2pw[coor2[1]])
        for n1pw, n2cpw in zip(n1_prev_weights, n2_cnnctd_prev_weights):
            n1pw[coor[1]] = n2cpw
        model.layers[coor[0]].set_weights(n1_prev_weights)

    if mu_type == 'gf':
        old_weights = model.layers[coor[0]].get_weights()
        mutated_weights = []
        for mw in old_weights[0][coor[1]]:
            mutated_weights.append(np.random.normal(mw, 1, 1))

        old_weights[0][coor[1]] =  mutated_weights
        model.layers[coor[0]].set_weights(old_weights)

    eval_model(model, filtered_ind, call='after')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keras DNN mutation testing tool')
    parser.add_argument("model", help='Name of saved model')
    parser.add_argument("mutation", help='Type of mutation.',
                        choices=['nai','neb', 'ws', 'ns', 'gf'])
    parser.add_argument('-l1', '--layer', type=int, help='Layer of the neuron \
                        to be mutated/swtiched.')
    parser.add_argument('-n1', '--neuron', type=int, help='Index of the neuron\
                        to be mutated/switched.')
    parser.add_argument('-l2', '--layer2', type=int, help='Layer of the neuron \
                        to be switched')
    parser.add_argument('-n2', '--neuron2', type=int, help='Index of the neuron\
                        to be switched.')

    args = parser.parse_args()

    json_file = open(args.model+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(args.model + ".h5")

    model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

    mutation_schema(model, args.mutation, (args.layer, args.neuron),
                    (args.layer2, args.neuron2))

