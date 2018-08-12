from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from collections import defaultdict
import numpy as np
import argparse
from utils import load_model, load_data, get_layer_outs

parser = argparse.ArgumentParser(description='An MNIST Network\'s Neuron Analysis')
parser.add_argument("-ev", "--error_class", type=int, help='Label of the predicted class by NN.')
parser.add_argument('-tc', "--true_class",  type=int, help='Label of the true (expected) class.')
parser.add_argument('-met', "--metric",  type=str, help='Which metric to identify dominant neurons')
args = parser.parse_args()

#Provide a seed for reproducability
np.random.seed(7)

X_train, Y_train, X_test, Y_test = load_data()
model = load_model('simple_mnist_fnn')

error_class_to_input= []
predictions = model.predict(X_test)

if args.metric == 'intersection':
    idx = 1
    for pred, crrct in zip(predictions, Y_test):
        predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
        true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]

        if args.true_class == None:
            #if user does not specify the true class (it is optional),  we consider all predictions that are equal to "given error class" and not correct
            condition = predicted_class == args.error_class and predicted_class != true_class
        else:
            #if user specifies a true class we consider predictions that are equal to "given error class" and expected to be "given true class"
            condition = predicted_class == args.error_class and true_class == args.true_class

        #This condition gives us the indices of the inputs that are correctly
        #clasified and their label are as specified by the user.
        #condition = predicted_class == args.error_class and predicted_class == true_class

        if condition:
            error_class_to_input.append(idx)

        idx += 1

    class_specific_test_set = np.ndarray(shape=(len(error_class_to_input),1,28,28))

    cnt = 0
    for test_input in error_class_to_input:
        class_specific_test_set[cnt] = test_input
        cnt += 1

    layer_outs = get_layer_outs(model, class_specific_test_set)

    activated_list = []
    for l_out in layer_outs[1:]:
        activated = list(range(128))
        for l in l_out[0]:
            activated = np.intersect1d(activated, np.where(l > 0)) #list(set(activated) & set(list(np.where(l > 0)))) #list(numpy.where(l > 0)
        print activated
        activated_list.append(activated)

else:
    predictions = model.predict(X_test)
    test_result_list = []
    for pred, crrct in zip(predictions, Y_test):
        predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
        true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]
        if predicted_class == true_class:
            test_result_list.append(1)
        else:
            test_result_list.append(0)

    layer_outs = get_layer_outs(model, X_test)

    scores = []
    num_cf = []
    num_uf = []
    num_cs = []
    num_us = []
    for l_out in layer_outs[1:]:
        num_cf.append(np.zeros(len(l_out[0][0])))
        num_uf.append(np.zeros(len(l_out[0][0])))
        num_cs.append(np.zeros(len(l_out[0][0])))
        num_us.append(np.zeros(len(l_out[0][0])))
        scores.append(np.zeros(len(l_out[0][0])))

    layer_idx = 0
    for l_out in layer_outs[1:]:
        all_neuron_idx = list(range(len(l_out[0][0])))
        test_idx = 0
        print layer_idx
        for l in l_out[0]:
            covered_idx   = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx)-set(covered_idx))
            #test_idx = list(l_out[0]).index(l)

            if test_result_list[test_idx] == 1:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            else:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1
        layer_idx += 1

    #if args.metric == 'tarantula':
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if args.metric == 'tarantula':
                scores[i][j] = float(float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j])) / (float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j]) + float(num_cs[i][j]) / (num_cs[i][j] + num_us[i][j]))
            elif args.metric == 'ochiai':
                scores[i][j] = float(num_cf[i][j]) / ((num_cf[i][j] + num_uf[i][j]) * (num_cf[i][j] + num_cs[i][j])) **(.5)

    print scores

#Construct model
#model = Sequential()
#model.add(Flatten(input_shape=(1,28,28)))

#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))

#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

#model.fit(X_train, Y_train,
#          batch_size=32, nb_epoch=10, verbose=1)


#score = model.evaluate(X_test, Y_test, verbose=0)
#print score

# serialize model to JSON
#model_json = model.to_json()
#with open("simple_mnist_fnn.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("simple_mnist_fnn.h5")
#print("Model saved to disk")
