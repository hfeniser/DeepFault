import sys
sys.path.append('/home/hfeniser/DeepFault/')

import numpy as np
from keras.models import model_from_json
from keras import backend as K
from utils import load_MNIST, get_trainable_layers
from utils import filter_val_set
from test_nn import test_model
import matplotlib.pyplot as plt

'''
def calculate_final_relevances(trainable_layers, selected_class, X_val, model,
                              correct_classifications, layer_outs):
    final_relevances = []
    weights = model.layers[-1].get_weights()
    for i in range(len(X_val)):
        if not i in correct_classifications:
            final_relevances.append(0)
            continue

        louts = layer_outs[trainable_layers[-1]][0][i]
        final_relevance = 0
        for l,w in zip(louts,weights[0]):
            final_relevance += l*w[selected_class]

        final_relevances.append(final_relevance)

    return final_relevances
'''



def calculate_hidden_relevances(relevances_, model, layer_outs, trainable_layers,
                                correct_classifications,
                                selected_class, alpha=1, beta=0):

    relevant_neurons = []
    for test_idx in correct_classifications[:10]:
        weights = model.layers[-1].get_weights()
        print(test_idx)

        louts = layer_outs[trainable_layers[-1]][0][test_idx]
        final_relevance = 0
        for l,w in zip(louts,weights[0]):
            final_relevance += l*w[selected_class]

        # refresh relevances for each test input
        relevances_ = []

        # input layer relevance
        relevances_.append(np.zeros(model.layers[0].output_shape[1]))
        for tl in trainable_layers:
            relevances_.append(np.zeros(model.layers[tl].output_shape[1]))
        # output layer relevance
        relevances_.append(np.zeros(model.layers[-1].output_shape[1]))

        # relevances_[-1] = np.zeros(model.layers[-1].output_shape[1])
        relevances_[-1][selected_class] = final_relevance

        for i in range(2,len(relevances_))[::-1]:
            weights = model.layers[i].get_weights()[0]
            for j in range(model.layers[i].output_shape[1]):
                sum_rel_pos = 0
                sum_rel_neg = 0
                for k in range(model.layers[i-1].output_shape[1]):
                    if weights[k][j] > 0:
                        sum_rel_pos += layer_outs[i-1][0][test_idx][k] * weights[k][j]
                    else:
                        sum_rel_neg += layer_outs[i-1][0][test_idx][k] * weights[k][j]


                for k in range(model.layers[i-1].output_shape[1]):
                    if weights[k][j] > 0:
                        relevances_[i-1][k] += alpha * (float(layer_outs[i-1][0][test_idx][k] * \
                                                    weights[k][j]) / sum_rel_pos) * \
                                                    relevances_[i][j]
                    else:
                        relevances_[i-1][k] -= beta * (float(layer_outs[i-1][0][test_idx][k] * \
                                                    weights[k][j]) / sum_rel_neg) * \
                                                    relevances_[i][j]


        lower_bound = 0
        upper_bound = 1
        weights = model.layers[1].get_weights()[0]
        for j in range(model.layers[1].output_shape[1]):
            sum_rel = 0
            for k in range(model.layers[0].output_shape[1]):
                sum_rel += layer_outs[0][0][test_idx][k] * weights[k][j]
                if weights[k][j] > 0:
                    sum_rel -= lower_bound * weights[k][j]
                else:
                    sum_rel -= upper_bound * weights[k][j]

            for k in range(model.layers[0].output_shape[1]):
                if weights[k][j] > 0:
                    relevances_[0][k] += (float((layer_outs[0][0][test_idx][k] * \
                                               weights[k][j] - lower_bound * \
                                               weights[k][j])) / sum_rel) * \
                                            relevances_[1][j]
                else:
                    relevances_[0][k] += (float((layer_outs[0][0][test_idx][k] * \
                                               weights[k][j] - upper_bound * \
                                               weights[k][j])) / sum_rel) * \
                                           relevances_[1][j]

        relevant_neurons.append(relevances_[1].argsort()[-5:])
        plt.imshow(relevances_[0].reshape((28,28)), cmap='hot', interpolation='nearest')
        plt.savefig('figures/heat_' + str(test_idx) + '.png')
    return relevant_neurons


selected_class = 0
model_name = 'mnist_test_model'
json_file = open(model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights(model_name + '.h5')

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

trainable_layers = get_trainable_layers(model)

relevances_ = []

# input layer relevance
relevances_.append(np.zeros(model.layers[0].output_shape[1]))
for tl in trainable_layers:
    relevances_.append(np.zeros(model.layers[tl].output_shape[1]))
# output layer relevance
relevances_.append(np.zeros(model.layers[-1].output_shape[1]))


X_train, y_train, X_test, y_test = load_MNIST()

X_val, Y_val = filter_val_set(selected_class, X_test, y_test)
correct_classifications, misclassifications, layer_outs, _ =\
                test_model(model, X_val, Y_val)


relevant_neurons = calculate_hidden_relevances(relevances_, model, layer_outs,
                                          trainable_layers,
                                          correct_classifications,
                                          selected_class, alpha=4, beta=3)



print(relevant_neurons)

loutto = []
for test_idx in correct_classifications[:20]:
    loutto.append(layer_outs[1][0][test_idx][relevant_neurons[0]])


print(loutto)
