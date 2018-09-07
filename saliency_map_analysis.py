from keras.models import Sequential
from keras import backend as K
from collections import defaultdict
import numpy as np
import argparse
import tensorflow as tf
from utils import load_model, load_data, get_layer_outs

class JacobianSaliency(object):

    def __init__(self, model, layer_index = 0):
        # Define the function to compute the gradient
        self.compute_gradients = []
        input_tensors = [model.layers[layer_index].output]
        for i in range(model.outputs[0].shape[1]):
            gradients = model.optimizer.get_gradients(model.output[0][i],model.layers[layer_index].output)
            self.compute_gradients.append(K.function(inputs = input_tensors,
                                                     outputs = gradients))

    def get_saliency_map(self, layer_out, model, target_label, sal_type = 'increase'):
        gradient_repo = []
        for k in range(model.outputs[0].shape[1]):
            gradient_repo.append(self.compute_gradients[k]([layer_out])[0])

        saliency_map = []
        for i in range(len(layer_out)):
            saliency_map.append(np.zeros(len(layer_out[0])))

        for i in range(len(layer_out)):
            for j in range(len(layer_out[0])):
                other_label_effect  = 0
                target_label_effect = 0
                for k in range(len(gradient_repo)):
                    gradients = gradient_repo[k]
                    if k == target_label:
                        target_label_effect = gradients[i][j]
                    else:
                        other_label_effect += gradients[i][j]

                if sal_type == 'increase':
                    if target_label_effect < 0 or other_label_effect > 0:
                        saliency_map[i][j] = 0
                    else:
                        saliency_map[i][j] = target_label_effect * abs(other_label_effect)
                else:
                    if target_label_effect > 0 or other_label_effect < 0:
                        saliency_map[i][j] = 0
                    else:
                        saliency_map[i][j] = abs(target_label_effect) * other_label_effect

        return saliency_map


def saliency_map_analysis(correct_classification_idx, misclassification_idx, layer_outs, model, predictions):

    scores = []
    dominant_neuron_idx = []
    for l_out in layer_outs:
        scores.append(np.zeros(len(l_out[0][0])))
        dominant_neuron_idx.append([])

    for layer_idx in range(1, len(layer_outs[1:])):
        test_idx = 0
        print(layer_idx)
        for l in layer_outs[layer_idx][0]:
            if test_idx not in misclassification_idx:
                test_idx +=1
                continue

            saliency = JacobianSaliency(model, layer_idx)
            saliency_map = saliency.get_saliency_map(np.expand_dims(l, axis=0), model, predictions[test_idx], 'increase')[0]

            max_sal = max(saliency_map)
            max_ind = list(saliency_map).index(max_sal)
            scores[layer_idx][max_ind] += 1
            test_idx += 1

    print scores
    exit()
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] > 200: #score threshold, what's the correct value? maybe can be found through experiments?
                dominant_neuron_idx[i].append(j)
