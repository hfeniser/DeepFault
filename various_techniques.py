import numpy  as np
from keras.models import Sequential
from keras import backend as K
from collections import defaultdict
import argparse
import tensorflow as tf
from utils import load_model, load_data, get_layer_outs


#Provide a seed for reproducability
np.random.seed(7)

def coarse_weighted_analysis(correct_classification_idx, misclassification_idx, layer_outs):
#(predictions, true_classes,model, prdctn_tobe_anlyzd=None, true_tobe_anlyzd=None):

    scores = []
    suspicious_neuron_idx = []
    for l_out in layer_outs:
        scores.append(np.zeros(len(l_out[0][0])))
        suspicious_neuron_idx.append([])

    for layer_idx in range(1, len(layer_outs[1:])):
        test_idx = 0
        for l in layer_outs[layer_idx][0]:
            if test_idx not in misclassification_idx:
                test_idx +=1
                continue
            max_idx = np.where(l == l.max())
            scores[layer_idx][max_idx] += 1
            test_idx += 1

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] > 200:  # TODO: Threshold? what's the correct value? maybe can be found through experiments?
                suspicious_neuron_idx[i].append(j)

    return suspicious_neuron_idx[1:-1]



def fine_weighted_analysis(model, predictions, true_classes, prediction_tobe_analyzed, true_tobe_analyzed=None):

    error_class_to_input= []
    idx = 1
    for pred, crrct in zip(predictions, true_classes):
        predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
        true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]

        #if user does not specify the true class,  we consider all predictions that are equal to "given predicted class" and not correct
        if true_tobe_analyzed == None and predicted_class == prediction_tobe_analyzed and predicted_class != true_class:
            error_class_to_input.append(idx)
        #if user specifies a true class we consider predictions that are equal to "given predicted class" and expected to be "given true class"
        elif predicted_class == prediction_tobe_analyzed and true_class == true_tobe_analyzed and predicted_class != true_class:
            error_class_to_input.append(idx)

    class_specific_test_set = np.ndarray(shape=(len(error_class_to_input),1,28,28))

    cnt = 0
    for test_input in error_class_to_input:
        class_specific_test_set[cnt] = test_input
        cnt += 1

    layer_outs = get_layer_outs(model, class_specific_test_set)

    scores = []
    suspicious_neuron_idx = []
    for l_out in layer_outs:
        scores.append(np.zeros(len(l_out[0][0])))
        suspicious_neuron_idx.append([])

    for layer_idx in range(1, len(layer_outs[1:])):
        for l in layer_outs[layer_idx][0]:
            max_idx = np.where(l == l.max())
            scores[layer_idx-1][max_idx] += 1

    for i in len(scores):
        for j in len(scores[i]):
            if scores[i][j] > 5: # TODO: threshold?
                suspicious_neuron_idx[i].append(j)

    print(suspicious_neuron_idx)

    return suspicious_neuron_idx


def coarse_intersection_analysis(correct_classification_idx, misclassification_idx, layer_outs):

    suspicious_neuron_idx = []

    for l_out in layer_outs[1:]:
        dominant = range(len(l_out[0][0]))
        test_idx = 0
        for l in l_out[0]:
            if test_idx not in misclassification_idx: 
                test_idx += 1
                continue
            dominant = list(np.intersect1d(dominant, np.where(l > 0)))
            test_idx += 1
        suspicious_neuron_idx.append(dominant)

    return suspicious_neuron_idx[:-1]

def fine_intersection_analysis(model, predictions, true_classes,
                               prediction_tobe_analyzed,
                               true_tobe_analyzed=None):

    error_class_to_input= []
    idx = 1
    for pred, crrct in zip(predictions, true_classes):
        predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
        true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]

        #if user does not specify the true class,  we consider all predictions that are equal to "given predicted class" and not correct
        if true_tobe_analyzed == None and predicted_class == prediction_tobe_analyzed and predicted_class != true_class:
            error_class_to_input.append(idx)
        #if user specifies a true class we consider predictions that are equal to "given predicted class" and expected to be "given true class"
        elif predicted_class == prediction_tobe_analyzed and true_class == true_tobe_analyzed and predicted_class != true_class:
            error_class_to_input.append(idx)

        idx += 1

    class_specific_test_set = np.ndarray(shape=(len(error_class_to_input),1,28,28))

    cnt = 0
    for test_input in error_class_to_input:
        class_specific_test_set[cnt] = test_input
        cnt += 1

    layer_outs = get_layer_outs(model, class_specific_test_set)

    suspicious_neuron_idx= [[] for i in range(len(layer_outs))]

    for l_out in layer_outs[1:]:
        dominant = range(len(l_out[0][0]))
        for l in l_out[0]:
            dominant = np.intersect1d(dominant, np.where(l > 0))
        suspicious_neuron_idx.append(dominant)

    return suspicious_neuron_idx


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
    suspicious_neuron_idx = []
    for l_out in layer_outs:
        scores.append(np.zeros(len(l_out[0][0])))
        suspicious_neuron_idx.append([])

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

    print (scores)
    exit()
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] > 200: #score threshold, what's the correct value? maybe can be found through experiments?
                suspicious_neuron_idx[i].append(j)
