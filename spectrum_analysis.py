from keras import backend as K
from collections import defaultdict
import numpy as np
from utils import get_layer_outs
import math

#Provide a seed for reproducability
np.random.seed(7)

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


def tarantula_analysis(correct_classification_idx, misclassification_idx, layer_outs, model, suspicious_num):


    available_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            available_layers.append(model.layers.index(layer))
        except:
            pass
        
    available_layers = available_layers[1:] #ignore the input layer

    scores = []
    num_cf = []
    num_uf = []
    num_cs = []
    num_us = []


    for al in available_layers: 
        num_cf.append(np.zeros(model.layers[al].output_shape[1]))  # covered (activated) and failed
        num_uf.append(np.zeros(model.layers[al].output_shape[1]))  # uncovered (not activated) and failed
        num_cs.append(np.zeros(model.layers[al].output_shape[1]))  # covered and succeeded
        num_us.append(np.zeros(model.layers[al].output_shape[1]))  # uncovered and succeeded
        scores.append(np.zeros(model.layers[al].output_shape[1]))


    for al in available_layers:
        layer_idx = available_layers.index(al)
        all_neuron_idx = range(model.layers[al].output_shape[1]) 
        test_idx = 0
        for l in layer_outs[al][0]:
            covered_idx   = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx)-set(covered_idx))
            if test_idx  in correct_classification_idx:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassification_idx:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1

    suspicious_neuron_idx = [[] for i in range(1, len(available_layers))]

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = float(float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j])) / (float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j]) + float(num_cs[i][j]) / (num_cs[i][j] + num_us[i][j]))
            if np.isnan(score):
                score = 0
            scores[i][j] = score


    flat_scores = [float(item) for sublist in scores for item in sublist if not
               math.isnan(float(item))]

    relevant_vals = sorted(flat_scores, reverse=True)[:suspicious_num]

    suspicious_neuron_idx = []
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] in relevant_vals:
                if available_layers == None:
                    suspicious_neuron_idx.append((i,j))
                else:
                    suspicious_neuron_idx.append((available_layers[i],j))
            if len(suspicious_neuron_idx) == suspicious_num:
                break

    return suspicious_neuron_idx

    # flat_scores = [float(item) for sublist in scores for item in sublist if not
    #               math.isnan(item)]
    # percentile = np.percentile(flat_scores, percent)
    # # percentile = max(flat_scores)
    # for i in range(len(scores)):
    #     for j in range(len(scores[i])):
    #         if scores[i][j] >= percentile:
    #             suspicious_neuron_idx[i].append(j)

    # return suspicious_neuron_idx[:-1], scores


    # layer_idx = 0
    # for l_out in layer_outs[1:]:
    #     all_neuron_idx = range(len(l_out[0][0]))
    #     test_idx = 0
    #     for l in l_out[0]:
    #         covered_idx   = list(np.where(l > 0)[0])
    #         uncovered_idx = list(set(all_neuron_idx)-set(covered_idx))

    #         if test_idx  in correct_classification_idx:
    #             for cov_idx in covered_idx:
    #                 num_cs[layer_idx][cov_idx] += 1
    #             for uncov_idx in uncovered_idx:
    #                 num_us[layer_idx][uncov_idx] += 1
    #         elif test_idx in misclassification_idx:
    #             for cov_idx in covered_idx:
    #                 num_cf[layer_idx][cov_idx] += 1
    #             for uncov_idx in uncovered_idx:
    #                 num_uf[layer_idx][uncov_idx] += 1
    #         test_idx += 1
    #     layer_idx += 1



def ochiai_analysis(correct_classification_idx, misclassification_idx, layer_outs, percent):

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
        all_neuron_idx = range(len(l_out[0][0]))
        test_idx = 0
        for l in l_out[0]:
            covered_idx   = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx)-set(covered_idx))

            if test_idx  in correct_classification_idx:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassification_idx:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1
        layer_idx += 1


    suspicious_neuron_idx= [[] for i in range(1, len(layer_outs))]

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = float(num_cf[i][j]) / ((num_cf[i][j] + num_uf[i][j]) * (num_cf[i][j] + num_cs[i][j])) **(.5)
            scores[i][j] = score
            #if score > 0.29:  # TODO: Threshold? for identifying the dominant neurons. value via experimentation?
            #    suspicious_neuron_idx[i].append(j)

    flat_scores = [float(item) for sublist in scores for item in sublist if not
                  math.isnan(item)]
    ######!!!!!!!!For some reason it returns nan so i use nanpercentile !!!!!!!!!!!!########
    percentile = np.nanpercentile(flat_scores, percent)

    # percentile = max(flat_scores)
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] >= percentile:
                suspicious_neuron_idx[i].append(j)

    return suspicious_neuron_idx[:-1], scores


def dstar_analysis(correct_classification_idx, misclassification_idx,
                   layer_outs, percent, star):
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
        all_neuron_idx = range(len(l_out[0][0]))
        test_idx = 0
        for l in l_out[0]:
            covered_idx   = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx)-set(covered_idx))

            if test_idx  in correct_classification_idx:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassification_idx:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1
        layer_idx += 1


    suspicious_neuron_idx= [[] for i in range(1, len(layer_outs))]

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = float(num_cf[i][j]**star) / (num_cs[i][j] + num_uf[i][j])
            scores[i][j] = score

    flat_scores = [float(item) for sublist in scores for item in sublist if not
                  math.isnan(item)]

    '''
    ######!!!!!!!!For some reason it returns nan so i use nanpercentile !!!!!!!!!!!!########
    percentile = np.nanpercentile(flat_scores, percent)
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] >= percentile:
                suspicious_neuron_idx[i].append(j)
    '''

    return suspicious_neuron_idx[:-1], scores


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



def tarantula_analysis_for_class(correct_classification_idx, misclassification_idx, layer_outs, Y_set, class_index):
    class_indices = []
    #get indices of class_index
    for i in range(len(Y_set)):
        if np.argmax(Y_set[i]) == class_index:
            class_indices.append(i)

    scores = []
    num_cf = []
    num_uf = []
    num_cs = []
    num_us = []
    for l_out in layer_outs[1:]:
        num_cf.append(np.zeros(len(l_out[0][0])))  # covered (activated) and failed
        num_uf.append(np.zeros(len(l_out[0][0])))  # uncovered (not activated) and failed
        num_cs.append(np.zeros(len(l_out[0][0])))  # covered and succeeded
        num_us.append(np.zeros(len(l_out[0][0])))  # uncovered and succeeded
        scores.append(np.zeros(len(l_out[0][0])))

    layer_idx = 0
    for l_out in layer_outs[1:]:
        all_neuron_idx = range(len(l_out[0][0]))
        test_idx = 0
        for l in l_out[0]:
            if test_idx not in class_indices:
                test_idx += 1
                continue

            covered_idx   = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx)-set(covered_idx))

            if test_idx  in correct_classification_idx:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassification_idx:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1
        layer_idx += 1

    suspicious_neuron_idx = [[] for i in range(1, len(layer_outs))]

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = float(float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j])) / (float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j]) + float(num_cs[i][j]) / (num_cs[i][j] + num_us[i][j]))
            if np.isnan(score):
                score = 0
            scores[i][j] = score
            # if score > 0.53:  # TODO: threshold for identifying the dominant neurons. value via experimentation?
            #     suspicious_neuron_idx[i].append(j)

    flat_scores = [item for sublist in scores[:-1] for item in sublist]
    percentile = np.percentile(flat_scores, 95)
    # percentile = max(flat_scores)
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] >= percentile:
                suspicious_neuron_idx[i].append(j)

    print(suspicious_neuron_idx[:-1])
    return suspicious_neuron_idx[:-1]
