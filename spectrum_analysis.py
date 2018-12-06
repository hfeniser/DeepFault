from keras import backend as K
from collections import defaultdict
import numpy as np
from utils import get_layer_outs
import math

'''

[1] Empirical Evaulation of the Tarantaul automatic fault localization technique
[2] Zoogeographic studies on the soleoids fishes found in Japan and its
neighbourings regions.
[3] The Dstar Method for Effective Fault Localization

'''

def tarantula_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num):
    '''
    More information on Tarantula fault localization technique can be found in
    [1]
    '''
    suspicious_neuron_idx = [[] for i in range(1, len(trainable_layers))] 


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



def ochiai_analysis(available_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num):
    '''
    More information on Ochiai fault localization technique can be found in
    [2]
    '''

    suspicious_neuron_idx = [[] for i in range(1, len(available_layers))]

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = float(num_cf[i][j]) / ((num_cf[i][j] + num_uf[i][j]) * (num_cf[i][j] + num_cs[i][j])) **(.5)
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


def dstar_analysis(available_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, star):
    '''
    More information on DStar fault localization technique can be found in
    [3]
    '''

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = float(num_cf[i][j]**star) / (num_cs[i][j] + num_uf[i][j])
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

