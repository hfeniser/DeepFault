from utils import load_model, load_data, get_layer_outs
import numpy  as np

#Provide a seed for reproducability
np.random.seed(7)

def coarse_weighted_analysis(correct_classification_idx, misclassification_idx, layer_outs):
#(predictions, true_classes,model, prdctn_tobe_anlyzd=None, true_tobe_anlyzd=None):

    scores = []
    dominant_neuron_idx = []
    for l_out in layer_outs:
        scores.append(np.zeros(len(l_out[0][0])))
        dominant_neuron_idx.append([])

    for layer_idx in range(1, len(layer_outs[1:])):
        test_idx = 0
        for l in layer_outs[layer_idx][0]:
            if test_idx not in misclassification_idx: continue
            max_idx = np.where(l == l.max())
            scores[layer_idx-1][max_idx] += 1
            test_idx += 1

    for i in len(scores):
        for j in len(scores[i]):
            if scores[i][j] > 5:
                dominant_neuron_idx[i].append(j)

    print dominant_neuron_idx

    return dominant_neuron_idx



def fine_weighted_analysis(predictions, true_classes, prediction_tobe_analyzed, true_tobe_analyzed=None, model):

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
    dominant_neuron_idx = []
    for l_out in layer_outs:
        scores.append(np.zeros(len(l_out[0][0])))
        dominant_neuron_idx.append([])

    for layer_idx in range(1, len(layer_outs[1:])):
        for l in layer_outs[layer_idx][0]:
            max_idx = np.where(l == l.max())
            scores[layer_idx-1][max_idx] += 1

    for i in len(scores):
        for j in len(scores[i]):
            if scores[i][j] > 5:
                dominant_neuron_idx[i].append(j)

    print dominant_neuron_idx

    return dominant_neuron_idx
