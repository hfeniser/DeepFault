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

X_train, y_train, X_test, y_test = load_data()
model = load_model('simple_mnist_fnn')

predictions = model.predict(X_test)
idx = 0
for pred, crrct in zip(predictions, y_test):
    predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
    true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]
    if predicted_class != true_class:
        print 'predicted: ' + str( predicted_class)
        print 'true: ' + str(true_class)
        break
    idx += 1

tst = get_layer_outs(model,[X_test[idx]])
tst = tst[3][0][0]
tst = np.expand_dims(tst, axis=0)

saliency = JacobianSaliency(model, 5)
saliency_map = saliency.get_saliency_map(tst, model, 5, 'increase')[0]

max_sal = max(saliency_map)
max_ind = list(saliency_map).index(max_sal)
print max_sal
print max_ind



