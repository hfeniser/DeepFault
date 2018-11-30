from utils import load_model, load_data, get_layer_outs, normalize
from keras import backend as K
import numpy as np
import random

def synthesize(model, zipped_data, suspicious_indices, correct_classifications, step_size, d):
    input_tensor = model.layers[0].output

    perturbed_set_x = []
    perturbed_set_y = []
    original_set_x  = []

    for x, y in zipped_data:
        grads_for_doms = []
        flatX = [item for sublist in x[0] for item in sublist]
        for dom in suspicious_indices:
            loss = K.mean(model.layers[dom[0]].output[..., dom[1]])
            grads = normalize(K.gradients(loss, input_tensor)[0])
            iterate = K.function([input_tensor], [loss, grads])
            loss_val, grad_vals = iterate([np.expand_dims(flatX, axis=0)])
            grads_for_doms.append(grad_vals)

        perturbed_x = []
        for i in range(len(flatX)):

            allAgree = True
            min_abs_grad = float('inf')
            sum_grad = 0
            for j in range(len(grads_for_doms)):
                sum_grad += grads_for_doms[j][0][i]

            avg_grad = float(sum_grad) / len(suspicious_indices)
            avg_grad = avg_grad * step_size

            if avg_grad > d:
                avg_grad = d
            elif avg_grad < -d:
                avg_grad = -d

            perturbed_x.append(max(min(flatX[i] + avg_grad, 1), 0))

        perturbed_set_x.append(perturbed_x)
        perturbed_set_y.append(y)
        original_set_x.append(x)

    return perturbed_set_x, perturbed_set_y, original_set_x
