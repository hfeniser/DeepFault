from utils import normalize
from keras import backend as K
import numpy as np
import random

def synthesize(model, zipped_data, suspicious_indices, correct_classifications, step_size, d):

    input_tensor = model.layers[0].output

    perturbed_set_x = []
    perturbed_set_y = []
    original_set_x  = []

    for x, y in zipped_data:
        all_grads = []
        flatX = [item for sublist in x[0] for item in sublist]
        for s_ind in suspicious_indices:
            loss = K.mean(model.layers[s_ind[0]].output[..., s_ind[1]])
            grads = normalize(K.gradients(loss, input_tensor)[0]) #loss not used at the moment
            iterate = K.function([input_tensor], [loss, grads])
            _, grad_vals = iterate([np.expand_dims(x, axis=0)])
            all_grads.append(grad_vals)

        perturbed_x = []

        for i in range(x.shape[0]):
            for k in range(x.shape[1]):
                sum_grad = 0
                for j in range(len(all_grads)):
                    sum_grad += all_grads[j][0][i][k][0]

                avg_grad = float(sum_grad) / len(suspicious_indices)
                avg_grad = avg_grad * step_size

                if avg_grad > d:
                    avg_grad = d
                elif avg_grad < -d:
                    avg_grad = -d

                perturbed_x.append(max(min(x[i][k][0] + avg_grad, 1), 0))

        '''
        for i in range(len(flatX)):
            sum_grad = 0
            for j in range(len(all_grads)):
                sum_grad += all_grads[j][0][i]

            avg_grad = float(sum_grad) / len(suspicious_indices)
            avg_grad = avg_grad * step_size

            if avg_grad > d:
                avg_grad = d
            elif avg_grad < -d:
                avg_grad = -d

            perturbed_x.append(max(min(flatX[i] + avg_grad, 1), 0))
        '''

        perturbed_set_x.append(perturbed_x)
        perturbed_set_y.append(y)
        original_set_x.append(x)

        if len(perturbed_set_x) % 5 == 0:
            print(("Input index:", len(perturbed_set_x)))

    return perturbed_set_x, perturbed_set_y, original_set_x
