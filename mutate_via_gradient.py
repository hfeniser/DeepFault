from keras import backend as K
import numpy as np
from utils import load_model, load_data, get_layer_outs, normalize
import random

def mutate(model, X_val, Y_val, dominant_indices, correct_classifications, d):

    input_tensor = model.layers[0].output

    perturbed_set_x = []
    perturbed_set_y = []

    #selct 10 inputs randomly from the correct classification set.
    zipped_random_data = random(zip(list(np.array(X_val)[correct_classifications]),
                            list(np.array(Y_val)[correct_classifications])), 10)

    for x, y in zipped_random_data:
        grads_for_doms = []
        flatX = [item for sublist in x[0] for item in sublist]
        for dom in dominant_indices:
            loss = model.layers[dom[0]].output[..., dom[1]]
            grads = K.gradients(loss, input_tensor)[0]
            iterate = K.function([input_tensor], [loss, grads])
            loss_val, grad_vals = iterate([np.expand_dims(flatX, axis=0)])
            grads_for_doms.append(grad_vals)

        c1 = 0
        c2 = 0
        perturbed_x = []
        for i in range(len(flatX)):

            allAgree = True
            min_abs_grad = float('inf')
            sum_grad = 0
            for j in range(len(grads_for_doms)):
                sum_grad += grads_for_doms[j][0][i]
                if min_abs_grad < abs(grads_for_doms[j][0][i]):
                    min_abs_grad = abs(grads_for_doms[j][0][i])
                if not j == 0 and not np.sign(grads_for_doms[j-1][0][i]) == np.sign(grads_for_doms[j][0][i]):
                    allAgree = False

            avg_grad = float(sum_grad) / len(dominant_indices)
            if avg_grad > d:
                avg_grad = d
            elif avg_grad < -d:
                avg_grad = -d

            perturbed_x.append(max(min(flatX[i] + avg_grad, 1), 0))

            '''
            if min_abs_grad > d:
                min_abs_grad = d

            if allAgree and grads_for_doms[0][0][i] > 0:
                perturbed_x.append(min(flatX[i] + min_abs_grad, 1))
                #c1 += 1
            elif allAgree and grads_for_doms[0][0][i] < 0:
                perturbed_x.append(max(flatX[i] - min_abs_grad, 0))
                #c2 += 1
            else:
                perturbed_x.append(flatX[i])
            '''

        perturbed_set_x.append(perturbed_x)
        perturbed_set_y.append(y)

    '''
    for xv, xp in zip(list(np.array(X_val)[correct_classifications])[:10], perturbed_set_x):

        xv = np.asarray(xv).reshape(1, 1, 28, 28)
        layer_outs = get_layer_outs(model, xv)

        print('BEFORE:')
        for dom in dominant_indices:
            print(layer_outs[dom[0]][0][0][dom[1]])

        xp = np.asarray(xp).reshape(1, 1, 28, 28)
        layer_outs = get_layer_outs(model, xp)
        print('AFTER:')
        for dom in dominant_indices:
            print(layer_outs[dom[0]][0][0][dom[1]])

        print('========')
    '''

    return perturbed_set_x, perturbed_set_y



'''
        for j in range(len(grads_for_doms)): ##
            perturbed_x = []
            for i in range(len(flatX)):
                grad = grads_for_doms[j][0][i]
                if abs(grad) < d:
                    c1+=1
                    perturbed_x.append(flatX[i] + grad)
                else:
                    c2+=1
                    perturbed_x.append(flatX[i] + d)

            perturbed_set_x.append(perturbed_x)
            perturbed_set_y.append(y)
'''



