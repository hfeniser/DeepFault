from keras import backend as K
import numpy as np
from utils import load_model, load_data, get_layer_outs, normalize


def perturbe(model, X_val, Y_val, dominant, correct_classifications):    
 
    hidden_layers = [l for l in dominant.keys() if len(dominant[l]) > 0]
    target_layer = max(hidden_layers) 
    
    last_layer_doms = dominant[target_layer-2]
    
    input_tensor = model.layers[0].output

    perturbed_set_x = []
    perturbed_set_y = []
    for x, y in zip(list(np.array(X_val)[correct_classifications])[100:110], list(np.array(Y_val)[correct_classifications])[100:110]):
        grads_for_doms = []
        for dom in last_layer_doms:
            print(dom)
            flatX = [item for sublist in x[0] for item in sublist]
            
            loss = K.mean(model.layers[target_layer-2].output[..., dom]) #get_layer('leaky_re_lu_1').output[..., 2])
            grads = normalize(K.gradients(loss, input_tensor)[0])
            iterate = K.function([input_tensor], [loss, grads])
            loss_val, grad_vals = iterate([np.expand_dims(flatX, axis=0)]) 
            grads_for_doms.append(grad_vals)
        
        perturbed_x = []
        c1 = 0
        c2 = 0
        for i in range(len(flatX)):
            allAgree = True
            for j in range(len(grads_for_doms)):
                if not j == 0 and not np.sign(grads_for_doms[j-1][0][i]) == np.sign(grads_for_doms[j][0][i]):
                    allAgree = False

            if allAgree and grads_for_doms[0][0][i] > 0:
                perturbed_x.append(min(flatX[i] + 0.1, 1))
                c1 += 1
            elif allAgree and grads_for_doms[0][0][i] < 0:
                perturbed_x.append(max(flatX[i] - 0.1, 0))
                c2 += 1
            else:
                perturbed_x.append(flatX[i])

        perturbed_set_x.append(perturbed_x)
        perturbed_set_y.append(y)
        print(c1)
        print(c2)
        print('*****')
        if len(perturbed_set_x) == 10:
            break

    for xv, xp in zip(list(np.array(X_val)[correct_classifications])[100:110], perturbed_set_x):

        layer_outs = get_layer_outs(model, xv)
        print('BEFORE:')
        for dom in last_layer_doms:
            print(layer_outs[target_layer-2][0][0][dom])

        xp = np.asarray(xp).reshape(1, 1, 28, 28)
        layer_outs = get_layer_outs(model, xp)
        print('AFTER:')
        for dom in last_layer_doms:
            print(layer_outs[target_layer-1][0][0][dom])

        print('========')

    return perturbed_set_x, perturbed_set_y

    #perturbed_x = []
    #perturbed_x = [sum(x) for x in zip(grad_vals[0], flatX)] 
    

