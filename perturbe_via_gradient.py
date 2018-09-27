from keras import backend as K
import numpy as np
from utils import load_model, load_data, get_layer_outs

class Jacobian(object):

    def __init__(self, model, layer_index, dominant_index):
        # Define the function to compute the gradient
        self.compute_gradients = None
        input_tensors = [model.layers[0].output]
        gradients = model.optimizer.get_gradients(K.mean(model.get_layer('dense_2').output[..., 3]), model.layers[0].output)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_gradients(self, layer_out, model):
        grads = self.compute_gradients([layer_out])
        print('YEWS')
        print(len(grads))
        return grads[0]


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def perturbe(model, X_val, Y_val, dominant, correct_classifications):    
 
    hidden_layers = [l for l in dominant.keys() if len(dominant[l]) > 0]
    target_layer = max(hidden_layers) 
    
    last_layer_doms = dominant[target_layer]
    
    input_tensor = model.layers[0].output

    perturbed_set = []
    for x in X_val[:10]:
        grads_for_doms = []
        for dom in last_layer_doms:

            flatX = [item for sublist in x[0] for item in sublist]
            
            loss = K.mean(model.layers[-2].output[..., dom]) #get_layer('leaky_re_lu_1').output[..., 2])
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
                perturbed_x.append(min(flatX[i] + 0.2, 1))
                c1 += 1
            elif allAgree and grads_for_doms[0][0][i] < 0:
                perturbed_x.append(max(flatX[i] - 0.2, 0))
                c2 += 1
            else:
                perturbed_x.append(flatX[i])

        perturbed_set.append(perturbed_x)
        print(c1)
        print(c2)
        print('*****')

    for xv, xp in zip(X_val[:10], perturbed_set):

        layer_outs = get_layer_outs(model, xv)
        print('BEFORE:')
        for dom in last_layer_doms:
            print(layer_outs[-2][0][0][dom])
        
        xp = np.asarray(xp).reshape(1, 1, 28, 28)
        layer_outs = get_layer_outs(model, xp)
        print('AFTER:')
        for dom in last_layer_doms:
            print(layer_outs[-2][0][0][dom])

        print('========')

    return perturbed_set

        #perturbed_x = []
        #perturbed_x = [sum(x) for x in zip(grad_vals[0], flatX)] 
    

