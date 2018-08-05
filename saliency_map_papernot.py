
from keras.models import Sequential, model_from_json
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from collections import defaultdict
import numpy as np
import argparse
import tensorflow as tf

def data_mnist(one_hot=True):
    """
    Preprocess MNIST dataset
    """
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    print "Loaded MNIST test data."

    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, 10).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, 10).astype(np.float32)

    return X_train, y_train, X_test, y_test


class GradientSaliency(object):

    def __init__(self, model):
        # Define the function to compute the gradient
        self.compute_gradients = []
        input_tensors = [model.input]
        for i in range(model.outputs[0].shape[1]):
            gradients =  model.optimizer.get_gradients(model.output[0][i],
                                          model.input)
            self.compute_gradients.append(K.function(inputs = input_tensors,
                                                     outputs = gradients))

    def get_mask(self, input_image, model, target_label, sal_type = 'increase'):
        # Execute the function to compute the gradient
        #x_value = np.expand_dims(input_image, axis=0)
        gradient_repo = []
        for k in range(model.outputs[0].shape[1]):
            gradient_repo.append(self.compute_gradients[k]([input_image])[0][0][0])
        print target_label

        saliency_map = []
        for i in range(len(input_image[0][0])):
            saliency_map.append(np.zeros(len(input_image[0][0][0])))

        for i in range(len(input_image[0][0])):
            for j in range(len(input_image[0][0][0])):
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


json_file = open('simple_mnist_fnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into model
model.load_weights("simple_mnist_fnn.h5")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


X_train, y_train, X_test, y_test = data_mnist()

predictions = model.predict(X_test)
idx = 0
for pred, crrct in zip(predictions, y_test):
    predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
    true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]
    if predicted_class != true_class:
        print predicted_class
        print true_class
        break
    idx += 1

print y_test[idx]
img = np.array(X_test[idx])
img = np.expand_dims(img, axis=0)
print model.predict(img)
saliency = GradientSaliency(model)
saliency_map = saliency.get_mask(img, model, 5, 'increase')

max_grad = max([max(sublist) for sublist in saliency_map])

i = 0
for sub in saliency_map:
    if max_grad in sub:
        j = list(sub).index(max_grad)
        break
    i += 1

X_test[idx][0][i][j] = 1
img = np.array(X_test[idx])
img = np.expand_dims(img, axis=0)
print model.predict(img)

#jsma(model, X_test[0], 1)

#def jsma(model, original_array, count):
#    print model
#    gradients = K.gradients(model.output, model.input[0])
#    sess = tf.InteractiveSession()
#    sess.run(tf.initialize_all_variables())
#    evaluated_gradients = sess.run(gradients,feed_dict={model.input[0]:original_array})
#
#    exit()
#
#    gradients = K.gradients(model.model.output, model.model.input[0])
#    get_grad_values = K.function([model.model.input[0]], gradients)
#    print model.model.input[0].shape
#    print original_array.shape
#    grad_values     = get_grad_values([original_array])
#    print grad_values
#    exit()
#    max_grad_0 = np.max(grad_values[0][:,:,0])
#    max_grad_1 = np.max(grad_values[0][:,:,1])
#
#    max_grad_ind = unravel_index(grad_values[0][:,:,0].argmax(), grad_values[0][:,:,0].shape)
#    perturbation_pixel = original_array[0][:,:,0][max_grad_ind]
#    original_array[0][:,:,0][max_grad_ind] = original_array[0][:,:,0][max_grad_ind] + max_grad_0 
#    misc.imsave('perturbed/jsma/img_' + str(count) + '_0.jpg', original_array[0][:,:,0])
#
#    max_grad_ind = unravel_index(grad_values[0][:,:,1].argmax(), grad_values[0][:,:,1].shape)
#    perturbation_pixel = original_array[0][:,:,1][max_grad_ind]
#    original_array[0][:,:,1][max_grad_ind] = original_array[0][:,:,1][max_grad_ind] + max_grad_1
#    misc.imsave('perturbed/jsma/img_' + str(count) + '_1.jpg', original_array[0][:,:,1])
#
#    return original_array

