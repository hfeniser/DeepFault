from keras.models import Sequential, model_from_json
from keras import backend as K
from collections import defaultdict
from utils import load_data
import numpy as np
import argparse
import tensorflow as tf

class GradientSaliency(object):

    def __init__(self, model, output_index = 0):
        # Define the function to compute the gradient
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_mask(self, input_image):
        # Execute the function to compute the gradient
        #x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([input_image])[0][0]

        return gradients


json_file = open('simple_mnist_fnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into model
model.load_weights("simple_mnist_fnn.h5")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


X_train, y_train, X_test, y_test = load_data()

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
saliency = GradientSaliency(model, 5)
mask = saliency.get_mask(img)

max_grad = max([max(sublist) for sublist in mask[0]])

i= 0
for sub in mask[0]:
    if max_grad in sub:
        j = list(sub).index(max_grad)
        break
    i += 1

X_test[idx][0][i][j] = 1
img = np.array(X_test[idx])
img = np.expand_dims(img, axis=0)
print model.predict(img)
print X_test[idx][0][i][j]



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

