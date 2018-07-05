from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from collections import defaultdict
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='An MNIST Network\'s Neuron Analysis')
parser.add_argument("error_class", type=int, help='Label of the predicted class by NN.')
parser.add_argument('-tc', "--true_class", type=int, help='Label of the true (expected) class.')

args = parser.parse_args()

#Provide a seed for reproducability
np.random.seed(7)

#Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Preprocess dataset
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

json_file = open('simple_mnist_fnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("simple_mnist_fnn.h5")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


error_class_to_input= []

predictions = model.predict(X_test)

idx = 1
for pred, crrct in zip(predictions, Y_test):
    predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
    true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]

    if args.true_class == None:
        condition = predicted_class == args.error_class and predicted_class != true_class
    else:
        condition = predicted_class == args.error_class and true_class == args.true_class

#    This condition gives us the indices of the inputs that are correctly
#    clasified and their label are as specified by the user.
#    condition = predicted_class == args.error_class and predicted_class == true_class

    if condition:
        error_class_to_input.append(idx)

    idx += 1

print len(error_class_to_input)

class_specific_test_set = np.ndarray(shape=(len(error_class_to_input),1,28,28))

cnt = 0
for test_input in error_class_to_input:
    class_specific_test_set[cnt] = test_input
    cnt += 1


print class_specific_test_set.shape

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
# Testing
layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

activated_list = []
for l_out in layer_outs[1:]:
    activated = list(range(128))
    for l in l_out[0]:
        activated = np.intersect1d(activated, np.where(l == 0)) #list(set(activated) & set(list(np.where(l > 0)))) #list(numpy.where(l > 0)
    print activated
    activated_list.append(activated)




#Construct model
#model = Sequential()
#model.add(Flatten(input_shape=(1,28,28)))

#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))

#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

#model.fit(X_train, Y_train,
#          batch_size=32, nb_epoch=10, verbose=1)


#score = model.evaluate(X_test, Y_test, verbose=0)
#print score

# serialize model to JSON
#model_json = model.to_json()
#with open("simple_mnist_fnn.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("simple_mnist_fnn.h5")
#print("Model saved to disk")
