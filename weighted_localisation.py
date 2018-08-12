from utils import load_model, load_data, get_layer_outs
import numpy  as np
import argparse

parser = argparse.ArgumentParser(description='An MNIST Network\'s Weighted Neuron Analysis')
parser.add_argument("-ec", "--error_class", type=int, help='Label of the predicted class by NN.')
parser.add_argument('-tc', "--true_class",  type=int, help='Label of the true (expected) class.')
args = parser.parse_args()

#Provide a seed for reproducability
np.random.seed(7)

X_train, Y_train, X_test, Y_test = load_data()
model = load_model('simple_mnist_fnn')

error_class_to_input= []
predictions = model.predict(X_test)

idx = 1
for pred, crrct in zip(predictions, Y_test):
    predicted_class = np.unravel_index(pred.argmax(), pred.shape)[0]
    true_class = np.unravel_index(crrct.argmax(), crrct.shape)[0]

    if args.true_class == None:
        #if user does not specify the true class (it is optional),  we consider all predictions that are equal to "given error class" and not correct
        condition = predicted_class == args.error_class and predicted_class != true_class
    else:
        #if user specifies a true class we consider predictions that are equal to "given error class" and expected to be "given true class"
        condition = predicted_class == args.error_class and true_class == args.true_class

    if condition:
        error_class_to_input.append(idx)

    idx += 1

class_specific_test_set = np.ndarray(shape=(len(error_class_to_input),1,28,28))

cnt = 0
for test_input in error_class_to_input:
    class_specific_test_set[cnt] = test_input
    cnt += 1

    layer_outs = get_layer_outs(model, class_specific_test_set)

scores = []
for l_out in layer_outs[1:]:
    scores.append(np.zeros(len(l_out[0][0])))

for layer_idx in range(1, len(layer_outs[1:])):
    for l in layer_outs[layer_idx][0]:
        max_idx = np.where(l == l.max())
        scores[layer_idx-1][max_idx] += 1

print scores
