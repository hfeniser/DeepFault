from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU
from utils import get_python_version

python_version = get_python_version()


def __save_trained_model(model=None, num_hidden=None, num_neuron=None):
    directory = "neural_networks/"
    model_name = 'mnist_test_model_' + str(num_hidden) + '_' + str(num_neuron)
    model_filename = directory + model_name + ".json"
    weights_filename = directory + model_name + ".h5"

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)

    print("Model saved at: " + model_filename)

    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Weights saved at: " + weights_filename)

    return model_name


def train_model(args, X_train=None, Y_train=None, X_test=None, Y_test=None):
    """
    Construct a neural network, given the parameters, and train it on the MNIST dataset.
    Once done, save the neural network and its weights.
    :param args: arguments (# of hidden layers and neurons per layer)
    :return:
    """

    num_hidden = None if args == None else args['hidden']
    num_neuron = None if args == None else args['neurons']
    activation = args['activation']

    if python_version == 2 :
        if num_hidden is None:
            num_hidden = int(raw_input('Enter number of hidden layers: '))
        if num_neuron is None:
            num_neuron = int(raw_input('Enter number of neurons in each hidden layer: '))
    else:
        if num_hidden is None:
            num_hidden = int(input('Enter number of hidden layers: '))
        if num_neuron is None:
            num_neuron = int(input('Enter number of neurons in each hidden layer: '))

    print('Activations are ReLU. Optimizer is ADAM. Batch size 32. Fully connected network without dropout.')

    # Construct model
    model = Sequential()

    # Add input layer.
    # MNIST dataset: each image is a 28x28 pixel square (784 pixels total).
    model.add(Flatten(input_shape=(1, 28, 28)))

    # Add hidden layers.
    for _ in range(num_hidden):
        model.add(Dense(num_neuron,  use_bias=False))
        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.01))
        elif activation == 'relu':
            model.add(Activation('relu'))
    # Add output layer
    model.add(Dense(10, activation='softmax', use_bias=False))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print information about the model
    print(model.summary())

    # Fit the model
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    # Evaluate the model
    # if (X_test is not None and Y_test is not None):
    #     score = model.evaluate(X_test, Y_test, verbose=0)
    #     print('[loss, accuracy] -> ' + str(score))

    model_name = __save_trained_model(model, num_hidden, num_neuron)

    print("Training done")

    return model_name, model


if __name__ == "__main__":
    train_model(None)


def train_model_fault_localisation(model, X_train, Y_train, batch=100):
    """
    Method for retraining the neural network based on the perturbed inputs
    :param X_train: perturbed input
    :param Y_train: perturbed labels
    :return:
    """

    # Print information about the model
    # print(model.summary())

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch, epochs=10, verbose=1)
