from keras.models import Sequential
from keras.layers import Dense, Flatten
from utils import load_data
from utils import get_python_version

python_version = get_python_version()


def train_model(args):
    """
    Construct a neural network, given the parameters, and train it on the MNIST dataset.
    Once done, save the neural network and its weights.
    :param args: arguments (# of hidden layers and neurons per layer)
    :return:
    """
    #Load MNIST data
    X_train, Y_train, X_test, Y_test = load_data()

    num_hidden = None if args == None else args['layers']
    num_neuron = None if args == None else args['neurons']

    if (python_version==2):
        if num_hidden is None:
            num_hidden = int(raw_input('Enter number of hidden layers: '))
        if num_neuron is None:
            num_neuron = int(raw_input('Enter number of neurons in each hidden layer: '))
    else:
        if num_hidden is None:
            num_hidden = int(input('Enter number of hidden layers: '))
        if num_neuron is None:
            num_neuron = int(input('Enter number of neurons in each hidden layer: '))

    print ('Activations are ReLU. Optimizer is ADAM. Batch size 32. Fully connected network without dropout.')

    #Construct model
    model = Sequential()

    #Add input layer.
    #MNIST dataset: each image is a 28x28 pixel square (784 pixels total).
    model.add(Flatten(input_shape=(1,28,28)))

    #Add hidden layers.
    for _ in range(num_hidden):
        model.add(Dense(num_neuron, activation='relu', use_bias=False))

    #Add output layer
    model.add(Dense(10, activation='softmax', use_bias=False))

    #Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #Print information about the model
    print(model.summary())

    #Fit the model
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    #Evaluate the model
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('[loss, accuracy] -> ' + str(score))


    model_name = 'mnist_test_model_' + str(num_hidden) + '_' + str(num_neuron)
    model_filename = model_name + ".json"
    weights_filename = model_name + ".h5"

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Model saved to disk with name: " + model_name)

    print("Training done\n")

    return model_name


if __name__ == "__main__":
    train_model(None)
