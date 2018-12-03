from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation
from utils import get_python_version
from os import path

python_version = get_python_version()


def __save_trained_model(model, num_hidden, num_neuron,
                         model_prefix='mnist_test_model'):

    directory = "neural_networks/"

    if not path.exists(directory):
        makedirs(directory)

    model_name = model_prefix + '_' + str(num_hidden) + '_' + str(num_neuron) 
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


def train_model():
    """
    Construct a neural network, given the parameters, and train it.
    Once done, save the neural network and its weights.
    :param args: arguments (# of hidden layers and neurons per layer)
    :return:
    """

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

    print('Activations are LeakyReLU. Optimizer is ADAM. Batch sizei is 32.' + \
          'Fully connected network without dropout.')

    # Construct model
    model = Sequential()

    # Add input layer.
    # MNIST dataset: each image is a 28x28 pixel square (784 pixels total).
    model.add(Flatten(input_shape=(1, 28, 28)))

    # Add hidden layers.
    for _ in range(num_hidden):
        model.add(Dense(num_neuron,  use_bias=False))
        model.add(LeakyReLU(alpha=.01))

    # Add output layer
    model.add(Dense(10, activation='softmax', use_bias=False))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print information about the model
    print(model.summary())

    X_train, Y_train, X_test, Y_test = load_data()
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=1/6.0,
                                                      random_state=seed)

    # Fit the model
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    print("Save the model")
    model_name = __save_trained_model(model, num_hidden, num_neuron)

    print("Training done")

    return model_name, model


if __name__ == "__main__":
    train_model()


