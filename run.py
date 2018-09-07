"""
This is the main file that executes the flow
of our fault localisation technique
"""
import argparse
from train_nn import train_model, train_model_fault_localisation
from test_nn import test_model
from lp import run_lp
from os import path
from spectrum_analysis import *
from weighted_analysis import *
from utils import create_experiment_dir, get_dummy_dominants, save_perturbed_test_groups, load_perturbed_test_groups
from utils import load_classifications, save_classifications, save_layer_outs, load_layer_outs
from sklearn.model_selection import train_test_split

#Provide a seed for reproducability
seed = 7
experiment_path = "data"
model_path = "neural_networks"
group_index = 1


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Fault localisation for Deep Neural Networks'

    # print (os.getcwd())
    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version", help="show program version",    action="store_true")
    parser.add_argument("-H", "--hidden",  help="number of hidden layers")
    parser.add_argument("-N", "--neurons", help="number of neurons in each hidden layer")
    parser.add_argument("-M", "--model",   help="the model to be loaded. When set, the specified model is used")
    parser.add_argument("-T", "--test",   help="the model to be loaded. When set, the specified model is used")
    parser.add_argument("-A", "--approach", help="the approach to be employed to localize dominant neurons")
    parser.add_argument("-L", "--lp", help="whether to use linear programming to generate the perturbed inputs or" +
                                           " the perturbed input file to be used")

    # parse command-line arguments
    args = parser.parse_args()

    # check arguments
    valid_args = True
    try:
        if args.version:
            print("this is myprogram version 0.1")
        if args.hidden:
            if isinstance(args.layers, int) or args.layers.isdigit():
                value = int(args.hidden)
                args.hidden = value
                print("The NN has %s hidden layers" % args.layers)
            else:
                raise ValueError("%s is an invalid argument for the number of neurons in hidden layers" % args.layers)
        if args.neurons:
            if isinstance(args.neurons, int) or args.neurons.isdigit():
                value = int(args.neurons)
                args.neurons = value
                print("Each hidden layer has %s neurons" % args.neurons)
            else:
                raise ValueError("%s is an invalid argument for the number of neurons in hidden layers" % args.neurons)
        if args.model:
            if not (path.isfile(args.model+".json") and
                    path.isfile(args.model + ".h5")):
                raise FileNotFoundError("Model %s not found" % args.model)
        if args.test:
            if args.test == 'True':
                args.test = True
            elif args.test == 'False':
                args.test = False
            else:
                raise ValueError("Test argument is not valid ('%s')" % args.test)
    except (ValueError, FileNotFoundError) as e:
        valid_args = False
        print(e)

    if not valid_args:
        exit(-1)

    return vars(args)



if __name__ == "__main__":
    args = parse_arguments()
    args['model'] = "mnist_test_model_10_10"
    args['test'] = "mnist_test_model_10_10_2018-09-07 19:07:36"
    args['approach'] = 'tarantula'
    args['lp'] = "mnist_test_model_10_10_2018-09-07 19:18:30"
    # args['neurons'] = 10
    # args['layers'] = 10

    ####################
    # 0) Load MNIST data
    X_train, Y_train, X_test, Y_test = load_data()
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=1/6.0, random_state=seed)

    ####################
    # 1)train the neural network and save the network and its weights after the training
    if args['model'] is None or args['model'] is True:
        model_name, model = train_model(args, X_train, Y_train, X_test, Y_test)
    else:# if the model is given as a command-line argument don't train it again
        model_name = args['model']
        model = load_model(path.join(model_path, model_name))


    # create experiment name
    experiment_name = create_experiment_dir(experiment_path, model_name)

    # test set becomes validation set (temporary)
    test_model(model, X_test, Y_test)
    X_val, Y_val = X_test, Y_test

    ####################
    # 2)test the model and receive the indexes of correct and incorrect classifications
    # Also provide output of each neuron in each layer for test input x.
    if args['test'] is None or args['test'] is True:
        correct_classifications, misclassifications, layer_outs = test_model(model, X_val, Y_val)
        save_classifications(correct_classifications, misclassifications, experiment_name, group_index)
        save_layer_outs(layer_outs, experiment_name, group_index)
    else:
        filename = path.join(experiment_path, args['test'])
        correct_classifications, misclassifications = load_classifications(filename, group_index)
        layer_outs = load_layer_outs(filename, group_index)

    ####################
    # 3) Receive the correct classifications  & misclassifications and identify the dominant neurons per layer
    dominant_neuron_idx = []
    if args['approach'] == 'intersection':
        dominant_neuron_idx = coarse_intersection_analysis(correct_classifications, misclassifications, layer_outs)
    elif args['approach'] == 'tarantula':
        dominant_neuron_idx = tarantula_analysis(correct_classifications, misclassifications, layer_outs)
    elif args['approach'] == 'ochiai':
        dominant_neuron_idx = ochiai_analysis(correct_classifications, misclassifications, layer_outs)
    elif args['approach'] == 'weighted':
        dominant_neuron_idx = coarse_weighted_analysis(correct_classifications, misclassifications, layer_outs)
    else:
        print('Please enter a valid approach to localize dominant neurons.')

    # Assume these are generated in Step3
    if not dominant_neuron_idx:
        dominant = get_dummy_dominants(model)
        print("no fault localisation approach specified. Generating random dominant neurons", dominant_neuron_idx)
    else:
        dominant = {x: dominant_neuron_idx[x - 1] for x in range(1, len(dominant_neuron_idx) + 1)}

    ####################
    # 4) Run LP
    # Receive the set of dominant neurons for each layer from Step 3 # and will produce new inputs based on
    # the correct classifications (from the testing set) that exercise the dominant neurons
    if args['lp'] is None or args['lp'] is True:
        x_perturbed, y_perturbed = run_lp(model, X_val, Y_val, dominant, correct_classifications)
    else:
        filename = path.join(experiment_path, args['lp'])
        x_perturbed, y_perturbed = load_perturbed_test_groups(filename, group_index)

    # reshape them into the expected format
    x_perturbed = np.asarray(x_perturbed).reshape(np.asarray(x_perturbed).shape[0], 1, 28, 28)#
    y_perturbed = np.asarray(y_perturbed).reshape(np.asarray(y_perturbed).shape[0], 10)#

    ####################
    #save perturtbed inputs
    if args['lp'] is None or args['lp'] is True:
        save_perturbed_test_groups(x_perturbed, y_perturbed, experiment_name, group_index)

    ####################
    # retrain the model
    train_model_fault_localisation(model, x_perturbed, y_perturbed)

    ####################
    # retest the model
    test_model(model, X_test, Y_test)

