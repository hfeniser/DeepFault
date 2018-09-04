"""
This is the main file that executes the flow
of our fault localisation technique
"""
import argparse
from train_nn import train_model
from test_nn import test_model
from lp import run_lp
from os import path
from spectrum_analysis import *
from weighted_analysis import *
from utils import save_perturbed_test, save_perturbed_test_groups
from datetime import datetime

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
    parser.add_argument("-L", "--layers",  help="number of hidden layers")
    parser.add_argument("-N", "--neurons", help="number of neurons in each hidden layer")
    parser.add_argument("-M", "--model",   help="the model to be loaded. When set, the specified model is used")
    parser.add_argument("-T", "--test",   help="the model to be loaded. When set, the specified model is used")
    parser.add_argument("-A", "--approach", help="the approach to be employed to localize dominant neurons")

    # parse command-line arguments
    args = parser.parse_args()

    # check arguments
    valid_args = True
    try:
        if args.version:
            print("this is myprogram version 0.1")
        if args.layers:
            if isinstance(args.layers, int) or args.layers.isdigit():
                value = int(args.layers)
                args.layers = value
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
    args['model'] = "neural_networks/mnist_test_model_5_5"
    args['test'] = True
    # for key,value in args.items():
    #     print(key,"\t", value)

    # 1)train the neural network and save the network and its weights after the training
    # Note: if the model is given as a command-line argument don't train it again
    if not args['model'] is None:
        model_name = args['model']
    else:
        model_name = train_model(args)

    # define experiment name
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    experiment_name = model_name + "_" + timestamp

    # 2)test the model and receive the indexes of correct and incorrect classifications
    # Also provide output of each neuron in each layer for tst input x.
    if not args['test'] is None and args['test']:
        correct_classifications, incorrect_classifications, layer_outs = test_model(model_name)

    # TODO: Hasan: need to modify the scripts that perform the identification so that to match the workflow
    # This function will receive the incorrect classifications and identify the dominant neurons for each layer
    # 3) Identify dominant neurons
    # e.g., weighted_analysis (correct_classifications, incorrect_classifications)

    if args['approach'] == 'intersection':
        dominant_neuron_idx = coarse_intersection_analysis(correct_classifications, incorrect_classifications, layer_outs)
    elif args['approach'] == 'tarantula':
        dominant_neuron_idx = tarantula_analysis(correct_classifications, incorrect_classifications, layer_outs)
    elif args['approach'] == 'ochiai':
        dominant_neuron_idx = ochiai_analysis(correct_classifications, incorrect_classifications, layer_outs)
    elif args['approach'] == 'weighted':
        dominant_neuron_idx = coarse_weighted_analysis(correct_classifications, incorrect_classifications, layer_outs)
    else:
        print('Please enter a valid approach to localize dominant neurons.')

    # Assume these are generated in Step3
    from utils import get_dummy_dominants
    dominant_neuron_idx = get_dummy_dominants(model_name)

    # TODO: Simos: this function will receice the set of dominant neurons for each layer from Step 3
    # and will produce new inputs based on the correct classifications (from the testing set)
    # that exercise the dominant neurons
    # 4) Run LP
    x_perturbed, y_perturbed = run_lp(model_name, dominant_neuron_idx, correct_classifications)

    #save perturtbed inputs
    save_perturbed_test_groups(x_perturbed, y_perturbed, experiment_name, 1)


