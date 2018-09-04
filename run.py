"""
This is the main file that executes the flow
of our fault localisation technique
"""
import argparse
from train_nn import train_model
from test_nn import  test_model
from lp import run_lp
from os import path


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

    #add new command-line arguments
    parser.add_argument("-V", "--version", help="show program version",    action="store_true")
    parser.add_argument("-L", "--layers",  help="number of hidden layers")
    parser.add_argument("-N", "--neurons", help="number of neurons in each hidden layer")
    parser.add_argument("-M", "--model",   help="the model to be loaded. When this parameter is set, the specified model is used")
    parser.add_argument("-T", "--test",   help="the model to be loaded. When this parameter is set, the specified model is used")


    #parse command-line arguments
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
                raise ValueError("%s is not a valid argument for the number of neurons in each hidden layer" % args.layers)
        if args.neurons:
            if isinstance(args.neurons, int) or args.neurons.isdigit():
                value = int(args.neurons)
                args.neurons = value
                print("Each hidden layer has %s neurons" % args.neurons)
            else:
                raise ValueError("%s is not a valid argument for the number of neurons in each hidden layer" % args.neurons)
        if args.model:
            if not (path.isfile(args.model+".json") and
                    path.isfile(args.model + ".h5")):
                raise FileNotFoundError ("Model %s not found" % args.model)
                valid_args = False
        if args.test:
            if args.test == 'True':
                args.test = True
            elif args.test == 'False':
                args.test = False
            else:
                raise ValueError ("Test argument is not valid ('%s')" % args.test)
    except (ValueError, FileNotFoundError) as e:
        valid_args = False
        print (e)

    if not valid_args:
        exit(-1)

    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    # for key,value in args.items():
    #     print(key,"\t", value)

    #1) train the neural network and save the network and its weights after the training
    #Note: if the model is given as a command-line argument don't train it again
    if not args['model'] is None:
        model_name = args['model']
    else:
        model_name = train_model(args)

    #2) test the model and receive the indexes of correct and incorrect classifications
    if not args['test'] is None and args['test']:
        correct_classifications, incorrect_classifications = test_model(model_name)

    #TODO: Hasan: need to modify the scripts that perform the identification so that to match the workflow
    #This function will receive the incorrect classifications and identify the dominant neurons for each layer
    #3) Identify dominant neurons

    #Assume these are generate in Step3
    from utils import load_model
    model = load_model("neural_networks/mnist_test_model_5_5")
    import random
    dominant = {x: random.sample(range(model.layers[x].output_shape[1]), 2) for x in range(1, len(model.layers) - 1)}
    print(dominant)

    #TODO: Simos: this function will receice the set of dominant neurons for each layer from Step 3
    #and will produce new inputs based on the correct classifications (from the testing set)
    #that exercise the dominant neurons
    #4) Run LP
    # run_lp()

