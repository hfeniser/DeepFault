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
    parser.add_argument("-M", "--model",   help="the model to be loaded")


    #parse command-line arguments
    args = parser.parse_args()

    # check arguments
    valid_args = True
    if args.version:
        print("this is myprogram version 0.1")
    if args.layers:
        try:
            value = int(args.layers)
            args.layers = value
            print("The NN has %s hidden layers" % args.layers)
        except ValueError:
            valid_args = False
            print(args.hidden, "is not a valid argument for the number of hidden layers")
    if args.neurons:
        try:
            value = int(args.neurons)
            args.neurons = value
            print("Each hidden layer has %s neurons" % args.neurons)
        except ValueError:
            valid_args = False
            print(args.neurons, "is not a valid argument for the number of neurons in each hidden layer")
    if args.model:
        try:
            if not (path.isfile(args.model+".json") and
                    path.isfile(args.model + ".h5")):
                raise FileNotFoundError ("Model %s not found" % args.model)
                valid_args = False
        except FileNotFoundError as e:
            valid_args = False
            print(e)

    if not valid_args:
        exit(-1)

    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    # for key,value in args.items():
    #     print(key,"\t", value)

    #1) train the neural network and save the network and its weights after the training
    #if the model is given don't train it again
    if 'model' in args:
        model_name = args['model']
    else:
        model_name = train_model(args)

    #2) test the model
    correct_classifications, incorrect_classifications = test_model(model_name)

    #TODO: Hasan: need to modify the scripts that perform the identification so that to match the workflow
    #This function will receive the incorrect classifications and identify the dominant neurons for each layer
    #3) Identify dominant neurons


    #TODO: Simos: this function will receice the set of dominant neurons for each layer from Step 3
    #and will produce new inputs based on the correct classifications (from the testing set)
    #that exercise the dominant neurons
    #4) Run LP
    # run_lp()

