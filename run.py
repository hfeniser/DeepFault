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
from utils import create_experiment_dir, save_perturbed_test_groups, load_perturbed_test_groups
from utils import load_dominant_neurons, save_dominant_neurons
from utils import load_classifications, save_classifications, save_layer_outs, load_layer_outs
from utils import find_class_of, load_data
from mutate_via_gradient import mutate
from sklearn.model_selection import train_test_split
from saliency_map_analysis import saliency_map_analysis
import random
import datetime
#Provide a seed for reproducability
seed = 7
experiment_path = "experiment"
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
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-HL", "--hidden",  help="number of hidden layers")
    parser.add_argument("-N", "--neurons", help="number of neurons in each hidden layer")
    parser.add_argument("-M", "--model",   help="the model to be loaded. When set, the specified model is used")
    parser.add_argument("-T", "--test",   help="the model to be loaded. When set, the specified model is used")
    parser.add_argument("-A", "--approach", help="the approach to be employed to localize dominant neurons")
    parser.add_argument("-L", "--lp", help="whether to use linear programming to generate the perturbed inputs or" +
                                           " the perturbed input file to be used")
    parser.add_argument("-D", "--distance", help="the distance between the original and the mutated image.")
    parser.add_argument("-P", "--percentile", help="the percentage of suspicious neurons in all neurons.")
    parser.add_argument("-C", "--class", help="the label of inputs to analyze.")
    parser.add_argument("-MU", "--mutate", help="whether to mutate inputs or load previously mutated inputs")
    parser.add_argument("-AC", "--activation", help="activation function or  hidden neurons. it can be \"relu\" or \"leaky_relu\"")
    parser.add_argument("-LOG", "--logfile", help="path to log file")

    # parse command-line arguments
    args = parser.parse_args()

    # check arguments
    valid_args = True
    try:
        print(args)
        if args.version:
            print("this is myprogram version 1.0")
        if args.hidden:
            if isinstance(args.hidden, int) or args.hidden.isdigit():
                value = int(args.hidden)
                args.hidden = value
                print("The NN has %s hidden layers" % args.hidden)
            else:
                raise ValueError("%s is an invalid argument for the number of neurons in hidden layers" % args.hidden)
        if args.neurons:
            if isinstance(args.neurons, int) or args.neurons.isdigit():
                value = int(args.neurons)
                args.neurons = value
                print("Each hidden layer has %s neurons" % args.neurons)
            else:
                raise ValueError("%s is an invalid argument for the number of neurons in hidden layers" % args.neurons)
#        if args.model:
#            if not (path.isfile(args.model+".json") and
#                    path.isfile(args.model + ".h5")):
#                raise FileNotFoundError("Model %s not found" % args.model)
        if args.test:
            if args.test == 'True':
                args.test = True
            elif args.test == 'False':
                args.test = False
            else:
                raise ValueError("Test argument is not valid ('%s')" % args.test)
    except (ValueError,ValueError) as e:
        valid_args = False
        print(e)

    if not valid_args:
        exit(-1)

    return vars(args)



if __name__ == "__main__":

    args = parse_arguments()

    ####################
    # 0) Load MNIST data
    X_train, Y_train, X_test, Y_test = load_data()

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=1/6.0, random_state=seed)

    print args
    logfile = open(args['logfile'], 'a')
    logfile.write('\n')
    logfile.write('='*75)
    logfile.write('\n')
    logfile.write(str(args) + '\n')
    logfile.write('='*75)
    logfile.write('\n')


    ####################
    # 1)train the neural network and save the network and its weights after the training
    model_name = args['model']
    try:
        model = load_model(path.join(model_path, model_name))
    except:
        model_name, model = train_model(args, X_train, Y_train, X_test, Y_test)

    #if args['model'] is None or args['model'] is True:
    #    model_name, model = train_model(args, X_train, Y_train, X_test, Y_test)
    #else:# if the model is given as a command-line argument don't train it again
    #    model_name = args['model']
    #    model = load_model(path.join(model_path, model_name))

    experiment_name = model_name + '_' + args['class'] + '_' + args['activation'] + '_' + args['distance'] + '_' + args['approach'] + '_' + args['percentile']
    #experiment_name, timestamp = create_experiment_dir(experiment_path, model_name)

    # test set becomes validation set (temporary)
    # test_model(model, X_test, Y_test)
    X_val, Y_val = find_class_of(X_val, Y_val, int(args['class']))


    ####################
    # 2)test the model and receive the indexes of correct and incorrect classifications
    # Also provide output of each neuron in each layer for test input x.
    filename = experiment_path + '/' + model_name + '_' + args['class']
    try:
        correct_classifications, misclassifications = load_classifications(filename, group_index)
        layer_outs = load_layer_outs(filename, group_index)
    except:
        correct_classifications, misclassifications, layer_outs, y_predictions = test_model(model, X_val, Y_val)
        save_classifications(correct_classifications, misclassifications, filename, group_index)
        save_layer_outs(layer_outs, filename, group_index)

    ####################
    # 3) Receive the correct classifications  & misclassifications and identify the dominant neurons per layer
    filename = experiment_path + '/' + model_name + '_' + args['class'] + '_' + args['approach'] + '_' + args['percentile']
    if args['approach'] == 'tarantula':
        try:
            dominant_neuron_idx = load_dominant_neurons(filename, group_index)
        except:
            dominant_neuron_idx = tarantula_analysis(correct_classifications,
                                                 misclassifications,
                                                 layer_outs,
                                                 int(args['percentile']))
            save_dominant_neurons(dominant_neuron_idx, filename, group_index)

    elif args['approach'] == 'ochiai':
        try:
            dominant_neuron_idx = load_dominant_neurons(filename, group_index)
        except:
            dominant_neuron_idx = ochiai_analysis(correct_classifications,
                                                 misclassifications,
                                                 layer_outs,
                                                 int(args['percentile']))
            save_dominant_neurons(dominant_neuron_idx, filename, group_index)

    elif args['approach'] == 'random':
        filename = experiment_path + '/' + model_name + '_' + args['class'] + '_tarantula_' + args['percentile']
        dominant_neuron_idx_tarantula = load_dominant_neurons(filename, group_index)

        filename = experiment_path + '/' + model_name + '_' + args['class'] + '_ochiai_' + args['percentile']
        dominant_neuron_idx_ochiai = load_dominant_neurons(filename, group_index)

        dominant_neuron_idx = [[] for _ in range(len(dominant_neuron_idx_ochiai))]
        num_dominants = len([item for sub in dominant_neuron_idx_ochiai for item in sub])
        added = 0
        while added < num_dominants/2:

            rand_layer = random.randint(0, int(args['hidden'])-1)
            rand_idx   = random.randint(0, int(args['neurons']))

            if rand_idx not in dominant_neuron_idx_ochiai[2*rand_layer+1] and rand_idx not in dominant_neuron_idx_tarantula[2*rand_layer+1] and rand_idx not in dominant_neuron_idx[2*rand_layer]:
                dominant_neuron_idx[2*rand_layer].append(rand_idx)
                added += 1
        
    dominant = {x: dominant_neuron_idx[x - 1] for x in range(1, len(dominant_neuron_idx) + 1)}


    ####################
    # 4) Run Mutation Algorithm
    # Receive the set of dominant neurons for each layer from Step 3 # and will produce new inputs based on
    # the correct classifications (from the testing set) that exercise the
    # suspicious neurons

    layers = range(1, len(model.layers)-1)

    tot_start = datetime.datetime.now()
    for layer in layers[0::2]:
        print 'LAYER: ' + str(layer)
        dominant_indices = dominant[layer]
        print dominant_indices
        if len(dominant_indices) == 0:
            logfile.write('Model: ' + str(model_name) + ', Activation: ' +
                      args['activation'] + ', Class: ' + args['class'] + ', Layer ' + str(layer) +
                      ', Approach: ' + str(args['approach']) + ', Percentile: '
                      + str(args['percentile']) + ', Distance: ' +
                          str(args['distance']) + ' Score: No Suspicious. \n')
            continue

        if args['mutate'] is None or args['mutate'] is True:
             start = datetime.datetime.now()
             x_perturbed, y_perturbed = mutate(model, X_val, Y_val, layer,
                                               dominant_indices,
                                               correct_classifications,
                                               float(args['distance']))
             end = datetime.datetime.now()
        else:
            filename = args['mutate'] + args['approach']
            filename = path.join(experiment_path, filename)
            x_perturbed, y_perturbed = load_perturbed_test_groups(filename, group_index)

        # reshape them into the expected format
        x_perturbed = np.asarray(x_perturbed).reshape(np.asarray(x_perturbed).shape[0], 1, 28, 28)#
        y_perturbed = np.asarray(y_perturbed).reshape(np.asarray(y_perturbed).shape[0], 10)#

        ####################
        #save perturtbed inputs
        filename = path.join(experiment_path, experiment_name)
        filename = filename + '_layer' + str(layer)
        if args['mutate'] is None or args['mutate'] is True:
            save_perturbed_test_groups(x_perturbed, y_perturbed, filename, group_index)

        score = model.evaluate(x_perturbed, y_perturbed, verbose=0)
        logfile.write('Model: ' + str(model_name) + ', Activation: ' +
                      args['activation'] + ', Class: ' + args['class'] + ', Layer ' + str(layer) +
                      ', Approach: ' + str(args['approach']) + ', Percentile: '
                      + str(args['percentile']) + ', Distance: ' +
                      str(args['distance']) + ' Score: ' +
                      str(score) + '\n')
        logfile.write('Time: ' + str(end-start) + '\n\n')

    tot_end = datetime.datetime.now()
    logfile.write('Total Time: ' + str(tot_end-tot_start) + '\n\n')

        ####################
        # 5) Test if the mutated inputs are adversarial
        #test_model(model, x_perturbed, y_perturbed)

    logfile.close()

    '''
    ####################
    # 5) retrain the model
    # train_model_fault_localisation(model, x_perturbed, y_perturbed, len(x_perturbed))
    model.fit(x_perturbed, y_perturbed, batch_size=32, epochs=10, verbose=1)

    ####################
    # 6) retest the model
    test_model(model, X_test, Y_test)
    '''

    '''
    dominant_neuron_idx = []
    dominant_neurons_file_exists = False
    if args['approach'] == 'intersection':
        dominant_neuron_idx = coarse_intersection_analysis(correct_classifications, misclassifications, layer_outs)
    elif args['approach'] == 'tarantula':
        dominant_neuron_idx = tarantula_analysis(correct_classifications,
                                                 misclassifications,
                                                 layer_outs,
                                                 int(args['percentile']))
    elif args['approach'] == 'ochiai':
        dominant_neuron_idx = ochiai_analysis(correct_classifications,
                                              misclassifications, layer_outs,
                                              int(args['percentile']))
    elif args['approach'] == 'weighted':
        dominant_neuron_idx = coarse_weighted_analysis(correct_classifications, misclassifications, layer_outs)
    elif args['approach'] == 'saliency':
        dominant_neuron_idx = saliency_map_analysis(correct_classifications, misclassifications, layer_outs, model, y_predictions)
    elif args['approach'] is not None:
        filename = path.join(experiment_path, args['approach'])
        dominant_neuron_idx = load_dominant_neurons(filename, group_index)
        dominant_neurons_file_exists = True
    else:
        print('Please enter a valid approach to localize dominant neurons.')
        exit(-1)

    filename = model_name + "_" + args['approach']
    if not dominant_neurons_file_exists:
    '''
