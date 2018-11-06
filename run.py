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
from utils import create_experiment_dir, save_perturbed_test_groups, load_perturbed_test_groups
from utils import load_dominant_neurons, save_dominant_neurons
from utils import load_classifications, save_classifications, save_layer_outs, load_layer_outs
from utils import find_class_of, load_data, load_model, save_original_inputs
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
    parser.add_argument("-AC", "--activation", help="activation function or hidden neurons. it can be \"relu\" or \"leaky_relu\"")
    parser.add_argument("-SN", "--suspicious_num", help="number of suspicious neurons we consider")
    parser.add_argument("-SS", "--step_size", help="multiplication of gradients by step size")
    parser.add_argument("-R", "--repeat", help="index of the repeating. (for the cases where we run the same experiment multiple times)")
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

    #if args['activation'] == 'relu':
    model_name = model_name + '_' + args['activation']

    try:
        model = load_model(path.join(model_path, model_name))
    except:
        model_name, model = train_model(args, X_train, Y_train, X_test, Y_test)

    #if args['model'] is None or args['model'] is True:
    #    model_name, model = train_model(args, X_train, Y_train, X_test, Y_test)
    #else:# if the model is given as a command-line argument don't train it again
    #    model_name = args['model']
    #    model = load_model(path.join(model_path, model_name))

    experiment_name = model_name + '_' + str(args['class']) + '_' + \
    str(args['step_size']) + '_' + args['approach'] + '_SN' + \
    str(args['suspicious_num']) + '_R' + str(args['repeat'])
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
    # 3) Receive the correct classifications  & misclassifications and identify the suspicious neurons per layer
    ############################################
    #Preparations for finding suspicious neurons
    ############################################
    available_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            available_layers.append(model.layers.index(layer))
        except:
            pass
        
    available_layers = available_layers[1:] #ignore the input layer

    scores = []
    num_cf = []
    num_uf = []
    num_cs = []
    num_us = []
    for al in available_layers: 
        num_cf.append(np.zeros(model.layers[al].output_shape[1]))  # covered (activated) and failed
        num_uf.append(np.zeros(model.layers[al].output_shape[1]))  # uncovered (not activated) and failed
        num_cs.append(np.zeros(model.layers[al].output_shape[1]))  # covered and succeeded
        num_us.append(np.zeros(model.layers[al].output_shape[1]))  # uncovered and succeeded
        scores.append(np.zeros(model.layers[al].output_shape[1]))


    for al in available_layers:
        layer_idx = available_layers.index(al)
        all_neuron_idx = range(model.layers[al].output_shape[1]) 
        test_idx = 0
        for l in layer_outs[al][0]:
            covered_idx   = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx)-set(covered_idx))
            if test_idx  in correct_classifications:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassifications:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1
    ############################################
    ############################################

    filename = experiment_path + '/' + model_name + '_' + args['class'] + '_' +\
    args['approach'] +  '_SN' +  str(args['suspicious_num'])

    if args['approach'] == 'tarantula':
        try:
            suspicious_neuron_idx = load_dominant_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = tarantula_analysis(available_layers, scores, 
                                                 num_cf, num_uf, num_cs, num_us, 
                                                 int(args['suspicious_num']))


            save_dominant_neurons(suspicious_neuron_idx, filename, group_index)

    elif args['approach'] == 'ochiai':
        try:
            suspicious_neuron_idx = load_dominant_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = ochiai_analysis(available_layers, scores, 
                                                 num_cf, num_uf, num_cs, num_us, 
                                                 int(args['suspicious_num']))

            save_dominant_neurons(suspicious_neuron_idx, filename, group_index)

    elif args['approach'] == 'dstar':
        try:
            suspicious_neuron_idx = load_dominant_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = dstar_analysis(available_layers, scores, 
                                                 num_cf, num_uf, num_cs, num_us, 
                                                 int(args['suspicious_num']), 3)

            save_dominant_neurons(suspicious_neuron_idx, filename, group_index)


    elif args['approach'] == 'opposite':

        try:
            suspicious_neuron_idx = load_dominant_neurons(filename, group_index)
        except:
            _, scores = ochiai_analysis(correct_classifications,
                                        misclassifications, layer_outs,
                                        90) #temporary 90

            filename_ochiai = experiment_path + '/' + model_name + '_' + \
            str(args['class']) + '_ochiai_' + 'SN' + str(args['suspicious_num'])
            suspicious_neuron_idx_ochiai = load_dominant_neurons(filename_ochiai, group_index)

            available_layers = []
            filtered_scores = []
            for dom_ochiai in suspicious_neuron_idx_ochiai:
                if dom_ochiai[0] not in available_layers:
                    available_layers.append(dom_ochiai[0])
                    filtered_scores.append(scores[dom_ochiai[0]])

            suspicious_neuron_idx = find_indices(filtered_scores, 'lowest',
                                               int(args['suspicious_num']),
                                               available_layers)
            print suspicious_neuron_idx
            
            save_dominant_neurons(suspicious_neuron_idx, filename, group_index)

    elif args['approach'] == 'random':
        filename = experiment_path + '/' + model_name + '_' + args['class'] + \
        '_tarantula_' + 'SN' + str(args['suspicious_num'])

        suspicious_neuron_idx_tarantula = load_dominant_neurons(filename, group_index)

        filename = experiment_path + '/' + model_name + '_' + args['class'] + \
        '_ochiai_' + 'SN' + str(args['suspicious_num'])

        suspicious_neuron_idx_ochiai = load_dominant_neurons(filename, group_index)

        filename = experiment_path + '/' + model_name + '_' + args['class'] + \
        '_dstar_' + 'SN' + str(args['suspicious_num'])

        suspicious_neuron_idx_dstar = load_dominant_neurons(filename, group_index)


        forbiddens = suspicious_neuron_idx_ochiai + suspicious_neuron_idx_tarantula  + \
        suspicious_neuron_idx_dstar

        forbiddens = [list(forb) for forb in forbiddens]
        print forbiddens

        available_layers = list(set([elem[0] for elem in suspicious_neuron_idx_tarantula]))
        available_layers += list(set([elem[0] for elem in suspicious_neuron_idx_ochiai]))
        available_layers += list(set([elem[0] for elem in suspicious_neuron_idx_dstar]))
        
        suspicious_neuron_idx = []
        while len(suspicious_neuron_idx) < int(args['suspicious_num']):
            l_idx = random.choice(available_layers)
            n_idx = random.choice(range(model.layers[l_idx].output_shape[1]))

            if [l_idx, n_idx] not in forbiddens and [l_idx, n_idx] not in suspicious_neuron_idx:
                suspicious_neuron_idx.append([l_idx, n_idx])


    print suspicious_neuron_idx

    #dominant = {x: suspicious_neuron_idx[x - 1] for x in range(1, len(suspicious_neuron_idx) + 1)}
    logfile.write('Suspicous neurons: ' + str(suspicious_neuron_idx) + '\n')

    ####################
    # 4) Run Mutation Algorithm
    # Receive the set of dominant neurons for each layer from Step 3 # and will produce new inputs based on
    # the correct classifications (from the testing set) that exercise the
    # suspicious neurons

    perturbed_xs = []
    perturbed_ys = []


    #selct 10 inputs randomly from the correct classification set.
    zipped_data = random.sample(zip(list(np.array(X_val)[correct_classifications]),
                            list(np.array(Y_val)[correct_classifications])), 10)

    if args['mutate'] is None or args['mutate'] is True:
         start = datetime.datetime.now()
         x_perturbed, y_perturbed, x_original = mutate(model, zipped_data,
                                           suspicious_neuron_idx,
                                           correct_classifications,
                                           float(args['step_size']),
                                           float(args['distance']))
         end = datetime.datetime.now()
    else:
        filename = args['mutate'] + args['approach']
        filename = path.join(experiment_path, filename)
        x_perturbed, y_perturbed = load_perturbed_test_groups(filename, group_index)

    perturbed_xs = perturbed_xs + x_perturbed
    perturbed_ys = perturbed_ys + y_perturbed


    # reshape them into the expected format
    perturbed_xs = np.asarray(perturbed_xs).reshape(np.asarray(perturbed_xs).shape[0], 1, 28, 28)#
    perturbed_ys = np.asarray(perturbed_ys).reshape(np.asarray(perturbed_ys).shape[0], 10)#


    ####################
    #save perturtbed inputs
    filename = path.join(experiment_path, experiment_name)
    #filename = filename + '_layer' + str(layer)
    if args['mutate'] is None or args['mutate'] is True:
        save_perturbed_test_groups(perturbed_xs, perturbed_ys, filename, group_index)
        save_original_inputs(x_original, filename, group_index)

    score = model.evaluate(perturbed_xs, perturbed_ys, verbose=0)
    logfile.write('Model: ' + str(model_name) + ', Activation: ' +
                  args['activation'] + ', Class: ' + args['class'] +
                  ', Approach: ' + str(args['approach']) + ', Distance: ' +
                  str(args['distance']) + ', Score: ' +
                  str(score) + '\n')
    logfile.write('Mutation Time: ' + str(end-start) + '\n')

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

