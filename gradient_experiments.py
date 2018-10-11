from utils import load_model, find_class_of, load_data, load_classifications, load_model, load_dominant_neurons, save_perturbed_test_groups
from perturbe_via_gradient import perturbe
from sklearn.model_selection import train_test_split
import numpy as np
import random
import datetime

seed = 7
group_index = 1

X_train, Y_train, X_test, Y_test = load_data()
X_train, X_val, Y_train, Y_val   = train_test_split(X_train, Y_train, test_size=1/6.0, random_state=seed)
X_val, Y_val                     = find_class_of(X_val, Y_val, 3)


model_names    = ['3_50', '5_30', '8_20'] 
percents  = [95, 90]
distances = [0.2, 0.1, 0.05, 0.01]
log_file = open('data/experiment_results.log', 'a')

for model_name in model_names:
    print(model_name)
    model = load_model('neural_networks/mnist_test_model_' + str(model_name))
    layers = range(1, len(model.layers)-1)
    dominants = load_dominant_neurons('data/' + str(model_name) + '/tarantula', group_index)
    correct_classifications, _ = load_classifications('data/' + str(model_name) + '/' + str(model_name), group_index)

    for percent in percents:
        for d in distances:
            for layer in layers[0::2]:

                log_file = open('data/experiment_results.log', 'a')
                print(layer)
                dominant_indices = dominants[layer]
                print(dominant_indices)
                print(d)
                if not list(dominant_indices):
                    log_file.write('No dominant for model ' + str(model_name) + ' for layer ' + str(layer) + ' with faulty percent ' + str(percent) + '.\n')
                    continue

                ###########################
                ###Test Actual Dominants###
                ###########################
                start = datetime.datetime.now()
                perturbed_xs, perturbed_ys = perturbe(model, X_val, Y_val, layer, dominant_indices, correct_classifications, d) 
                end = datetime.datetime.now()
                print('It lasted: ' + str(end-start))
                perturbed_xs = np.asarray(perturbed_xs).reshape(np.asarray(perturbed_xs).shape[0], 1, 28, 28)#
                perturbed_ys = np.asarray(perturbed_ys).reshape(np.asarray(perturbed_ys).shape[0], 10)

                perturb_file = 'data/' + str(model_name) + '/' + str(model_name) + '_' + str(layer) + '_' + str(d) 
                save_perturbed_test_groups(perturbed_xs, perturbed_ys,  perturb_file, group_index)

                score = model.evaluate(perturbed_xs, perturbed_ys, verbose=0)
                log_file.write('Accuracy for model ' + str(model_name) + ' for layer ' + str(layer) + ' with faulty percent ' + str(percent) + ' and distance ' + str(d) + ': ' + str(score) + ' (Actual Dominants).\n')

                #############################
                ####Test Random Dominants####
                #############################
                #select randoms from out of all neurons except from actual faulties.

                random_dominants = random.sample([e for e in range(model.layers[layer].output_shape[1]) if e not in dominant_indices], len(dominant_indices))
                print(random_dominants)
                start = datetime.datetime.now()
                perturbed_xs, perturbed_ys = perturbe(model, X_val, Y_val, layer, random_dominants, correct_classifications, d)
                end = datetime.datetime.now()
                print('It lasted: ' + str(end-start))

                perturbed_xs = np.asarray(perturbed_xs).reshape(np.asarray(perturbed_xs).shape[0], 1, 28, 28)#
                perturbed_ys = np.asarray(perturbed_ys).reshape(np.asarray(perturbed_ys).shape[0], 10)

                perturb_file += '_random'
                save_perturbed_test_groups(perturbed_xs, perturbed_ys,  perturb_file, group_index)

                score = model.evaluate(perturbed_xs, perturbed_ys, verbose=0)
                log_file.write('Accuracy for model ' + str(model_name) + ' for layer ' + str(layer) + ' with faulty percent ' + str(percent) + ' and distance ' + str(d) + ': ' + str(score) + '(Random Dominants).\n\n')

                log_file.close() 
        exit()


