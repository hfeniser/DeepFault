
from utils import get_layer_outs, load_model, load_data
from utils import save_layer_outs, save_classifications
from utils import load_layer_outs, load_classifications
from utils import filter_val_set, get_trainable_layers
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from scipy import stats
from imblearn.over_sampling import SMOTE
from test_nn import test_model
from os import path
import numpy as np
import random 
import argparse

seed = random.randint(0,10)

experiment_path = 'experiment_results'
model_path  = 'neural_networks'
model_name  = 'mnist_test_model_5_30_leaky_relu'
group_index = 1

def parse_arguments():

    # initiate the parser
    parser = argparse.ArgumentParser()

    # add new command-line arguments
    parser.add_argument("-C", "--class", type=int)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":

    args = parse_arguments()

    selected_class = args["class"]

    X_train, Y_train, X_test, Y_test = load_data()

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                          test_size=1/6.0,
                                                          random_state=seed)

    X_val, Y_val = filter_val_set(selected_class, X_val, Y_val)

    try:
        model = load_model(path.join(model_path, model_name))
    except:
        print("Model not found! Provide a pre-trained model model as input.")
        exit(1)


    filename = experiment_path + '/' + model_name + '_' + str(selected_class)
    try:
        correct_classifications, misclassifications = load_classifications(filename, group_index)
        layer_outs = load_layer_outs(filename, group_index)
    except:
        correct_classifications, misclassifications, layer_outs, predictions = test_model(model, X_val, Y_val)
        save_classifications(correct_classifications, misclassifications, filename, group_index)
        save_layer_outs(layer_outs, filename, group_index)


    trainable_layers = get_trainable_layers(model)

    ###############################
    ### LAYER-BY-LAYER ANALYSIS ###
    ###############################
    for tl in trainable_layers:
        louts = layer_outs[tl][0]
        lc = louts[correct_classifications][0]
        lm = louts[misclassifications][0]
        print("DISTANCE: " + str(np.linalg.norm(lc-lm)))


    exit()

    ############################
    ### CLUSTERING ANALYSSIS ###
    ############################
    penultimate_outs = layer_outs[trainable_layers[-5]][0]
    labels = np.zeros(len(penultimate_outs))
    labels[misclassifications] = 1
    sm = SMOTE(random_state = seed)
    penultimate_outs, _ = sm.fit_sample(penultimate_outs, labels)

    clustering = KMeans(n_clusters = 2).fit(penultimate_outs) # there are 2 clusters: correct and mis-classifications

    #clustering = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
    #                                     linkage='ward').fit(penultimate_outs)

    #clustering = SpectralClustering(n_clusters=2, assign_labels='discretize',
    #                        random_state=0, affinity='rbf').fit(penultimate_outs)

    unique, counts = np.unique(clustering.labels_, return_counts = True)
    print(dict(zip(unique,counts)))


    #########################################
    ### STATISTICAL SIGNIFICANCE ANALYSIS ###
    #########################################
    print("CLASS: " + str(selected_class))
    for tl in trainable_layers:
        louts = layer_outs[tl][0]

        diff_cnt = 0
        for i in range(len(louts[0])):
            _, p_val = stats.ttest_ind(louts[correct_classifications][:,i],
                                       louts[misclassifications][:,i])
            if p_val < 0.05:
                diff_cnt += 1

        #    print(np.mean(penultimate_outs[correct_classifications][440:], axis=0))[i]
        #    print(np.mean(penultimate_outs[correct_classifications][:440], axis=0))[i]
        #    print("=====")

        print("LAYER " + str(tl) + ": " + str(diff_cnt))

    print("+++++")
    print("")


