
from utils import get_layer_outs, load_model, load_data
from utils import save_layer_outs, save_classifications
from utils import load_layer_outs, load_classifications
from utils import filter_val_set, get_trainable_layers
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from test_nn import test_model
from os import path
import numpy as np
import random 

seed = random.randint(0,10)

experiment_path = 'experiment_results'
model_path  = 'neural_networks'
model_name  = 'mnist_test_model_5_30_leaky_relu'
group_index = 1
selected_class = 0

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

#penultimate_outs = layer_outs[trainable_layers[-1]][0]
penultimate_outs = layer_outs[10][0]

print(penultimate_outs[0])
print(penultimate_outs[1])

kmeans = KMeans(n_clusters = 2) # there are 2 clusters: correct and mis-classifications
kmeans.fit(penultimate_outs)
unique, counts = np.unique(kmeans.labels_, return_counts = True)
print(dict(zip(unique,counts)))
