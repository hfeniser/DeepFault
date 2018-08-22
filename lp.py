from utils import load_data, load_model, get_layer_outs
import cplex
import numpy as np

target_neuron_layer = 2
target_neuron_index = 20

X_train, y_train, X_test, y_test = load_data()
model = load_model('simple_mnist_fnn')

X = np.array(X_test[0])
X = X.expand_dims(img, axis=0)
layer_outs = get_layer_outs(model, X)

print model.layers[0].get_weights()
exit()

var_names=['d']
objective=[1]
lower_bounds=[0.0]
upper_bounds=[1.1]

for i in range(target_neuron_layer):
    for j in range(model.layers[i]):
        var_names.append('x_'+str(i)+'_'+str(j))
        objective.append(0)
        lower_bounds.append(-cplex.infinity)
        upper_bounds.append(cplex.infinity)

var_names.append('x_'+str(target_neuron_layer)+'_'+str(target_neuron_index))
objective.append(0)
lower_bounds.append(-cplex.infinity)
upper_bounds.append(cplex.infinity)


constraints=[]
rhs=[]
constraint_senses=[]
constraint_names=[]

for i in range(0, len(X)):
    # x<=x0+d
    constraints.append([[0, i+1], [-1, 1]])
    rhs.append(X[i])
    constraint_senses.append("L")
    constraint_names.append("x<=x"+str(i)+"+d")
    # x>=x0-d
    constraints.append([[0, i+1], [1, 1]])
    rhs.append(X[i])
    constraint_senses.append("G")
    constraint_names.append("x>=x"+str(i)+"-d")
    # x<=1
    constraints.append([[i+1], [1]])
    rhs.append(1.0)
    constraint_senses.append("L")
    constraint_names.append("x<=1")
    # x>=0
    constraints.append([[i+1], [1]])
    rhs.append(0.0)
    constraint_senses.append("G")
    constraint_names.append("x>=0")


for i in range(1, target_neuron_layer+1):
    for j in range(model.layers[i]):

        if i == target_neuron_layer and not j == target_neuron_index: continue

        constraint = [[], []]

        for k in range(model.layers[i-1]):
            constraint[0].append("x_"+str(i-1)+"_"+str(k))
            constraint[1].append(model.layer[i-1].get_weights()[k][j])

        rhs.append(0)
        if not (i == target_neuron_layer and j == target_neuron_index):
            constraint[0].append("x_"+str(i)+"_"+str(j))
            constraint[1].append(-1)
            constraint_senses.append("E")
            constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))
        else:
            constraint_senses.append("G")
            constraint_names.append("act:"+"x_"+str(i)+"_"+str(j))

        constraints.append(constraint)



