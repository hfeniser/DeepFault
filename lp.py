from utils import load_data, load_model, get_layer_outs
import cplex
import numpy as np

target_neuron_layer = 2
target_neuron_index = 20

X_train, y_train, X_test, y_test = load_data()
model = load_model('mnist_test_model_2_100')

X = X_train[0]
X = np.expand_dims(X, axis=0)

layer_outs = get_layer_outs(model, X)

var_names=['d']
objective=[1]
lower_bounds=[0.0]
upper_bounds=[1.1]

for i in range(target_neuron_layer):
    for j in range(model.layers[i].output_shape[1]):
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

flatX = [item for sublist in X[0][0] for item in sublist]


for i in range(0, len(flatX)):
    # x<=x0+d
    constraints.append([[0, i+1], [-1, 1]])
    rhs.append(float(flatX[i]))
    constraint_senses.append("L")
    constraint_names.append("x<=x"+str(i)+"+d")
    # x>=x0-d
    constraints.append([[0, i+1], [1, 1]])
    rhs.append(float(flatX[i]))
    constraint_senses.append("G")
    constraint_names.append("x>=x"+str(i)+"-d")
    # x<=1
    constraints.append([[i+1], [1]])
    rhs.append(float(1.0))
    constraint_senses.append("L")
    constraint_names.append("x<=1")
    # x>=0
    constraints.append([[i+1], [1]])
    rhs.append(float(0.0))
    constraint_senses.append("G")
    constraint_names.append("x>=0")

for i in range(1, target_neuron_layer+1):
    for j in range(model.layers[i].output_shape[1]):
        if i == target_neuron_layer and not j == target_neuron_index: continue

        constraint = [[], []]

        for k in range(model.layers[i-1].output_shape[1]):
            constraint[0].append("x_"+str(i-1)+"_"+str(k))

            #for some reason 0th layer has no weights thus we say i instead of i-1
            constraint[1].append(float(model.layers[i].get_weights()[0][k][j]))

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

        # relu part
        _constraint=[[],[]]
        _constraint[0].append("x_"+str(i)+"_"+str(j))
        _constraint[1].append(1)
        constraints.append(_constraint)
        rhs.append(0)

        if layer_outs[i][0][0][j] > 0:
            constraint_senses.append("G")
        else:
            constraint_senses.append("L")

        constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))



#print constraints

print '--------'
print 'SETUP OK'
print '--------'


problem=cplex.Cplex()
problem.variables.add(obj = objective,
                  lb = lower_bounds,
                  ub = upper_bounds,
                  names = var_names)
problem.linear_constraints.add(lin_expr=constraints,
                           senses = constraint_senses,
                           rhs = rhs,
                           names = constraint_names)
problem.solve()

####
d=problem.solution.get_values("d")
new_x=[]
for i in range(0, len(flatX)):
    v=(problem.solution.get_values('x_0_'+str(i)))
    if v<0: print 'WRONG'
    new_x.append(v)

print new_x
