from utils import load_data, load_model, get_layer_outs, show_image
import cplex
import numpy as np


def run_lp(model=None, dominant=None, correct_classifications=None):
    """

    :param dominant:
    :return:
    """
    # Load MNIST data
    x_train, y_train, x_test, y_test = load_data()

    x_perturbed = []
    y_perturbed = []

    # print(model.summary())

    # for all the testing set
    for test_index in range(0, len(x_test)):

        # if this testing input has been classified correctly, generate perturbations
        if test_index not in correct_classifications:
            continue

        # Here we get the first input from the testing set
        x = x_test[test_index]
        x = np.expand_dims(x, axis=0)

        # Flatten first input
        flatX = [item for sublist in x[0][0] for item in sublist]

        # What do we do here?
        layer_outs = get_layer_outs(model, x)

        # print("Dominant ", dominant)

        # Find max hidden layer with dominant neurons
        hidden_layers = [l for l in dominant.keys() if dominant[l]]
        target_layer = max(hidden_layers)

        # Set the objective: d (the distance between the current and perturbed image) should be minimised
        var_names = ['d']
        objective = [1]
        lower_bounds = [0.0]
        upper_bounds = [1.0]

        # Initialise the constraints
        constraints = []
        rhs = []
        constraint_senses = []
        constraint_names = []

        # add objectives for all neurons until the target layer
        # we have an objective function like the following
        # MIN (d + 0x_00 + 0x_01 + ... + 0x_14)
        # this occurs only to define the var names
        for i in range(target_layer):
            for j in range(model.layers[i].output_shape[1]):
                var_names.append('x_'+str(i)+'_'+str(j))
                objective.append(0)
                lower_bounds.append(-cplex.infinity)
                upper_bounds.append(cplex.infinity)

        # as above, define variables for all neurons in target layer
        for target_neuron_index in dominant[target_layer]:
            var_names.append('x_' + str(target_layer) + '_' + str(target_neuron_index))
            objective.append(0)
            lower_bounds.append(-cplex.infinity)
            upper_bounds.append(cplex.infinity)

        # Add constraints for the input (e.g., 28x28)
        # It should be very similar to the input image (d should be minimised)
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

        # for all the hidden layers until the target layer (inclusive)
        for i in range(1, target_layer + 1):
            for j in range(model.layers[i].output_shape[1]):

                # ignore any neurons in the last hidden layer (target_layer)
                if i == target_layer and j not in dominant[target_layer]:
                    continue

                constraint = [[], []]

                ###
                for k in range(model.layers[i-1].output_shape[1]):
                    constraint[0].append("x_"+str(i-1)+"_"+str(k))

                    if layer_outs[i][0][0][j] > 0 or j in dominant[i]:
                        # for some reason 0th layer has no weights thus we say i instead of i-1
                        constraint[1].append(float(model.layers[i].get_weights()[0][k][j]))
                    else:
                        constraint[1].append(0)

                rhs.append(0)
                if j not in dominant[i]:  # not (i == target_layer and j == target_neuron_index):
                    constraint[0].append("x_"+str(i)+"_"+str(j))
                    constraint[1].append(-1)      # deactivated X11==> 0X00+001x01+... -x11= 0 ==> x11=0
                    constraint_senses.append("E")  # activ X11  ==> w00X00+w01x01+...-x11 = 0==>w00X00+w01x01+...= x11
                    constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))
                else:
                    # if it among the dominant neurons, we ignore completely the previous value of the
                    # neuron (i.e., x_ij). Idea for future work --> neuron boundary
                    constraint_senses.append("G")  # w00X00+w01x01+.. >= 0;
                    constraint_names.append("act:"+"x_"+str(i)+"_"+str(j))

                constraints.append(constraint)

                ###########################
                # relu
                relu_constraint = [[], []]
                relu_constraint[0].append("x_" + str(i) + "_" + str(j))  # x11 >= 0 || x11 <= 0
                relu_constraint[1].append(1)
                constraints.append(relu_constraint)
                rhs.append(0)

                if layer_outs[i][0][0][j] > 0:
                    constraint_senses.append("G")
                else:
                    constraint_senses.append("L")

                constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))

        ############################
        ############################
        print('--------')
        print('SETUP OK')
        print('--------')

        # Initialise Cplex problem
        problem = cplex.Cplex()

        # Default sense is minimisation
        problem.objective.set_sense(problem.objective.sense.minimize)

        # Add variables
        problem.variables.add(obj=objective,
                              lb=lower_bounds,
                              ub=upper_bounds,
                              names=var_names)

        # Add constraints
        problem.linear_constraints.add(lin_expr=constraints,
                                       senses=constraint_senses,
                                       rhs=rhs,
                                       names=constraint_names)

        # Solve the problem
        problem.solve()

        # Get solution
        d = problem.solution.get_values("d")
        new_x = []
        for i in range(0, len(flatX)):
            v = (problem.solution.get_values('x_0_'+str(i)))
            if v < 0:
                print('WRONG')
            new_x.append(v)

        print(d)
        print(flatX)
        print(new_x)


        # append perturbed input
        if (d>0 and d<1):
            print("perturbation for ", test_index)
            x_perturbed.append(new_x)
            y_perturbed.append(y_test[test_index])

            dims = int(np.sqrt(len(flatX)))
            show_image(np.asarray(flatX).reshape(dims, dims))
            show_image(np.asarray(new_x).reshape(dims, dims))

        if len(x_perturbed) > 5:
            return x_perturbed, y_perturbed

def run_lp_old():
    """
    :return:
    """
    # This is the set of dominant neurons per layer
    # They will be identified by the DNN-fault localisation approach
    target_layer = 2
    target_neuron_index = 4

    # Load MNIST data
    X_train, y_train, X_test, y_test = load_data()

    # Load saved model
    model = load_model('neural_networks/mnist_test_model_5_10')

    # Here we get the first input from the testing set
    X = X_train[1]
    X = np.expand_dims(X, axis=0)

    # What do we do here?
    layer_outs = get_layer_outs(model, X)

    var_names = ['d']
    objective = [1]
    lower_bounds = [0.0]
    upper_bounds = [1.0]

    print(model.summary())

    # why do we add 0s in all neurons until the target layer(s)
    # we have an objective function like the following
    # MIN (d + 0x_00 + 0x_01 + ... + 0x_14)
    for i in range(target_layer):
        for j in range(model.layers[i].output_shape[1]):
            var_names.append('x_'+str(i)+'_'+str(j))
            objective.append(0)
            lower_bounds.append(-cplex.infinity)
            upper_bounds.append(cplex.infinity)

    var_names.append('x_' + str(target_layer) + '_' + str(target_neuron_index))
    objective.append(0)
    lower_bounds.append(-cplex.infinity)
    upper_bounds.append(cplex.infinity)

    constraints = []
    rhs = []
    constraint_senses = []
    constraint_names = []

    # Flatten first input
    flatX = [item for sublist in X[0][0] for item in sublist]

    # Add constraints for the input (e.g., 28x28)
    # It should be very similar to the input image (d should be minimised)
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

    # for all the hidden layers until the target layer (inclusive)
    for i in range(1, target_layer + 1):
        for j in range(model.layers[i].output_shape[1]):
            #ignore all neurons in target layer
            if i == target_layer and not j == target_neuron_index:
                continue

            constraint = [[], []]

            for k in range(model.layers[i-1].output_shape[1]):
                constraint[0].append("x_"+str(i-1)+"_"+str(k))

                if layer_outs[i][0][0][j] > 0 or (i == target_layer and j == target_neuron_index):
                    # for some reason 0th layer has no weights thus we say i instead of i-1
                    constraint[1].append(float(model.layers[i].get_weights()[0][k][j]))
                else:
                    constraint[1].append(0)

            rhs.append(0)
            if not (i == target_layer and j == target_neuron_index):
                constraint[0].append("x_"+str(i)+"_"+str(j))
                constraint[1].append(-1)      # deactivated X11==> 0X00+001x01+... -x11= 0 ==> x11=0
                constraint_senses.append("E")  # activated X11  ==> w00X00+w01x01+...-x11 = 0 ==> w00X00+w01x01+...= x11
                constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))
            else:
                constraint_senses.append("G")  # w00X00+w01x01+.. >= 0;
                constraint_names.append("act:"+"x_"+str(i)+"_"+str(j))

            constraints.append(constraint)

            # relu part
            _constraint = [[], []]
            _constraint[0].append("x_"+str(i)+"_"+str(j))  # x11 >= 0 || x11 <= 0
            _constraint[1].append(1)
            constraints.append(_constraint)
            rhs.append(0)

            if layer_outs[i][0][0][j] > 0:
                constraint_senses.append("G")
            else:
                constraint_senses.append("L")

            constraint_names.append("relu:" + "x_"+str(i)+"_"+str(j))

    # print constraints
    print('--------')
    print('SETUP OK')
    print('--------')

    # Initialise Cplex problem
    problem = cplex.Cplex()

    # Default sense is minimisation
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Add variables
    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=var_names)

    # Add constraints
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses=constraint_senses,
                                   rhs=rhs,
                                   names=constraint_names)

    # Solve the problem
    problem.solve()

    ####
    d = problem.solution.get_values("d")
    new_x = []
    for i in range(0, len(flatX)):
        v = (problem.solution.get_values('x_0_'+str(i)))
        if v < 0:
            print('WRONG')
        new_x.append(v)

    print(d)
    print(flatX)
    print(new_x)

    dims = int(np.sqrt(len(flatX)))

    show_image(np.asarray(flatX).reshape(dims, dims))
    show_image(np.asarray(new_x).reshape(dims, dims))


if __name__ == "__main__":
    model_name = "neural_networks/mnist_test_model_5_5"
    from utils import get_dummy_dominants
    dominant = get_dummy_dominants(model_name)
    run_lp(model_name, dominant)



