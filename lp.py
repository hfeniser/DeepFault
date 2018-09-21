from utils import get_layer_outs, show_image
import cplex
import numpy as np
from random import shuffle



def run_lp(model, X_val, Y_val, dominant, correct_classifications):
    """

    :param dominant:
    :return:
    """
    # Load MNIST data
    # x_train, y_train, x_test, y_test = load_data()

    x_perturbed = []
    y_perturbed = []

    # print(model.summary())

    # for all the testing set
    for test_index in range(0, len(X_val)):

        # if this testing input has been classified correctly, generate perturbations
        if test_index not in correct_classifications:
            continue

        # Here we get the first input from the testing set
        x = X_val[test_index]
        x = np.expand_dims(x, axis=0)

        # Flatten first input
        flatX = [item for sublist in x[0][0] for item in sublist]

        # What do we do here?
        layer_outs = get_layer_outs(model, x)

        # print("Dominant ", dominant)

        # Find max hidden layer with dominant neurons
        hidden_layers = [l for l in dominant.keys() if len(dominant[l])>0]
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

                if layer_outs[i][0][0][j] > 0 or j in dominant[i]:
                    constraint_senses.append("G")
                else:
                    constraint_senses.append("L")

                constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))

        ############################
        ############################
        # print('--------')
        # print('SETUP OK')
        # print('--------')

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
            x_perturbed.append(new_x)
            y_perturbed.append(Y_val[test_index])
            print("perturbation for ", test_index, " perturbed inputs", len(x_perturbed))

            # dims = int(np.sqrt(len(flatX)))
            # show_image(np.asarray(flatX).reshape(dims, dims))
            # show_image(np.asarray(new_x).reshape(dims, dims))

        if len(x_perturbed) >= 100:
            return x_perturbed, y_perturbed

    return x_perturbed, y_perturbed


def __run_cplex(model, dominant, target_layer, flatX, layer_outs):
    # dominant = {x: [] for x in range(1, len(dominant) + 1)}
    # target_layer = 10

    # Set the objective: d (the distance between the current and perturbed image) should be minimised
    var_names = ['d']
    objective = [1]
    lower_bounds = [0.0]
    upper_bounds = [1.0]

    # var_names = []
    # objective = []
    # lower_bounds = []
    # upper_bounds = []
    # for i in range(0, len(flatX)):
    #     var_names.append('d_' + str(i))
    #     objective.append(1)
    #     lower_bounds.append(0.0)
    #     upper_bounds.append(1.0)

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
            var_names.append('x_' + str(i) + '_' + str(j))
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
        x_name = "x_0_" + str(i)
        d_name = "d"
        # d_name = "d_" + str(i)
        # x<=x0+d
        # constraints.append([[0, i + 1], [-1, 1]])
        constraints.append([[d_name, x_name], [-1, 1]])
        rhs.append(float(flatX[i]))
        constraint_senses.append("L")
        constraint_names.append("x<=x" + str(i) + "+d")
        # x>=x0-d
        # constraints.append([[0, i + 1], [1, 1]])
        constraints.append([[d_name, x_name], [1, 1]])
        rhs.append(float(flatX[i]))
        constraint_senses.append("G")
        constraint_names.append("x>=x" + str(i) + "-d")
        # x<=1
        # constraints.append([[i + 1], [1]])
        constraints.append([[x_name], [1]])
        rhs.append(float(1.0))
        constraint_senses.append("L")
        constraint_names.append("x<=1")
        # x>=0
        # constraints.append([[i + 1], [1]])
        constraints.append([[x_name], [1]])
        rhs.append(float(0.0))
        constraint_senses.append("G")
        constraint_names.append("x>=0")

    # for all the hidden layers until the target layer (inclusive)
    for i in range(1, target_layer + 1):
        # for all the neurons in the i-th layer
        for j in range(model.layers[i].output_shape[1]):

            # ignore any neurons in the last hidden layer (target_layer)
            if i == target_layer and j not in dominant[target_layer]:
                continue

            constraint = [[], []]

            ###
            # for all the neurons in the i-1 layer
            for k in range(model.layers[i - 1].output_shape[1]):
                constraint[0].append("x_" + str(i - 1) + "_" + str(k))  # generate a constraint
                if i == 1 or layer_outs[i-1][0][0][k] > 0 or k in dominant[i-1]:
                    constraint[1].append(float(model.layers[i].get_weights()[0][k][j]))  # add its weight (multiplier)
                else:
                    constraint[1].append(0.0)

            # if the j-th neuron in the i-th layer is activated or dominant then its value should be equal to
            # W_(i-1,k)X_(i-1,k) = Xij
            if layer_outs[i][0][0][j] > 0 or j in dominant[i]:
                constraint[0].append("x_" + str(i) + "_" + str(j))
                constraint[1].append(-1)  # w00X00+w01x01+... -x11= 0
                constraint_senses.append("E")
                txt = ""
                if layer_outs[i][0][0][j] > 0:
                    txt += "act-"
                if j in dominant[i]:
                    txt += "dom"
                constraint_names.append(txt + ":" + "x_" + str(i) + "_" + str(j))
            else:
                constraint[0].append("x_" + str(i) + "_" + str(j))
                constraint[1].append(-1)
                constraint_senses.append("E")  # w00X00+w01x01+... -x22 <= 0, x22=0, w00X00+w01x01+...<= 0
                constraint_names.append("none:" + "x_" + str(i) + "_" + str(j))

            rhs.append(0.0)  # append rhs

            constraints.append(constraint)

            ###########################
            # relu
            relu_constraint = [[], []]
            relu_constraint[0].append("x_" + str(i) + "_" + str(j))  # x11 >= 0 || x11 <= 0
            relu_constraint[1].append(1)
            constraints.append(relu_constraint)

            txt = "relu-"
            if layer_outs[i][0][0][j] > 0 or j in dominant[i]:
                constraint_senses.append("G")
                if layer_outs[i][0][0][j] > 0:
                    txt += "act"
                    rhs.append(min(float(layer_outs[i][0][0][j]), 0.01))
                else:# j in dominant[i]:
                    txt += "dom-"
                    rhs.append(0.01)
            else:
                constraint_senses.append("L")  # x22=0
                rhs.append(0.0)

            constraint_names.append(txt + ":" + "x_" + str(i) + "_" + str(j))

    ############################
    ############################
    # print('--------')
    # print('SETUP OK')
    # print('--------')
    try:
        # Initialise Cplex problem
        problem = cplex.Cplex()

        cplex_file = "data/cplex.txt"
        problem.set_log_stream(cplex_file)
        problem.set_error_stream(cplex_file)
        problem.set_warning_stream(cplex_file)
        problem.set_results_stream(cplex_file)

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


        # Check solution status
        solution_status = problem.solution.get_status()

        if solution_status != 1:
            return None, None, solution_status

        # Get solution
        d = problem.solution.get_values("d")
        new_x = []
        # d_max = 0
        for i in range(0, len(flatX)):
            # d_name = "d_" + str(i)
            # d = problem.solution.get_values(d_name)
            # if d_max < d: d_max = d
            x_name = 'x_0_' + str(i)
            v = problem.solution.get_values(x_name)
            if v < 0 or v > 1 or d <= 0 or d > 1:
                print("WRONG: ", x_name, "\t:", v, "\t", "d", ":", d)
                return None, None, solution_status
            new_x.append(v)


        # print(d)
        # print(flatX)
        # print(new_x)

        return new_x, d, solution_status
    except Exception as e:
        import traceback
        traceback.print_exc(e)
        exit()


def run_lp_revised(model, X_val, Y_val, dominant, correct_classifications):
    """

    :param dominant:
    :return:
    """
    x_perturbed = []
    y_perturbed = []

    # print(model.summary())

    # Find max hidden layer with dominant neurons
    hidden_layers = [l for l in dominant.keys() if len(dominant[l]) > 0]
    target_layer = max(hidden_layers)

    # for all the testing set
    # indexes = np.arange(start, len(X_val))
    # shuffle(indexes)
    # for test_index in indexes:
    for test_index in range(0, len(X_val)):

        # if np.argmax(Y_val[test_index]) != class_index:
        #     continue

        # if this testing input has been classified correctly, generate perturbations
        if test_index not in correct_classifications:
            continue

        # Here we get the first input from the testing set
        x = X_val[test_index]
        x = np.expand_dims(x, axis=0)

        # Flatten first input
        flatX = [item for sublist in x[0][0] for item in sublist]

        # What do we do here?
        layer_outs = get_layer_outs(model, x)

        # setup and run cplex
        new_x, d, solution_status = __run_cplex(model, dominant, target_layer, flatX, layer_outs)

        # append perturbed input
        if new_x is not None:
            x_perturbed.append(new_x)
            y_perturbed.append(Y_val[test_index])
            print("perturbation for (", test_index, ")", np.where(Y_val[test_index] == Y_val[test_index].max())[0],
                  " perturbed inputs", len(x_perturbed), "\td:", d,
                  " status:\t", solution_status)

            # from test_nn import test_model
            # test_model(model,
            #            np.asarray(x_perturbed).reshape(np.asarray(x_perturbed).shape[0], 1, 28, 28),
            #            np.asarray(y_perturbed).reshape(np.asarray(y_perturbed).shape[0], 10))

            # dims = int(np.sqrt(len(flatX)))
            # show_image(np.asarray(flatX).reshape(dims, dims))
            # show_image(np.asarray(new_x).reshape(dims, dims))

            if len(x_perturbed) >= 1000:
                return x_perturbed, y_perturbed
        else:
            print("perturbation for ", test_index, " not found", ", status:\t", solution_status)

    #
    return x_perturbed, y_perturbed


if __name__ == "__main__":
    print("lp.py")
    # model_name = "neural_networks/mnist_test_model_10_10"
    # from utils import get_dummy_dominants, load_model
    # model = load_model(model_name)
    # dominant = get_dummy_dominants(model)
    # run_lp(model, None, None, dominant, None)



