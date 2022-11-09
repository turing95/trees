import numpy as np


def get_node_status(grb_model, b, beta_zero, p, n, beta=None):
    '''
    This function give the status of a given node in a tree. By status we mean whether the node
        1- is pruned? i.e., we have made a prediction at one of its ancestors
        2- is a branching node? If yes, what feature do we branch on
        3- is a leaf? If yes, what is the prediction at this node?

    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :param n: A valid node index in the tree
    :return: pruned, branching, selected_feature, leaf, value

    pruned=1 iff the node is pruned
    branching = 1 iff the node branches at some feature f
    selected_feature: The feature that the node branch on
    leaf = 1 iff node n is a leaf in the tree
    value: if node n is a leaf, value represent the prediction at this node
    '''
    tree = grb_model.tree
    mode = grb_model.mode
    pruned = False
    branching = False
    leaf = False
    value = None
    selected_feature = None

    p_sum = 0
    for m in tree.get_ancestors(n):
        p_sum = p_sum + p[m]
    if p[n] > 0.5:  # leaf
        leaf = True
        if mode == "regression":
            value = beta_zero[n, 1]
            if beta is not None:
                value += sum(beta[n, f] for f in grb_model.cat_features)
        elif mode == "classification":
            for k in grb_model.labels:
                if beta_zero[n, k] > 0.5:
                    value = k
    elif p_sum == 1:  # Pruned
        pruned = True

    if n in tree.Nodes:
        if (pruned == False) and (leaf == False):  # branching
            for f in grb_model.cat_features:
                if b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, value


def print_tree(grb_model, b, beta, p):
    '''
    This function print the derived tree with the branching features and the predictions asserted for each node
    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :return: print out the tree in the console
    '''
    tree = grb_model.tree
    for n in tree.Nodes + tree.Leaves:
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta, p, n)
        print('#########node ', n)
        if pruned:
            print("pruned")
        elif branching:
            print(selected_feature)
        elif leaf:
            print('leaf {}'.format(value))


def get_predicted_value(grb_model, local_data, b, beta_zero, p, i, beta):
    '''
    This function returns the predicted value for a given datapoint
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :param i: Index of the datapoint we are interested in
    :return: The predicted value for datapoint i in dataset local_data
    '''
    tree = grb_model.tree
    current = 1

    while True:
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta_zero, p, current, beta)
        if leaf:
            return value
        elif branching:
            if local_data.at[i, selected_feature] == 1:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)


def get_res_err(grb_model, local_data, b, beta_zero, p, beta=None):
    label = grb_model.label
    res_err = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta_zero, p, i, beta)
        y_i = local_data.at[i, label]
        res_err += abs(yhat_i - y_i)
    return res_err


def get_mae(grb_model, local_data, b, beta_zero, p, beta=None):
    '''
    This function returns the MAE for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The MAE
    '''
    res_err = get_res_err(grb_model, local_data, b, beta_zero, p, beta)

    err = res_err / len(local_data.index)
    return err


def get_mse(grb_model, local_data, b, beta_zero, p, beta=None):
    '''
    This function returns the MSE for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The MSE
    '''
    label = grb_model.label
    err = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta_zero, p, i, beta)
        y_i = local_data.at[i, label]
        err += np.power(yhat_i - y_i, 2)

    err = err / len(local_data.index)
    return err


def get_r_squared(grb_model, local_data, b, beta_zero, p, beta=None):
    '''
    This function returns the R^2 for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The R^2
    '''
    label = grb_model.label
    R_squared = 0
    y_bar = local_data[label].mean()
    # print(y_bar)
    SS_Residuals = 0
    SS_Total = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta_zero, p, i, beta)
        y_i = local_data.at[i, label]
        SS_Residuals += np.power(yhat_i - y_i, 2)
        SS_Total += np.power(y_bar - y_i, 2)

    R_squared = 1 - SS_Residuals / SS_Total
    return R_squared


def get_r_lad(label, local_data, mae):
    '''
    This function returns the R^2 for a given dataset
    :param mae:
    :param label:
    :param local_data: The dataset we want to compute accuracy for
    :return: The R^2
    '''
    median = local_data[label].median()
    tss_total = 0
    y_trues = []
    for i in local_data.index:
        tss_total += abs(local_data.at[i, label] - median)
        y_trues.append(local_data.at[i, label])

    r2_lad = (mae * len(y_trues)) / tss_total
    return r2_lad


def get_model_accuracy(model, data, b, beta_zero, p, beta):
    err = get_res_err(model, data, b, beta_zero, p, beta)
    mae = get_mae(model, data, b, beta_zero, p, beta)

    mse = get_mse(model, data, b, beta_zero, p, beta)

    r2 = get_r_squared(model, data, b, beta_zero, p, beta)

    r2_lad_alt = 1 - get_r_lad(model.label, data, mae)
    return err, mae, mse, r2, r2_lad_alt
