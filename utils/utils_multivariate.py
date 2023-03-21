import numpy as np


def get_node_status(grb_model, a, b, beta_zero, n, i, data, beta=None):
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
    pruned = False
    branching = False
    leaf = False
    value = None
    a_x_n = None
    b_n = None

    if n in grb_model.tree.Leaves:  # leaf
        leaf = True
        value = beta_zero[n]
        if beta is not None:
            value += sum(beta[n, f] * data.at[i, f] for f in grb_model.cat_features)

    if n in tree.Nodes:
        if (pruned == False) and (leaf == False):  # branching
            a_x_n = sum(a[n, f] * data.at[i, f] for f in grb_model.cat_features)
            b_n = b[n]
            branching = True
    return pruned, branching, a_x_n, b_n, leaf, value


def get_predicted_value(grb_model, local_data, a, b, beta_zero, i, beta, g):
    tree = grb_model.tree
    current = 1
    while True:
        pruned, branching, a_n_f, b_n, leaf, value = get_node_status(grb_model, a, b, beta_zero, current, i,
                                                                     local_data, beta)
        if leaf:
            return value
        elif branching:
            if a_n_f > b_n:  # going right on the branch

                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)


'''def get_predicted_value(grb_model, local_data, a, b, beta_zero, i, beta, g):
    for n in grb_model.tree.Leaves:
        if g[i, n] > 0.5:
            return beta_zero[n] + sum(beta[n, f] * local_data.at[i, f] for f in grb_model.cat_features)'''


def get_res_err(grb_model, local_data, a, b, beta_zero, g, beta=None):
    label = grb_model.label
    res_err = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, a, b, beta_zero, i, beta, g)
        y_i = local_data.at[i, label]
        res_err += abs(yhat_i - y_i)
    return res_err


def get_mae(grb_model, local_data, a, b, beta_zero, g, beta=None):
    '''
    This function returns the MAE for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The MAE
    '''
    res_err = get_res_err(grb_model, local_data, a, b, beta_zero, g, beta)

    err = res_err / len(local_data.index)
    return err


def get_mse(grb_model, local_data, a, b, beta_zero, beta=None):
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
        yhat_i = get_predicted_value(grb_model, local_data, a, b, beta_zero, i, beta)
        y_i = local_data.at[i, label]
        err += np.power(yhat_i - y_i, 2)

    err = err / len(local_data.index)
    return err


def get_r_squared(grb_model, local_data, a, b, beta_zero, g, beta=None):
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
        yhat_i = get_predicted_value(grb_model, local_data, a, b, beta_zero, i, beta, g)
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


def get_model_accuracy(model, data, a, b, beta_zero, beta, g):
    mae = get_mae(model, data, a, b, beta_zero, g, beta)

    r2 = get_r_squared(model, data, a, b, beta_zero, g, beta)

    return mae, r2
