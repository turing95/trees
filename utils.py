import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error


def get_node_status(grb_model, b, beta, n):
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
    selected_feature = None

    if n in tree.Nodes:
        if (pruned == False) and (leaf == False):  # branching
            for f in grb_model.cat_features:
                if b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, value


def print_tree(grb_model, b, beta):
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
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta, n)
        print('#########node ', n)
        if pruned:
            print("pruned")
        elif branching:
            print(selected_feature)
        elif leaf:
            print('leaf {}'.format(value))


def get_predicted_value(grb_model, local_data, b, beta, i):
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
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta, current)
        if leaf:
            return value
        elif branching:
            if local_data.at[i, selected_feature] == 1:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)


def get_acc(grb_model, local_data, b, beta, p):
    '''
    This function returns the accuracy of the prediction for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The accuracy (fraction of datapoints which are correctly classified)
    '''
    label = grb_model.label
    acc = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, i)
        y_i = local_data.at[i, label]
        if yhat_i == y_i:
            acc += 1

    acc = acc / len(local_data.index)
    return acc


def get_mae(grb_model, local_data, b, beta, p):
    '''
    This function returns the MAE for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The MAE
    '''
    label = grb_model.label
    err = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, p, i)
        y_i = local_data.at[i, label]
        err += abs(yhat_i - y_i)

    err = err / len(local_data.index)
    return err


def get_mse(grb_model, local_data, b, beta, p):
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
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, p, i)
        y_i = local_data.at[i, label]
        err += np.power(yhat_i - y_i, 2)

    err = err / len(local_data.index)
    return err


def get_r_squared(grb_model, local_data, b, beta):
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
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, i)
        y_i = local_data.at[i, label]
        SS_Residuals += np.power(yhat_i - y_i, 2)
        SS_Total += np.power(y_bar - y_i, 2)

    R_squared = 1 - SS_Residuals / SS_Total
    return R_squared


def get_model_accuracy(data, datapoints, z, beta_zero, depth, label):
    y_trues = []
    y_preds = []
    tss_total = 0
    ess_total = 0
    median = np.median(data[label])
    mean = np.mean(data[label])

    for i in datapoints:
        max_value = -1
        node = None
        for t in range(1, np.power(2, depth + 1)):
            if max_value < z[i, t]:
                node = t
                max_value = z[i, t]
        y_trues.append(data.at[i, label])
        y_preds.append(beta_zero[node])
        tss_total += abs(data.at[i, label] - median)
    mae = mean_absolute_error(y_trues, y_preds)
    r2_lad = (mae * len(y_trues)) / tss_total
    return r2_score(y_trues, y_preds), mean_squared_error(y_trues, y_preds), mae, r2_lad, 1 - r2_lad, mean, median
