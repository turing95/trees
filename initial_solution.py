from max_cut_node_means_pca import max_cut_node_means_pca_bottom_up
from linear_svc import linear_svc, linear_svc_alternative
from Tree import Tree
import statistics
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_a_x_n(data, i, initial_a_b, current):
    a_x_n = 0
    for idx, a in enumerate(initial_a_b[current - 1][0]):
        '''print(idx)
        print(a)
        print(data.iloc[i, idx])
        print('\n')'''

        feature_value = data.iloc[i, idx]
        a_x_n += a * feature_value
    return a_x_n


def get_predicted_value(data, i, tree, initial_a_b):
    current = 1
    while True:
        print(current)
        print('\n\n')
        if current in tree.Leaves:

            return current, data.at[i, 'target']
        else:
            a_x_n = get_a_x_n(data, i, initial_a_b, current)

            b_n = initial_a_b[current - 1][1]
            if a_x_n + b_n > 0:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)


def get_beta_zeros(data, tree: Tree, initial_a_b):
    leaves_predictions = {}
    intercept = []
    coef = []
    median = []
    for leaf in tree.Leaves:
        leaves_predictions[leaf] = {'y': [], 'x': [], 'indexes': []}

    for i in data.index:
        node, value = get_predicted_value(data, i, tree, initial_a_b)
        leaves_predictions[node]['y'].append(value)
        leaves_predictions[node]['x'].append(data.iloc[i, :-1])
        leaves_predictions[node]['indexes'].append(i)
    for k, v in leaves_predictions.items():

        if v['y']:
            # median.append(statistics.median(v['y']))
            lg = LinearRegression()
            lg.fit(v['x'], v['y'])
            coef.append(lg.coef_)
            intercept.append(lg.intercept_)
        else:
            coef.append(None)
            intercept.append(None)
            # median.append(None)

    print('\n')
    return coef, intercept, leaves_predictions


def get_e_i_n(data, tree, beta, beta_zero, predictions):
    e = [[0 for n in tree.Leaves] for i in data.index]
    for i in data.index:
        for n in tree.Leaves:
            if i in predictions[n]['indexes']:
                leaf_shifted_index = n - 2 ** tree.depth
                e[i][leaf_shifted_index] = data.at[i, 'target'] - (beta_zero[leaf_shifted_index]+sum(beta[leaf_shifted_index][data.columns.get_loc(f)]*data.at[i,f] for f in data.columns[:-1]))

    return e


def get_initial_solution(data, tree):
    initial_a_b, clusters = linear_svc_alternative(data.copy(), tree.depth)
    init_beta, init_beta_zero, leaves_predictions = get_beta_zeros(data.copy(), tree, initial_a_b)
    init_e_i_n = get_e_i_n(data, tree, init_beta, init_beta_zero, leaves_predictions)
    return init_beta, init_beta_zero, initial_a_b,init_e_i_n


if __name__ == "__main__":
    dataframe = pd.read_csv('./DataSets/airfoil_self_noise_reg.csv')
    depth = 1
    b, b_z, a_b,e = get_initial_solution(dataframe, Tree(depth))
    print('\n')
