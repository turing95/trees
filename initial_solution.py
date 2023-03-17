from max_cut_node_means_pca import max_cut_node_means_pca_bottom_up
from linear_svc import linear_svc, linear_svc_alternative
from Tree import Tree
import statistics
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold


def get_a_x_n(data, i, initial_a_b, current):
    curr_a_b = initial_a_b[current]
    a_x_n = 0
    for idx, a in enumerate(curr_a_b[0]):

        a_x_n += a * data.at[i, data.columns[idx]]
    return a_x_n


def get_predicted_value(data, i, tree, initial_a_b):
    current = 1
    while True:
        if current in tree.Leaves:

            return current, data.at[i, 'target']
        else:
            a_x_n = get_a_x_n(data, i, initial_a_b, current)

            b_n = initial_a_b[current][1]
            #TODO switch sign for linear_svc
            if a_x_n - b_n > 0:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)
    '''lsvc = initial_a_b[1][-1]
    return lsvc.predict([data.loc[i]])[0], data.at[i, 'target']'''


def get_beta_zeros(data, tree: Tree, initial_a_b, clusters):
    leaves_predictions = {}
    for leaf in tree.Leaves:
        leaves_predictions[leaf] = {'y': [], 'x': [], 'indexes': [], 'indexes_not_in_cluster': 0,
                                    'cluster': clusters[leaf]['cluster']}

    for i in data.index:
        node, value = get_predicted_value(data, i, tree, initial_a_b)
        leaves_predictions[node]['y'].append(value)
        leaves_predictions[node]['x'].append(data.loc[i, [c for c in data.columns if c != data.columns[-1]]])
        leaves_predictions[node]['indexes'].append(i)
        if i not in clusters[node]['cluster'].index:
            leaves_predictions[node]['indexes_not_in_cluster'] += 1
    for k, v in leaves_predictions.items():

        if v['y']:
            # median.append(statistics.median(v['y']))
            lg = Ridge()
            lg.fit(v['x'], v['y'])

            v['coef'] = lg.coef_
            v['intercept'] = lg.intercept_
            v['score'] = lg.score(v['x'], v['y'])
            # median.append(None)

    return leaves_predictions


def get_e_g_i_n(data, tree, predictions):
    e = {}
    g = {}
    for i in data.index:
        e[i] = {}
        g[i] = {}
    for i in data.index:

        for n in tree.Leaves:
            v = predictions[n]
            if i in v['indexes']:
                try:
                    beta_z = v['intercept']
                    beta_i = sum(
                        v['coef'][data.columns.get_loc(f)] * data.at[i, f] for f in data.columns[:-1])
                    temp_e = data.at[i, 'target'] - (beta_z + beta_i)
                    e[i][n] = temp_e
                    g[i][n] = 1
                except Exception as exc:
                    print(i)
                    print(beta_z)
                    print(beta_i)
                    raise exc
            else:
                g[i][n] = 0

        if i == 628:
            pass
    return e, g


def check_leaves_clusters(leaves_predictions, clusters):
    mismatches = 0
    for k, v in leaves_predictions.items():
        indexes = v['indexes']
        for i in indexes:
            if i not in clusters[k]['cluster'].index:
                print(f'{i} not in cluster {k}')
                mismatches += 1
    print(mismatches)


def get_initial_solution(data, tree, check=False):
    initial_a_b, clusters = max_cut_node_means_pca_bottom_up(data.copy(), tree.depth)
    leaves_predictions = get_beta_zeros(data.copy(), tree, initial_a_b, clusters)
    if check is True:
        check_leaves_clusters(leaves_predictions, clusters)
    init_e_i_n, init_g = get_e_g_i_n(data, tree, leaves_predictions)
    return leaves_predictions, initial_a_b, init_e_i_n, init_g, clusters


def test_initial_solution_k_fold(data, tree):
    x = data.iloc[:, :-1]
    k_folds = 5
    random_state = 1
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    n_k_folds = kf.get_n_splits(data)
    rs = []
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        rs.append(get_initial_solution(data_train, tree))
    return rs


if __name__ == "__main__":
    dataframe = pd.read_csv('./DataSets/airfoil_self_noise_reg.csv')
    depth = 1
    l, a_b, e_i, g_i_n, cl = get_initial_solution(dataframe, Tree(depth), check=False)
    # rs = test_initial_solution_k_fold(dataframe,Tree(depth))
    print('\n')
