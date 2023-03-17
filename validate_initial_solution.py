from initial_solution import get_initial_solution
from Tree import Tree
import pandas as pd


def normalize(a_b):
    normalized_a_b = {}
    for k, v in a_b.items():
        l = len(v[0]) - 1
        max_val = max(max(abs(v[0])), abs(v[1]))
        den = max_val * l
        normalized_a_b[k] = [[i / den for i in v[0]], v[1] / den]
    return normalized_a_b


def validate_initial_solution(initial_beta_beta_zero, initial_a_b, initial_e_i, init_g, tree, data):
    '''for n in self.tree.Leaves:
        self.model.addConstrs(
            (self.e[i] + self.big_m * (1 - self.g[i, n]) >= self.beta_zero[n] + quicksum(
                self.beta[n, f] * self.data.at[i, f] for f in self.cat_features) - self.data.at[
                 i, self.label]) for
            i in self.datapoints)'''
    valid = True
    y_max = None
    y_min = None
    # M = maxyi- minyi
    label = 'target'
    cat_features = data.columns[data.columns != label]
    features = data.columns
    for i in data.index:
        y_i = data.at[i, label]
        if y_max is None or y_i > y_max:
            y_max = y_i
        if y_min is None or y_i < y_min:
            y_min = y_i
    big_m = 9213365350734
    w = 0.0000005
    d = tree.depth
    cs_1_wrong = []
    cs_1 = []
    cs_2_wrong = []
    cs_2 = []
    obj = 0
    for n in tree.Nodes:
        if not -1 <= all(x for x in initial_a_b[n][0]) <= 1:
            raise AssertionError
        if not -1 <= sum(x for x in initial_a_b[n][0]) <= 1:
            raise AssertionError
    for i in data.index:
        e = sum(abs(v) for x, v in initial_e_i[i].items())
        obj += e
        if not sum(g for n, g in init_g[i].items()) == 1:
            raise AssertionError

        for n in tree.Leaves:
            y_i = data.at[i, label]
            g_i_n = init_g[i][n]
            intercept = initial_beta_beta_zero[n]['intercept']
            sum_coef = sum(
                initial_beta_beta_zero[n]['coef'][idx] * data.at[i, f] for idx, f in enumerate(cat_features))
            val = intercept + sum_coef - y_i
            if not (-e - big_m * (1 - g_i_n) <= val):
                print(
                    f'n{n} \n i {i} \n e {e} \n g {g_i_n} \n int {intercept} \n s_coef {sum_coef} \n v {val} \n y {y_i} \n {initial_e_i[i]}')
                raise AssertionError
            if not (e + big_m * (1 - g_i_n) >= val):
                print(
                    f'n{n} \n i {i} \n e {e} \n g {g_i_n} \n int {intercept} \n s_coef {sum_coef} \n v {val} \n y {y_i} \n {initial_e_i[i]}')
                raise AssertionError

        for n in tree.Nodes:
            v = initial_a_b[n]
            if not -1 <= sum(v[0][idx] for idx, f in enumerate(cat_features)) <= 1:
                for idx, f in enumerate(cat_features):
                    print('hp val', v[0][idx])
                    print('feat', f)
                    print('idx', idx)
                raise AssertionError
            left_leaves = tree.get_left_leaves(n)
            right_leaves = tree.get_right_leaves(n)
            s_f = sum(v[0][idx] * data.at[i, f] for idx, f in enumerate(cat_features))
            tau = -v[1]
            dt = data.loc[i]
            s_g_l = sum(init_g[i][x] for x in left_leaves)
            s_g_r = sum(init_g[i][x] for x in right_leaves)
            c_l = {'n': n, 'i': i, 's_f': s_f, 'tau': tau, 'hp': v[0], 's_g_l': s_g_l, 'w': w,'dt':dt}
            c_r = {'n': n, 'i': i, 's_f': s_f, 'tau': tau, 'hp': v[0], 's_g_r': s_g_r, 'w': w,'dt':dt}
            cs_1.append(c_l)
            cs_2.append(c_r)
            if not s_f + w <= tau + (2 + w) * (1 - s_g_l):
                cs_1_wrong.append(c_l)
                print(f'n{n} \n i {i}\n {data.loc[i]}\n ds_f {s_f}\n tau {tau}\n hp {v[0]}\n s_g_l {s_g_l}\n w {w} \n')
                valid = False
            if not s_f >= tau - 2 * (1 - s_g_r):
                cs_2_wrong.append(c_r)
                print(f'n{n} \n i {i}\n {data.loc[i]}\n s_f {s_f}\n tau {tau}\n hp {v[0]}\n s_g_r {s_g_r}\n w {w} \n')
                valid = False
    return cs_1, cs_2, cs_1_wrong, cs_2_wrong,obj,valid


if __name__ == "__main__":
    dataframe = pd.read_csv('./DataSets/housing_reg.csv')
    depth = 1
    tr = Tree(depth)
    l, a_b, e_i, g_i, cl = get_initial_solution(dataframe, tr)
    cs_l, cs_r, cs_l_wrong, cs_r_wrong,obj,valid = validate_initial_solution(l, normalize(a_b), e_i, g_i, tr, dataframe)
    print('\n')
