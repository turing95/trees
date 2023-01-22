#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from FlowORT import FlowORT
from FlowOCT import FlowOCT
from FlowORT_v2 import FlowORT as FlowORT_v2
from BendersOCT import BendersOCT
from BendersORT import BendersORT
import logger
import getopt
import csv
import numpy as np
from utils import get_model_train_accuracy, get_model_test_accuracy
from utils_oct import get_model_accuracy as get_model_accuracy_v3, get_node_status
from logger import logger
from sklearn.model_selection import KFold


def get_left_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 0)

    return lhs


def get_right_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 1)

    return lhs


def get_target_exp_integer(master, beta, n, i):
    label_i = master.data.at[i, master.label]
    # min (m[i]*p[n] - y[i]*p[n] + beta[n] , m[i]*p[n] + y[i]*p[n] - beta[n])
    if master.m[i] - label_i + beta[n, 1] < master.m[i] + label_i - beta[n, 1]:
        lhs = -1 * (master.m[i] - label_i + master.beta[n, 1])
    else:
        lhs = -1 * (master.m[i] + label_i - master.beta[n, 1])

    return lhs


def get_cut_integer(master, b, beta, left, right, target, i):
    lhs = LinExpr(0) + master.g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(master, beta, n, i)
        lhs = lhs + tmp_lhs

    return lhs


def subproblem_oct(master, b, beta, i):
    label_i = master.data.at[i, master.label]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        pruned, branching, selected_feature, terminal, current_value = get_node_status(master, b, beta, current, i,
                                                                                       master.data)
        if terminal:
            target.append(current)
            if current in master.tree.Nodes:
                left.append(current)
                right.append(current)
            if master.mode == "regression":
                subproblem_value = master.m[i] - abs(current_value - label_i)
            elif master.mode == "classification" and beta[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if master.data.at[i, selected_feature] == 1:  # going right on the branch
                left.append(current)
                target.append(current)
                current = master.tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = master.tree.get_left_children(current)

    return subproblem_value, left, right, target


def subproblem_ort(master, b, beta, i):
    current = 1

    while True:
        pruned, branching, selected_feature, terminal, current_value = get_node_status(master, b, beta, current, i,
                                                                                       master.data)
        if terminal:
            target = current
            break
        elif branching:
            if master.data.at[i, selected_feature] == 1:  # going right on the branch
                current = master.tree.get_right_children(current)
            else:  # going left on the branch
                current = master.tree.get_left_children(current)

    return target


##########################################################
# Defining the callback function
###########################################################
def mycallback_oct(model, where):
    '''
    This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    Using the argument "where" we can see where the callback has been called. We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve the subproblem which is a minimum cut and
    check if g[i] <= value of subproblem[i]. If this is violated we add the corresponding benders constraint as lazy
    constraint to the master problem and proceed. Whenever we have no violated constraint! It means that we have found
    the optimal solution.
    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    '''
    data_train = model._master.data
    mode = model._master.mode

    local_eps = 0.0001
    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b,w and g
        g = model.cbGetSolution(model._vars_g)
        b = model.cbGetSolution(model._vars_b)
        beta = model.cbGetSolution(model._vars_beta)

        added_cut = 0
        # We only want indices that g_i is one!
        for i in data_train.index:
            if mode == "classification":
                g_threshold = 0.5
            elif mode == "regression":
                g_threshold = 0
            if g[i] > g_threshold:
                subproblem_value, left, right, target = subproblem_oct(model._master, b, beta, i)
                if mode == "classification" and subproblem_value == 0:
                    added_cut = 1
                    lhs = get_cut_integer(model._master, b, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)
                elif mode == "regression" and ((subproblem_value + local_eps) < g[i]):
                    added_cut = 1
                    lhs = get_cut_integer(model._master, b, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time


def mycallback_ort(model, where):
    '''
    This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    Using the argument "where" we can see where the callback has been called. We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve the subproblem which is a minimum cut and
    check if g[i] <= value of subproblem[i]. If this is violated we add the corresponding benders constraint as lazy
    constraint to the master problem and proceed. Whenever we have no violated constraint! It means that we have found
    the optimal solution.
    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    '''
    data_train = model._master.data
    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b,w and g
        b = model.cbGetSolution(model._vars_b)
        beta = model.cbGetSolution(model._vars_beta_zero)
        e = model._vars_e

        for i in data_train.index:

            target = subproblem_ort(model._master, b, beta, i)

            for n in model._master.tree.Leaves:
                if n != target:
                    if i == 0:
                        print(target)
                        print(model._master.big_m)
                        print(e[i, n])
                    model.cbLazy(e[i, n] >= model._master.big_m)
        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time


def main(argv):
    print(argv)
    input_file = None
    depth = None
    time_limit = None
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:m:",
                                   ["input_file=", "depth=", "timelimit=", "lambda="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--input_file"):
            input_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)

    data_path = os.getcwd() + '/DataSets/'
    data = pd.read_csv(data_path + input_file)
    '''Name of the column in the dataset representing the class label.
    In the datasets we have, we assume the label is target. Please change this value at your need'''
    label = 'target'

    # Tree structure: We create a tree object of depth d
    tree = Tree(depth)
    print(tree.Nodes)
    print(tree.Leaves)

    ##########################################################
    # output setup
    ##########################################################
    approach_name_1 = 'FlowORT'
    out_put_name = input_file + '_' + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'
    # Using logger we log the output of the console in a text file
    out_put_path = os.getcwd() + '/Results/'
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    out_put_name_1 = input_file + '_' + approach_name_1 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_2 = 'FlowORT_binary'
    out_put_name_2 = input_file + '_' + approach_name_2 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_3 = 'FlowOCT'
    out_put_name_3 = input_file + '_' + approach_name_3 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_4 = 'BenderOCT'
    out_put_name_4 = input_file + '_' + approach_name_4 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_5 = 'BenderOCT'
    out_put_name_5 = input_file + '_' + approach_name_5 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    ##########################################################
    # data splitting
    ##########################################################
    '''
    Creating  train, test and calibration datasets
    We take 50% of the whole data as training, 25% as test and 25% as calibration

    '''
    '''data_train, data_test = train_test_split(data, test_size=1.0, random_state=random_states_list[input_sample - 1])
    data_train_calibration, data_calibration = train_test_split(data_train, test_size=1.0,
                                                                random_state=random_states_list[input_sample - 1])

    data_train = data_train_calibration

    train_len = len(data_train.index)
    
    '''
    train_len = len(data.index)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    k_folds = 5 if train_len < 300 else 10
    random_state = 1
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    maes_train_v1 = []
    maes_train_v2 = []
    maes_train_v3 = []
    maes_train_v4 = []
    maes_train_v5 = []
    maes_test_v1 = []
    maes_test_v2 = []
    maes_test_v3 = []
    maes_test_v4 = []
    maes_test_v5 = []
    mip_gaps_v1 = []
    mip_gaps_v2 = []
    mip_gaps_v3 = []
    mip_gaps_v4 = []
    mip_gaps_v5 = []
    solving_times_v1 = []
    solving_times_v2 = []
    solving_times_v3 = []
    solving_times_v4 = []
    solving_times_v5 = []
    r2_lads_train_v1 = []
    r2_lads_train_v2 = []
    r2_lads_train_v3 = []
    r2_lads_train_v4 = []
    r2_lads_train_v5 = []
    r2_lads_test_v1 = []
    r2_lads_test_v2 = []
    r2_lads_test_v3 = []
    r2_lads_test_v4 = []
    r2_lads_test_v5 = []
    orig_obj_v1 = []
    orig_obj_v2 = []
    orig_obj_v3 = []
    orig_obj_v4 = []
    orig_obj_v5 = []
    n_k_folds = kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        ##########################################################
        # Creating and Solving the problem
        ##########################################################
        # We create the MIP problem by passing the required arguments
        start_time = time.time()
        primal_v1 = FlowORT(data_train, label, tree, time_limit)

        primal_v1.create_primal_problem()
        primal_v1.model.update()
        primal_v1.model.optimize()

        end_time = time.time()

        solving_time_v1 = end_time - start_time
        start_time = time.time()

        primal_v2 = FlowORT_v2(data_train, label, tree, time_limit)

        primal_v2.create_primal_problem()
        primal_v2.model.update()
        primal_v2.model.optimize()
        end_time = time.time()

        solving_time_v2 = end_time - start_time

        start_time = time.time()
        primal_v3 = FlowOCT(data_train, label, tree, time_limit)

        primal_v3.create_primal_problem()
        primal_v3.model.update()
        primal_v3.model.optimize()

        end_time = time.time()

        solving_time_v3 = end_time - start_time

        ##########################################################
        # Preparing the output
        ##########################################################
        b_value_v1 = primal_v1.model.getAttr("X", primal_v1.b)
        b_value_v2 = primal_v2.model.getAttr("X", primal_v2.b)
        b_value_v3 = primal_v3.model.getAttr("X", primal_v3.b)

        beta_zero_v1 = primal_v1.model.getAttr("x", primal_v1.beta_zero)
        beta_v1 = None
        beta_zero_v2 = primal_v2.model.getAttr("x", primal_v2.beta_zero)
        beta_v2 = None
        beta_zero_v3 = primal_v3.model.getAttr("x", primal_v3.beta)
        beta_v3 = None
        # beta_v3 = None

        lower_bound_v2 = primal_v2.model.getAttr("ObjBound")

        print("\n\n")
        print('\n\nTotal Solving Time v1', solving_time_v1)
        print('Total Solving Time v2', solving_time_v2)
        print('Total Solving Time v3', solving_time_v3)
        print("\n\nobj value v1", primal_v1.model.getAttr("ObjVal"))
        print("obj value v2", primal_v2.model.getAttr("ObjVal"))
        print("obj value v3", primal_v3.model.getAttr("ObjVal"))
        print('\n\nbnf_v1', b_value_v1)
        print('bnf_v2', b_value_v2)
        print('bnf_v3', b_value_v3)
        print('\n\npi_n v1')
        print('#####')
        print('pi_n v2')

        print(f'\n\nbeta_zero V1 {beta_zero_v1}')
        print(f'beta_zero V2 {beta_zero_v2}')
        print(f'beta_zero V3 {beta_zero_v3}')
        print(f'\n\nbeta V1 {beta_v1}')
        print(f'beta V3 {beta_v3}')
        print(f'\n\nlower_bound_v2 {lower_bound_v2}')

        reg_res_v1, mae_v1, mse_v1, r2_v1, r2_lad_alt_v1 = get_model_accuracy_v3(primal_v1,
                                                                                 data_train,
                                                                                 b_value_v1,
                                                                                 beta_zero_v1,
                                                                                 beta_v1)
        reg_res_v1_test, mae_v1_test, mse_v1_test, r2_v1_test, r2_lad_alt_v1_test = get_model_accuracy_v3(
            primal_v1,
            data_test,
            b_value_v1,
            beta_zero_v1,
            beta_v1)
        reg_res_v2, mae_v2, mse_v2, r2_v2, r2_lad_alt_v2 = get_model_accuracy_v3(primal_v2, data_train, b_value_v2,
                                                                                 beta_zero_v2,
                                                                                 beta_v2)
        reg_res_v2_test, mae_v2_test, mse_v2_test, r2_v2_test, r2_lad_alt_v2_test = get_model_accuracy_v3(primal_v2,
                                                                                                          data_test,
                                                                                                          b_value_v2,
                                                                                                          beta_zero_v2,
                                                                                                          beta_v2)

        reg_res_v3, mae_v3, mse_v3, r2_v3, r2_lad_alt_v3 = get_model_accuracy_v3(primal_v3, data_train, b_value_v3,
                                                                                 beta_zero_v3, beta_v3)

        _, mae_v3_test, _, _, r2_lad_alt_v3_test = get_model_accuracy_v3(primal_v3, data_test, b_value_v3, beta_zero_v3,
                                                                         beta_v3)

        maes_train_v1.append(mae_v1)
        maes_train_v2.append(mae_v2)
        maes_train_v3.append(mae_v3)

        maes_test_v1.append(mae_v1_test)
        maes_test_v2.append(mae_v2_test)
        maes_test_v3.append(mae_v3_test)

        orig_obj_v1.append(reg_res_v1)
        orig_obj_v2.append(reg_res_v2)
        orig_obj_v3.append(reg_res_v3)

        mip_gaps_v1.append((reg_res_v1 - lower_bound_v2) / lower_bound_v2)
        mip_gaps_v2.append((reg_res_v2 - lower_bound_v2) / lower_bound_v2)
        mip_gaps_v3.append((reg_res_v3 - lower_bound_v2) / lower_bound_v2)
        solving_times_v1.append(solving_time_v1)
        solving_times_v2.append(solving_time_v2)
        solving_times_v3.append(solving_time_v3)

        r2_lads_train_v1.append(r2_lad_alt_v1)
        r2_lads_train_v2.append(r2_lad_alt_v2)
        r2_lads_train_v3.append(r2_lad_alt_v3)

        r2_lads_test_v1.append(r2_lad_alt_v1_test)
        r2_lads_test_v2.append(r2_lad_alt_v2_test)
        r2_lads_test_v3.append(r2_lad_alt_v3_test)

    # writing info to the file
    result_file_v1 = out_put_name_1 + '.csv'
    print('mip gaps v1', mip_gaps_v1)
    print('mip gaps v2', mip_gaps_v2)
    print('mip gaps v3', mip_gaps_v3)
    print('mip gaps v4', mip_gaps_v4)
    print('mip gaps v5', mip_gaps_v5)
    print('solving times v1', solving_times_v1)
    print('solving times v2', solving_times_v2)
    print('solving times v3', solving_times_v3)
    print('solving times v4', solving_times_v4)
    print('solving times v5', solving_times_v5)
    print('maes v1', maes_train_v1)
    print('maes v2', maes_train_v2)
    print('maes v3', maes_train_v3)
    print('maes v4', maes_train_v4)
    print('maes v5', maes_train_v5)
    print('orig obj v1', orig_obj_v1)
    print('orig obj v2', orig_obj_v2)
    print('orig obj v3', orig_obj_v3)
    print('orig obj v4', orig_obj_v4)
    print('orig obj v5', orig_obj_v5)
    print('maes v1 test', maes_train_v1)
    print('maes v2 test', maes_train_v2)
    print('maes v3 test', maes_train_v3)
    print('maes v4 test', maes_train_v4)
    print('maes v5 test', maes_train_v5)
    row_1 = [approach_name_1, input_file, depth, n_k_folds, time_limit, np.average(mip_gaps_v1) * 100,
             np.average(solving_times_v1), np.average(maes_train_v1), np.average(r2_lads_train_v1),
             np.average(maes_test_v1), np.average(r2_lads_test_v1)]
    row_2 = [approach_name_2, input_file, depth, n_k_folds, time_limit, np.average(mip_gaps_v2) * 100,
             np.average(solving_times_v2), np.average(maes_train_v2), np.average(r2_lads_train_v2),
             np.average(maes_test_v2), np.average(r2_lads_test_v2)]
    row_3 = [approach_name_3, input_file, depth, n_k_folds, time_limit, np.average(mip_gaps_v3) * 100,
             np.average(solving_times_v3), np.average(maes_train_v3), np.average(r2_lads_train_v3),
             np.average(maes_test_v3), np.average(r2_lads_test_v3)]
    with open(out_put_path + result_file_v1, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            row_1)

    # writing info to the file
    result_file_v2 = out_put_name_2 + '.csv'
    with open(out_put_path + result_file_v2, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_2)

    # writing info to the file
    result_file_v3 = out_put_name_3 + '.csv'
    with open(out_put_path + result_file_v3, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_3)

    result_file_v5 = out_put_name + '.csv'
    with open(out_put_path + result_file_v5, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_1)
        results_writer.writerow(row_2)
        results_writer.writerow(row_3)


if __name__ == "__main__":
    main(sys.argv[1:])
