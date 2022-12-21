#!/usr/bin/python
from utils import *
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from FlowORT import FlowORT
from FlowOCT import FlowOCT
from FlowORT_v2 import FlowORT as FlowORT_v2
from BendersORT import BendersORT
import logger
import getopt
import csv
import numpy as np
from utils import get_model_train_accuracy, get_model_test_accuracy
from utils_oct import get_model_accuracy as get_model_accuracy_v3
from logger import logger
from sklearn.model_selection import KFold


def get_left_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 0)

    return lhs


def get_right_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 1)

    return lhs


def get_target_exp_integer(master, p, beta, n, i):
    label_i = master.data.at[i, master.label]

    if master.mode == "classification":
        lhs = -1 * master.beta[n, label_i]
    elif master.mode == "regression":
        # min (m[i]*p[n] - y[i]*p[n] + beta[n] , m[i]*p[n] + y[i]*p[n] - beta[n])
        if master.m[i] * p[n] - label_i * p[n] + beta[n, 1] < master.m[i] * p[n] + label_i * p[n] - beta[n, 1]:
            lhs = -1 * (master.m[i] * master.p[n] - label_i * master.p[n] + master.beta[n, 1])
        else:
            lhs = -1 * (master.m[i] * master.p[n] + label_i * master.p[n] - master.beta[n, 1])

    return lhs


def get_cut_integer(master, b, p, beta, left, right, target, i):
    lhs = LinExpr(0) + master.g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(master, p, beta, n, i)
        lhs = lhs + tmp_lhs

    return lhs


def subproblem(master, b, p, beta, i):
    label_i = master.data.at[i, master.label]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        pruned, branching, selected_feature, terminal, current_value = get_node_status(master, b, beta, current)
        if terminal:
            target.append(current)
            if current in master.tree.Nodes:
                left.append(current)
                right.append(current)
                subproblem_value = master.m[i] - abs(current_value - label_i)
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


##########################################################
# Defining the callback function
###########################################################
def mycallback(model, where):
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
        beta = model.cbGetSolution(model._vars_beta_zero)

        added_cut = 0
        # We only want indices that g_i is one!
        for i in data_train.index:
            g_threshold = 0
            if g[i] > g_threshold:
                subproblem_value, left, right, target = subproblem(model._master, b, p, beta, i)
                added_cut = 1
                lhs = get_cut_integer(model._master, b, p, beta, left, right, target, i)
                model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time


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
    ##########################################################
    # output setup
    ##########################################################
    approach_name_1 = 'FlowORT'
    out_put_name = input_file + '_' + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_linear_cross_validation'
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

    approach_name_4 = 'BendersORT'
    out_put_name_4 = input_file + '_' + approach_name_3 + '_d_' + str(depth) + '_t_' + str(
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
    maes_test_v1 = []
    maes_test_v2 = []
    maes_test_v3 = []
    maes_test_v4 = []
    mip_gaps_v1 = []
    mip_gaps_v2 = []
    mip_gaps_v3 = []
    mip_gaps_v4 = []
    solving_times_v1 = []
    solving_times_v2 = []
    solving_times_v3 = []
    solving_times_v4 = []
    r2_lads_train_v1 = []
    r2_lads_train_v2 = []
    r2_lads_train_v3 = []
    r2_lads_train_v4 = []
    r2_lads_test_v1 = []
    r2_lads_test_v2 = []
    r2_lads_test_v3 = []
    r2_lads_test_v4 = []
    orig_obj_v1 = []
    orig_obj_v2 = []
    orig_obj_v3 = []
    orig_obj_v4 = []
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

        start_time = time.time()

        # We create the master problem by passing the required arguments
        master = BendersORT(data_train, label, tree, time_limit)

        master.create_master_problem()
        master.model.update()
        master.model.optimize(mycallback)
        solving_time_v4 = end_time - start_time

        ##########################################################
        # Preparing the output
        ##########################################################
        b_value_v1 = primal_v1.model.getAttr("X", primal_v1.b)
        b_value_v2 = primal_v2.model.getAttr("X", primal_v2.b)
        b_value_v3 = primal_v3.model.getAttr("X", primal_v3.b)
        b_value_v4 = master.model.getAttr("X", master.b)

        beta_zero_v1 = primal_v1.model.getAttr("x", primal_v1.beta_zero)
        beta_v1 = None
        beta_zero_v2 = primal_v2.model.getAttr("x", primal_v2.beta_zero)
        beta_v2 = None
        beta_zero_v3 = primal_v3.model.getAttr("x", primal_v3.beta)
        beta_v3 = None
        beta_zero_v4 = master.model.getAttr("x", master.beta_zero)
        beta_v4 = None
        # beta_v3 = None
        # zeta = primal.model.getAttr("x", primal.zeta)
        z_v1 = primal_v1.model.getAttr("x", primal_v1.z)
        z_v2 = primal_v2.model.getAttr("x", primal_v2.z)

        lower_bound_v2 = primal_v2.model.getAttr("ObjBound")

        print("\n\n")
        print('\n\nTotal Solving Time v1', solving_time_v1)
        print('Total Solving Time v2', solving_time_v2)
        print('Total Solving Time v3', solving_time_v3)
        print('Total Solving Time v4', solving_time_v4)
        print("\n\nobj value v1", primal_v1.model.getAttr("ObjVal"))
        print("obj value v2", primal_v2.model.getAttr("ObjVal"))
        print("obj value v3", primal_v3.model.getAttr("ObjVal"))
        print("obj value v4", master.model.getAttr("ObjVal"))
        print('\n\nbnf_v1', b_value_v1)
        print('bnf_v2', b_value_v2)
        print('bnf_v3', b_value_v3)
        print('\n\npi_n v1')
        print(z_v1)
        print('#####')
        print('pi_n v2')
        print(z_v2)

        print(f'\n\nbeta_zero V1 {beta_zero_v1}')
        print(f'beta_zero V2 {beta_zero_v2}')
        print(f'beta_zero V3 {beta_zero_v3}')
        print(f'beta_zero V4 {beta_zero_v4}')
        print(f'lower_bound_v2 {lower_bound_v2}')

        r2_v1, mse_v1, mae_v1, r2_lad_v1, r2_lad_alt_v1, reg_res_v1 = get_model_train_accuracy(data_train,
                                                                                               primal_v1.datapoints,
                                                                                               z_v1,
                                                                                               beta_zero_v1,
                                                                                               depth, primal_v1,
                                                                                               beta_v1)
        r2_v1_test, mse_v1_test, mae_v1_test, r2_lad_v1_test, r2_lad_alt_v1_test, reg_res_v1_test = get_model_test_accuracy(
            primal_v1,
            data_test,
            b_value_v1,
            beta_zero_v1,
            beta_v1)
        r2_v2, mse_v2, mae_v2, r2_lad_v2, r2_lad_alt_v2, reg_res_v2 = get_model_train_accuracy(data_train,
                                                                                               primal_v2.datapoints,
                                                                                               z_v2, beta_zero_v2,
                                                                                               depth, primal_v2,
                                                                                               beta_v2)
        r2_v2_test, mse_v2_test, mae_v2_test, r2_lad_v2_test, r2_lad_alt_v2_test, reg_res_v2_test = get_model_test_accuracy(
            primal_v2,
            data_test,
            b_value_v2,
            beta_zero_v2, beta_v2)

        reg_res_v3, mae_v3, mse_v3, r2_v3, r2_lad_alt_v3 = get_model_accuracy_v3(primal_v3, data_train, b_value_v3,
                                                                                 beta_zero_v3, beta_v3)

        _, mae_v3_test, _, _, r2_lad_alt_v3_test = get_model_accuracy_v3(primal_v3, data_test, b_value_v3, beta_zero_v3,
                                                                         beta_v3)

        reg_res_v4, mae_v4, mse_v4, r2_v4, r2_lad_alt_v4 = get_model_accuracy_v3(master, data_train, b_value_v4,
                                                                                 beta_zero_v4, beta_v4)

        _, mae_v4_test, _, _, r2_lad_alt_v4_test = get_model_accuracy_v3(master, data_test, b_value_v4, beta_zero_v4,
                                                                         beta_v4)
        maes_train_v1.append(mae_v1)
        maes_train_v2.append(mae_v2)
        maes_train_v3.append(mae_v3)
        maes_train_v4.append(mae_v4)
        maes_test_v1.append(mae_v1_test)
        maes_test_v2.append(mae_v2_test)
        maes_test_v3.append(mae_v3_test)
        maes_test_v4.append(mae_v4_test)
        orig_obj_v1.append(reg_res_v1)
        orig_obj_v2.append(reg_res_v2)
        orig_obj_v3.append(reg_res_v3)
        orig_obj_v4.append(reg_res_v4)
        mip_gaps_v1.append((reg_res_v1 - lower_bound_v2) / lower_bound_v2)
        mip_gaps_v2.append((reg_res_v2 - lower_bound_v2) / lower_bound_v2)
        mip_gaps_v3.append((reg_res_v3 - lower_bound_v2) / lower_bound_v2)
        mip_gaps_v3.append((reg_res_v4 - lower_bound_v2) / lower_bound_v2)
        solving_times_v1.append(solving_time_v1)
        solving_times_v2.append(solving_time_v2)
        solving_times_v3.append(solving_time_v3)
        solving_times_v3.append(solving_time_v4)
        r2_lads_train_v1.append(1 - r2_lad_v1)
        r2_lads_train_v2.append(1 - r2_lad_v2)
        r2_lads_train_v3.append(r2_lad_alt_v3)
        r2_lads_train_v4.append(r2_lad_alt_v4)
        r2_lads_test_v1.append(r2_lad_alt_v1_test)
        r2_lads_test_v2.append(r2_lad_alt_v2_test)
        r2_lads_test_v3.append(r2_lad_alt_v3_test)
        r2_lads_test_v4.append(r2_lad_alt_v4_test)
    # writing info to the file
    result_file_v1 = out_put_name_1 + '.csv'
    print('mip gaps v1', mip_gaps_v1)
    print('mip gaps v2', mip_gaps_v2)
    print('mip gaps v3', mip_gaps_v3)
    print('mip gaps v4', mip_gaps_v4)
    print('solving times v1', solving_times_v1)
    print('solving times v2', solving_times_v2)
    print('solving times v3', solving_times_v3)
    print('solving times v4', solving_times_v4)
    print('maes v1 train', maes_train_v1)
    print('maes v2 train', maes_train_v2)
    print('maes v3 train', maes_train_v3)
    print('maes v4 train', maes_train_v4)
    print('orig obj v1', orig_obj_v1)
    print('orig obj v2', orig_obj_v2)
    print('orig obj v3', orig_obj_v3)
    print('orig obj v4', orig_obj_v4)
    row_1 = [approach_name_1, input_file, depth, n_k_folds, time_limit, np.average(mip_gaps_v1) * 100,
             np.average(solving_times_v1), np.average(maes_train_v1), np.average(r2_lads_train_v1),
             np.average(maes_test_v1), np.average(r2_lads_test_v1)]
    row_2 = [approach_name_2, input_file, depth, n_k_folds, time_limit, np.average(mip_gaps_v2) * 100,
             np.average(solving_times_v2), np.average(maes_train_v2), np.average(r2_lads_train_v2),
             np.average(maes_test_v2), np.average(r2_lads_test_v2)]
    row_3 = [approach_name_3, input_file, depth, n_k_folds, time_limit, np.average(mip_gaps_v3) * 100,
             np.average(solving_times_v3), np.average(maes_train_v3), np.average(r2_lads_train_v3),
             np.average(maes_test_v3), np.average(r2_lads_test_v3)]
    row_4 = [approach_name_4, input_file, depth, n_k_folds, time_limit, np.average(mip_gaps_v4) * 100,
             np.average(solving_times_v4), np.average(maes_train_v4), np.average(r2_lads_train_v4),
             np.average(maes_test_v4), np.average(r2_lads_test_v4)]
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
    result_file_v4 = out_put_name + '.csv'
    with open(out_put_path + result_file_v4, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_1)
        results_writer.writerow(row_2)
        results_writer.writerow(row_3)
        results_writer.writerow(row_4)


if __name__ == "__main__":
    main(sys.argv[1:])
