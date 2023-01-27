#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from FlowORT_no_pi import FlowORT as FlowORT_no_pi
from FlowORT_with_pi import FlowORT as FlowORT_with_pi
from FlowORT_no_e_n import FlowORT as FlowORT_no_e_n
from FlowOCT_no_p import FlowOCT as FlowOCT_no_p
from FlowOCT_with_p import FlowOCT as FlowOCT_with_p
from FlowORT_binary import FlowORT as FlowORT_binary
from BendersOCT import BendersOCT as BendersOCT_with_p
import logger
import getopt
import csv
import numpy as np
from utils.utils_oct_no_p import get_model_accuracy
from logger import logger
from sklearn.model_selection import KFold
from utils.utils_oct_with_p import mycallback, get_model_accuracy as get_model_accuracy_with_p


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
    approach_name_1 = 'FlowORT_no_pi'
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

    approach_name_3 = 'FlowOCT_no_p'
    out_put_name_3 = input_file + '_' + approach_name_3 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_4 = 'BenderOCT_with_p'
    out_put_name_4 = input_file + '_' + approach_name_3 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_5 = 'FlowORT_with_pi'
    out_put_name_5 = input_file + '_' + approach_name_5 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_6 = 'FlowORT_no_e_n'
    out_put_name_6 = input_file + '_' + approach_name_6 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_7 = 'FlowOCT_with_p'
    out_put_name_7 = input_file + '_' + approach_name_7 + '_d_' + str(depth) + '_t_' + str(
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
    features_count = len(data.columns)-1
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    #k_folds = 5 if train_len < 300 else 10
    k_folds = 5
    random_state = 1
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    maes_train_no_pi = []
    maes_train_binary = []
    maes_train_oct_no_p = []
    maes_train_oct_with_p = []
    maes_train_benders_oct_with_p = []
    maes_train_with_pi = []
    maes_train_no_e_n = []
    maes_test_no_pi = []
    maes_test_binary = []
    maes_test_oct_no_p = []
    maes_test_oct_with_p = []
    maes_test_benders_oct_with_p = []
    maes_test_with_pi = []
    maes_test_no_e_n = []
    mip_gaps_no_pi = []
    mip_gaps_binary = []
    mip_gaps_oct_no_p = []
    mip_gaps_oct_with_p = []
    mip_gaps_benders_oct_with_p = []
    mip_gaps_with_pi = []
    mip_gaps_no_e_n = []
    solving_times_no_pi = []
    solving_times_binary = []
    solving_times_oct_no_p = []
    solving_times_oct_with_p = []
    solving_times_benders_oct_with_p = []
    solving_times_with_pi = []
    solving_times_no_e_n = []
    r2_lads_train_no_pi = []
    r2_lads_train_binary = []
    r2_lads_train_oct_no_p = []
    r2_lads_train_oct_with_p = []
    r2_lads_train_benders_oct_with_p = []
    r2_lads_train_with_pi = []
    r2_lads_train_no_e_n = []
    r2_lads_test_no_pi = []
    r2_lads_test_binary = []
    r2_lads_test_oct_no_p = []
    r2_lads_test_oct_with_p = []
    r2_lads_test_benders_oct_with_p = []
    r2_lads_test_with_pi = []
    r2_lads_test_no_e_n = []
    orig_obj_no_pi = []
    orig_obj_binary = []
    orig_obj_oct_no_p = []
    orig_obj_oct_with_p = []
    orig_obj_benders_oct_with_p = []
    orig_obj_with_pi = []
    orig_obj_no_e_n = []
    n_k_folds = kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        ##########################################################
        # Creating and Solving the problem
        ##########################################################
        # We create the MIP problem by passing the required arguments
        start_time = time.time()
        primal_no_pi = FlowORT_no_pi(data_train, label, tree, time_limit)

        primal_no_pi.create_primal_problem()
        primal_no_pi.model.update()
        primal_no_pi.model.optimize()

        end_time = time.time()

        solving_time_no_pi = end_time - start_time

        start_time = time.time()
        primal_with_pi = FlowORT_with_pi(data_train, label, tree, time_limit)

        primal_with_pi.create_primal_problem()
        primal_with_pi.model.update()
        primal_with_pi.model.optimize()

        end_time = time.time()

        solving_time_with_pi = end_time - start_time

        start_time = time.time()
        primal_no_e_n = FlowORT_no_e_n(data_train, label, tree, time_limit)

        primal_no_e_n.create_primal_problem()
        primal_no_e_n.model.update()
        primal_no_e_n.model.optimize()

        end_time = time.time()

        solving_time_no_e_n = end_time - start_time

        start_time = time.time()

        primal_binary = FlowORT_binary(data_train, label, tree, time_limit)

        primal_binary.create_primal_problem()
        primal_binary.model.update()
        primal_binary.model.optimize()
        end_time = time.time()

        solving_time_binary = end_time - start_time

        start_time = time.time()
        primal_oct_no_p = FlowOCT_no_p(data_train, label, tree, time_limit)

        primal_oct_no_p.create_primal_problem()
        primal_oct_no_p.model.update()
        primal_oct_no_p.model.optimize()

        end_time = time.time()

        solving_time_oct_no_p = end_time - start_time

        start_time = time.time()
        primal_oct_with_p = FlowOCT_with_p(data_train, label, tree, time_limit)

        primal_oct_with_p.create_primal_problem()
        primal_oct_with_p.model.update()
        primal_oct_with_p.model.optimize()

        end_time = time.time()

        solving_time_oct_with_p = end_time - start_time

        start_time = time.time()
        primal_benders_oct_with_p = BendersOCT_with_p(data_train, label, tree, time_limit)

        primal_benders_oct_with_p.create_master_problem()
        primal_benders_oct_with_p.model.update()
        primal_benders_oct_with_p.model.optimize(mycallback)
        end_time = time.time()

        solving_time_benders_oct_with_p = end_time - start_time
        ##########################################################
        # Preparing the output
        ##########################################################
        b_value_no_pi = primal_no_pi.model.getAttr("X", primal_no_pi.b)
        b_value_with_pi = primal_with_pi.model.getAttr("X", primal_with_pi.b)
        b_value_no_e_n = primal_no_e_n.model.getAttr("X", primal_no_e_n.b)
        b_value_binary = primal_binary.model.getAttr("X", primal_binary.b)
        b_value_oct_no_p = primal_oct_no_p.model.getAttr("X", primal_oct_no_p.b)
        b_value_oct_with_p = primal_oct_with_p.model.getAttr("X", primal_oct_with_p.b)
        b_value_benders_oct_with_p = primal_benders_oct_with_p.model.getAttr("X", primal_benders_oct_with_p.b)

        beta_zero_no_pi = primal_no_pi.model.getAttr("x", primal_no_pi.beta_zero)
        beta_no_pi = None

        beta_zero_with_pi = primal_with_pi.model.getAttr("x", primal_with_pi.beta_zero)
        beta_with_pi = None

        beta_zero_no_e_n = primal_no_e_n.model.getAttr("x", primal_no_e_n.beta_zero)
        beta_no_e_n = None

        beta_zero_binary = primal_binary.model.getAttr("x", primal_binary.beta_zero)
        beta_binary = None

        beta_zero_oct_no_p = primal_oct_no_p.model.getAttr("x", primal_oct_no_p.beta)
        beta_oct_no_p = None

        beta_zero_oct_with_p = primal_oct_with_p.model.getAttr("x", primal_oct_with_p.beta)
        beta_oct_with_p = None

        beta_zero_benders_oct_with_p = primal_benders_oct_with_p.model.getAttr("x", primal_benders_oct_with_p.beta)
        p_benders_oct_with_p = primal_benders_oct_with_p.model.getAttr("x", primal_benders_oct_with_p.p)
        beta_benders_oct_with_p = None

        lower_bound_binary = primal_binary.model.getAttr("ObjBound")

        print("\n\n")
        print('\n\nTotal Solving Time no_pi', solving_time_no_pi)
        print('Total Solving Time with_pi', solving_time_with_pi)
        print('Total Solving Time no_e_n', solving_time_no_e_n)
        print('Total Solving Time binary', solving_time_binary)
        print('Total Solving Time oct_no_p', solving_time_oct_no_p)
        print('Total Solving Time oct_with_p', solving_time_oct_with_p)
        print('Total Solving Time benders_oct_with_p', solving_time_benders_oct_with_p)
        print("\n\nobj value no_pi", primal_no_pi.model.getAttr("ObjVal"))
        print("obj value with_pi", primal_with_pi.model.getAttr("ObjVal"))
        print("obj value no_e_n", primal_no_e_n.model.getAttr("ObjVal"))
        print("obj value binary", primal_binary.model.getAttr("ObjVal"))
        print("obj value oct_no_p", primal_oct_no_p.model.getAttr("ObjVal"))
        print("obj value oct_with_p", primal_oct_with_p.model.getAttr("ObjVal"))
        print("obj value benders_oct_with_p", primal_benders_oct_with_p.model.getAttr("ObjVal"))
        print('\n\nbnf_no_pi', b_value_no_pi)
        print('bnf_with_pi', b_value_with_pi)
        print('bnf_no_e_n', b_value_no_e_n)
        print('bnf_binary', b_value_binary)
        print('bnf_oct_no_p', b_value_oct_no_p)
        print('bnf_oct_with_p', b_value_oct_with_p)
        print('bnf_benders_oct_with_p', b_value_benders_oct_with_p)
        print(f'\n\nbeta_zero no_pi {beta_zero_no_pi}')
        print(f'beta_zero with_pi {beta_zero_with_pi}')
        print(f'beta_zero no_e_n {beta_zero_no_e_n}')
        print(f'beta_zero binary {beta_zero_binary}')
        print(f'beta_zero oct_no_p {beta_zero_oct_no_p}')
        print(f'beta_zero oct_with_p {beta_zero_oct_with_p}')
        print(f'beta_zero benders_oct_with_p {beta_zero_benders_oct_with_p}')
        print(f'\n\nlower_bound_binary {lower_bound_binary}')

        reg_res_no_pi, mae_no_pi, mse_no_pi, r2_no_pi, r2_lad_alt_no_pi = get_model_accuracy(primal_no_pi,
                                                                                             data_train,
                                                                                             b_value_no_pi,
                                                                                             beta_zero_no_pi,
                                                                                             beta_no_pi)
        reg_res_no_pi_test, mae_no_pi_test, mse_no_pi_test, r2_no_pi_test, r2_lad_alt_no_pi_test = get_model_accuracy(
            primal_no_pi,
            data_test,
            b_value_no_pi,
            beta_zero_no_pi,
            beta_no_pi)

        reg_res_with_pi, mae_with_pi, mse_with_pi, r2_with_pi, r2_lad_alt_with_pi = get_model_accuracy(primal_with_pi,
                                                                                                       data_train,
                                                                                                       b_value_with_pi,
                                                                                                       beta_zero_with_pi,
                                                                                                       beta_with_pi)
        reg_res_with_pi_test, mae_with_pi_test, mse_with_pi_test, r2_with_pi_test, r2_lad_alt_with_pi_test = get_model_accuracy(
            primal_with_pi,
            data_test,
            b_value_with_pi,
            beta_zero_with_pi,
            beta_with_pi)

        reg_res_no_e_n, mae_no_e_n, mse_no_e_n, r2_no_e_n, r2_lad_alt_no_e_n = get_model_accuracy(primal_no_e_n,
                                                                                                  data_train,
                                                                                                  b_value_no_e_n,
                                                                                                  beta_zero_no_e_n,
                                                                                                  beta_no_e_n)
        reg_res_no_e_n_test, mae_no_e_n_test, mse_no_e_n_test, r2_no_e_n_test, r2_lad_alt_no_e_n_test = get_model_accuracy(
            primal_no_e_n,
            data_test,
            b_value_no_e_n,
            beta_zero_no_e_n,
            beta_no_e_n)

        reg_res_binary, mae_binary, mse_binary, r2_binary, r2_lad_alt_binary = get_model_accuracy(
            primal_binary,
            data_train,
            b_value_binary,
            beta_zero_binary,
            beta_binary)
        reg_res_binary_test, mae_binary_test, mse_binary_test, r2_binary_test, r2_lad_alt_binary_test = get_model_accuracy(
            primal_binary,
            data_test,
            b_value_binary,
            beta_zero_binary,
            beta_binary)

        reg_res_oct_no_p, mae_oct_no_p, mse_oct_no_p, r2_oct_no_p, r2_lad_alt_oct_no_p = get_model_accuracy(
            primal_oct_no_p, data_train, b_value_oct_no_p,
            beta_zero_oct_no_p, beta_oct_no_p)

        _, mae_oct_no_p_test, _, _, r2_lad_alt_oct_no_p_test = get_model_accuracy(primal_oct_no_p, data_test,
                                                                                  b_value_oct_no_p,
                                                                                  beta_zero_oct_no_p,
                                                                                  beta_oct_no_p)

        reg_res_oct_with_p, mae_oct_with_p, mse_oct_with_p, r2_oct_with_p, r2_lad_alt_oct_with_p = get_model_accuracy(
            primal_oct_with_p, data_train, b_value_oct_with_p,
            beta_zero_oct_with_p, beta_oct_with_p)

        _, mae_oct_with_p_test, _, _, r2_lad_alt_oct_with_p_test = get_model_accuracy(primal_oct_with_p, data_test,
                                                                                  b_value_oct_with_p,
                                                                                  beta_zero_oct_with_p,
                                                                                  beta_oct_with_p)

        reg_res_benders_oct_with_p, mae_benders_oct_with_p, mse_benders_oct_with_p, r2_benders_oct_with_p, r2_lad_alt_benders_oct_with_p = get_model_accuracy_with_p(
            primal_benders_oct_with_p, data_train, b_value_benders_oct_with_p,
            beta_zero_benders_oct_with_p, p_benders_oct_with_p)

        _, mae_benders_oct_with_p_test, _, _, r2_lad_alt_benders_oct_with_p_test = get_model_accuracy_with_p(
            primal_benders_oct_with_p, data_test,
            b_value_benders_oct_with_p,
            beta_zero_benders_oct_with_p,
            p_benders_oct_with_p)

        maes_train_no_pi.append(mae_no_pi)
        maes_train_with_pi.append(mae_with_pi)
        maes_train_no_e_n.append(mae_no_e_n)
        maes_train_binary.append(mae_binary)
        maes_train_oct_no_p.append(mae_oct_no_p)
        maes_train_oct_with_p.append(mae_oct_with_p)
        maes_train_benders_oct_with_p.append(mae_benders_oct_with_p)

        maes_test_no_pi.append(mae_no_pi_test)
        maes_test_with_pi.append(mae_with_pi_test)
        maes_test_no_e_n.append(mae_no_e_n_test)
        maes_test_binary.append(mae_binary_test)
        maes_test_oct_no_p.append(mae_oct_no_p_test)
        maes_test_oct_with_p.append(mae_oct_with_p_test)
        maes_test_benders_oct_with_p.append(mae_benders_oct_with_p_test)

        orig_obj_no_pi.append(reg_res_no_pi)
        orig_obj_with_pi.append(reg_res_with_pi)
        orig_obj_no_e_n.append(reg_res_no_e_n)
        orig_obj_binary.append(reg_res_binary)
        orig_obj_oct_no_p.append(reg_res_oct_no_p)
        orig_obj_oct_with_p.append(reg_res_oct_with_p)
        orig_obj_benders_oct_with_p.append(reg_res_benders_oct_with_p)

        mip_gaps_no_pi.append((reg_res_no_pi - lower_bound_binary) / lower_bound_binary)
        mip_gaps_with_pi.append((reg_res_with_pi - lower_bound_binary) / lower_bound_binary)
        mip_gaps_no_e_n.append((reg_res_no_e_n - lower_bound_binary) / lower_bound_binary)
        mip_gaps_binary.append((reg_res_binary - lower_bound_binary) / lower_bound_binary)
        mip_gaps_oct_no_p.append((reg_res_oct_no_p - lower_bound_binary) / lower_bound_binary)
        mip_gaps_oct_with_p.append((reg_res_oct_with_p - lower_bound_binary) / lower_bound_binary)
        mip_gaps_benders_oct_with_p.append((reg_res_benders_oct_with_p - lower_bound_binary) / lower_bound_binary)

        solving_times_no_pi.append(solving_time_no_pi)
        solving_times_with_pi.append(solving_time_with_pi)
        solving_times_no_e_n.append(solving_time_no_e_n)
        solving_times_binary.append(solving_time_binary)
        solving_times_oct_no_p.append(solving_time_oct_no_p)
        solving_times_oct_with_p.append(solving_time_oct_with_p)
        solving_times_benders_oct_with_p.append(solving_time_benders_oct_with_p)

        r2_lads_train_no_pi.append(r2_lad_alt_no_pi)
        r2_lads_train_with_pi.append(r2_lad_alt_with_pi)
        r2_lads_train_no_e_n.append(r2_lad_alt_no_e_n)
        r2_lads_train_binary.append(r2_lad_alt_binary)
        r2_lads_train_oct_no_p.append(r2_lad_alt_oct_no_p)
        r2_lads_train_oct_with_p.append(r2_lad_alt_oct_with_p)
        r2_lads_train_benders_oct_with_p.append(r2_lad_alt_benders_oct_with_p)

        r2_lads_test_no_pi.append(r2_lad_alt_no_pi_test)
        r2_lads_test_with_pi.append(r2_lad_alt_with_pi_test)
        r2_lads_test_no_e_n.append(r2_lad_alt_no_e_n_test)
        r2_lads_test_binary.append(r2_lad_alt_binary_test)
        r2_lads_test_oct_no_p.append(r2_lad_alt_oct_no_p_test)
        r2_lads_test_oct_with_p.append(r2_lad_alt_oct_with_p_test)
        r2_lads_test_benders_oct_with_p.append(r2_lad_alt_benders_oct_with_p_test)

    print('mip gaps no_pi', mip_gaps_no_pi)
    print('mip gaps with_pi', mip_gaps_with_pi)
    print('mip gaps no_e_n', mip_gaps_no_e_n)
    print('mip gaps binary', mip_gaps_binary)
    print('mip gaps oct_no_p', mip_gaps_oct_no_p)
    print('mip gaps oct_with_p', mip_gaps_oct_with_p)
    print('mip gaps benders_oct_with_p', mip_gaps_benders_oct_with_p)
    print('\n\nsolving times no_pi', solving_times_no_pi)
    print('solving times with_pi', solving_times_with_pi)
    print('solving times no_e_n', solving_times_no_e_n)
    print('solving times binary', solving_times_binary)
    print('solving times oct_no_p', solving_times_oct_no_p)
    print('solving times oct_with_p', solving_times_oct_with_p)
    print('solving times benders_oct_with_p', solving_times_benders_oct_with_p)

    print('\n\norig obj no_pi', orig_obj_no_pi)
    print('orig obj with_pi', orig_obj_with_pi)
    print('orig obj no_e_n', orig_obj_no_e_n)
    print('orig obj binary', orig_obj_binary)
    print('orig obj oct_no_p', orig_obj_oct_no_p)
    print('orig obj oct_with_p', orig_obj_oct_with_p)
    print('orig obj benders_oct_with_p', orig_obj_benders_oct_with_p)

    print('\n\nmaes no_pi', maes_train_no_pi)
    print('maes with_pi', maes_train_with_pi)
    print('maes no_e_n', maes_train_no_e_n)
    print('maes binary', maes_train_binary)
    print('maes oct_no_p', maes_train_oct_no_p)
    print('maes oct_with_p', maes_train_oct_with_p)
    print('maes benders_oct_with_p', maes_train_benders_oct_with_p)
    print('\n\nmaes no_pi test', maes_test_no_pi)
    print('maes with_pi test', maes_test_with_pi)
    print('maes no_e_n test', maes_test_no_e_n)
    print('maes binary test', maes_test_binary)
    print('maes oct_no_p test', maes_test_oct_no_p)
    print('maes oct_with_p test', maes_test_oct_with_p)
    print('maes benders_oct_with_p test', maes_test_benders_oct_with_p)
    row_1 = [approach_name_1, input_file,train_len,features_count, depth, n_k_folds, time_limit, np.average(mip_gaps_no_pi) * 100,
             np.average(solving_times_no_pi), np.average(maes_train_no_pi), np.average(r2_lads_train_no_pi),
             np.average(maes_test_no_pi), np.average(r2_lads_test_no_pi)]
    row_2 = [approach_name_2, input_file,train_len,features_count, depth, n_k_folds, time_limit, np.average(mip_gaps_binary) * 100,
             np.average(solving_times_binary), np.average(maes_train_binary), np.average(r2_lads_train_binary),
             np.average(maes_test_binary), np.average(r2_lads_test_binary)]
    row_3 = [approach_name_3, input_file,train_len,features_count, depth, n_k_folds, time_limit, np.average(mip_gaps_oct_no_p) * 100,
             np.average(solving_times_oct_no_p), np.average(maes_train_oct_no_p), np.average(r2_lads_train_oct_no_p),
             np.average(maes_test_oct_no_p), np.average(r2_lads_test_oct_no_p)]

    row_4 = [approach_name_4, input_file,train_len,features_count, depth, n_k_folds, time_limit, np.average(mip_gaps_benders_oct_with_p) * 100,
             np.average(solving_times_benders_oct_with_p), np.average(maes_train_benders_oct_with_p),
             np.average(r2_lads_train_benders_oct_with_p),
             np.average(maes_test_benders_oct_with_p), np.average(r2_lads_test_benders_oct_with_p)]

    row_5 = [approach_name_5, input_file,train_len,features_count, depth, n_k_folds, time_limit, np.average(mip_gaps_with_pi) * 100,
             np.average(solving_times_with_pi), np.average(maes_train_with_pi), np.average(r2_lads_train_with_pi),
             np.average(maes_test_with_pi), np.average(r2_lads_test_with_pi)]

    row_6 = [approach_name_6, input_file,train_len,features_count, depth, n_k_folds, time_limit, np.average(mip_gaps_no_e_n) * 100,
             np.average(solving_times_no_e_n), np.average(maes_train_no_e_n), np.average(r2_lads_train_no_e_n),
             np.average(maes_test_no_e_n), np.average(r2_lads_test_no_e_n)]

    row_7 = [approach_name_3, input_file,train_len,features_count, depth, n_k_folds, time_limit, np.average(mip_gaps_oct_with_p) * 100,
             np.average(solving_times_oct_with_p), np.average(maes_train_oct_with_p), np.average(r2_lads_train_oct_with_p),
             np.average(maes_test_oct_with_p), np.average(r2_lads_test_oct_with_p)]

    result_file_no_pi = out_put_name_1 + '.csv'
    with open(out_put_path + result_file_no_pi, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            row_1)

    # writing info to the file
    result_file_binary = out_put_name_2 + '.csv'
    with open(out_put_path + result_file_binary, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_2)

    # writing info to the file
    result_file_oct_no_p = out_put_name_3 + '.csv'
    with open(out_put_path + result_file_oct_no_p, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_3)

    # writing info to the file
    result_file_benders_oct_with_p = out_put_name_4 + '.csv'
    with open(out_put_path + result_file_benders_oct_with_p, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_4)

    result_file_with_pi = out_put_name_5 + '.csv'
    with open(out_put_path + result_file_with_pi, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_5)

    result_file_no_e_n = out_put_name_6 + '.csv'
    with open(out_put_path + result_file_no_e_n, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_6)

    result_file_tot = out_put_name + '.csv'

    # writing info to the file
    result_file_oct_with_p = out_put_name_7 + '.csv'
    with open(out_put_path + result_file_oct_with_p, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_7)

    with open(out_put_path + result_file_tot, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_1)
        results_writer.writerow(row_2)
        results_writer.writerow(row_3)
        results_writer.writerow(row_4)
        results_writer.writerow(row_5)
        results_writer.writerow(row_6)
        results_writer.writerow(row_7)


if __name__ == "__main__":
    main(sys.argv[1:])
