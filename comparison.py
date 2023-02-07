#!/usr/bin/python
from gurobipy import *
import pandas as pd
from datetime import date
import sys
import time
from Tree import Tree
from FlowORT_light import FlowORT as FlowORT_light
from FlowORT_light_e_n import FlowORT as FlowORT_light_e_n
from FlowOCT_with_p import FlowOCT as FlowOCT_with_p
from BendersOCT import BendersOCT as BendersOCT_with_p
import logger
import getopt
import csv
import numpy as np
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
    out_put_name = input_file + '_' + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'
    # Using logger we log the output of the console in a text file
    out_put_path = os.getcwd() + '/Results/'
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    approach_name_4 = 'BendersOCT_with_p'
    out_put_name_4 = input_file + '_' + approach_name_4 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_6 = 'FlowORT_light_lazy'
    out_put_name_6 = input_file + '_' + approach_name_6 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_7 = 'FlowOCT_with_p'
    out_put_name_7 = input_file + '_' + approach_name_7 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_constant_cross_validation'

    approach_name_8 = 'FlowORT_light_e_n_lazy'
    out_put_name_8 = input_file + '_' + approach_name_8 + '_d_' + str(depth) + '_t_' + str(
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
    features_count = len(data.columns) - 1
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # k_folds = 5 if train_len < 300 else 10
    k_folds = 5
    random_state = 1
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    maes_train_oct_with_p = []
    maes_train_benders_oct_with_p = []
    maes_train_light = []
    maes_train_light_e_n = []

    maes_test_oct_with_p = []
    maes_test_benders_oct_with_p = []
    maes_test_light = []
    maes_test_light_e_n = []

    mip_gaps_oct_with_p = []
    mip_gaps_benders_oct_with_p = []
    mip_gaps_light = []
    mip_gaps_light_e_n = []

    solving_times_oct_with_p = []
    solving_times_benders_oct_with_p = []
    solving_times_light = []
    solving_times_light_e_n = []

    r2_lads_train_oct_with_p = []
    r2_lads_train_benders_oct_with_p = []
    r2_lads_train_light = []
    r2_lads_train_light_e_n = []

    r2_lads_test_oct_with_p = []
    r2_lads_test_benders_oct_with_p = []
    r2_lads_test_light = []
    r2_lads_test_light_e_n = []

    n_k_folds = kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        ##########################################################
        # Creating and Solving the problem
        ##########################################################
        # We create the MIP problem by passing the required arguments

        start_time = time.time()
        primal_light = FlowORT_light(data_train, label, tree, time_limit)

        primal_light.create_primal_problem()
        primal_light.model.update()
        primal_light.model.optimize()

        end_time = time.time()

        solving_time_light = end_time - start_time

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

        start_time = time.time()
        primal_light_e_n = FlowORT_light_e_n(data_train, label, tree, time_limit)

        primal_light_e_n.create_primal_problem()
        primal_light_e_n.model.update()
        primal_light_e_n.model.optimize()

        end_time = time.time()

        solving_time_light_e_n = end_time - start_time
        ##########################################################
        # Preparing the output

        lower_bound_reference = primal_light.model.getAttr("ObjBound")

        print("\n\n")
        primal_light.print_results(solving_time_light)
        primal_light_e_n.print_results(solving_time_light_e_n)
        primal_oct_with_p.print_results(solving_time_oct_with_p)
        primal_benders_oct_with_p.print_results(solving_time_benders_oct_with_p)
        print(f'\n\nlower_bound_reference {lower_bound_reference}')

        reg_res_light, mae_light, mse_light, r2_light, r2_lad_alt_light = primal_light.get_accuracy(data_train)
        reg_res_light_test, mae_light_test, mse_light_test, r2_light_test, r2_lad_alt_light_test = primal_light.get_accuracy(
            data_test)

        reg_res_light_e_n, mae_light_e_n, mse_light_e_n, r2_light_e_n, r2_lad_alt_light_e_n = primal_light_e_n.get_accuracy(
            data_train)
        reg_res_light_e_n_test, mae_light_e_n_test, mse_light_e_n_test, r2_light_e_n_test, r2_lad_alt_light_e_n_test = primal_light_e_n.get_accuracy(
            data_test)

        reg_res_oct_with_p, mae_oct_with_p, mse_oct_with_p, r2_oct_with_p, r2_lad_alt_oct_with_p = primal_oct_with_p.get_accuracy(
            data_train)

        _, mae_oct_with_p_test, _, _, r2_lad_alt_oct_with_p_test = primal_oct_with_p.get_accuracy(data_test)

        reg_res_benders_oct_with_p, mae_benders_oct_with_p, mse_benders_oct_with_p, r2_benders_oct_with_p, r2_lad_alt_benders_oct_with_p = primal_benders_oct_with_p.get_accuracy(
            data_train)

        _, mae_benders_oct_with_p_test, _, _, r2_lad_alt_benders_oct_with_p_test = primal_benders_oct_with_p.get_accuracy(
            data_test)
        maes_train_light.append(mae_light)
        maes_train_light_e_n.append(mae_light_e_n)
        maes_train_oct_with_p.append(mae_oct_with_p)
        maes_train_benders_oct_with_p.append(mae_benders_oct_with_p)

        maes_test_light.append(mae_light_test)
        maes_test_light_e_n.append(mae_light_e_n_test)
        maes_test_oct_with_p.append(mae_oct_with_p_test)
        maes_test_benders_oct_with_p.append(mae_benders_oct_with_p_test)

        mip_gaps_light.append(primal_light.model.getAttr("MIPGap") * 100)
        mip_gaps_light_e_n.append(primal_light_e_n.model.getAttr("MIPGap") * 100)
        mip_gaps_oct_with_p.append(primal_oct_with_p.model.getAttr("MIPGap") * 100)
        mip_gaps_benders_oct_with_p.append(primal_benders_oct_with_p.model.getAttr("MIPGap") * 100)

        solving_times_light.append(solving_time_light)
        solving_times_light_e_n.append(solving_time_light_e_n)
        solving_times_oct_with_p.append(solving_time_oct_with_p)
        solving_times_benders_oct_with_p.append(solving_time_benders_oct_with_p)

        r2_lads_train_light.append(r2_lad_alt_light)
        r2_lads_train_light_e_n.append(r2_lad_alt_light_e_n)
        r2_lads_train_oct_with_p.append(r2_lad_alt_oct_with_p)
        r2_lads_train_benders_oct_with_p.append(r2_lad_alt_benders_oct_with_p)

        r2_lads_test_light.append(r2_lad_alt_light_test)
        r2_lads_test_light_e_n.append(r2_lad_alt_light_e_n_test)
        r2_lads_test_oct_with_p.append(r2_lad_alt_oct_with_p_test)
        r2_lads_test_benders_oct_with_p.append(r2_lad_alt_benders_oct_with_p_test)
    print('\n')
    print('mip gaps light', mip_gaps_light)
    print('mip gaps light_e_n', mip_gaps_light_e_n)
    print('mip gaps oct_with_p', mip_gaps_oct_with_p)
    print('mip gaps benders_oct_with_p', mip_gaps_benders_oct_with_p)
    print('\n')
    print('solving times light', solving_times_light)
    print('solving times light_e_n', solving_times_light_e_n)
    print('solving times oct_with_p', solving_times_oct_with_p)
    print('solving times benders_oct_with_p', solving_times_benders_oct_with_p)
    print('\n')
    print('maes light', maes_train_light)
    print('maes light_e_n', maes_train_light_e_n)
    print('maes oct_with_p', maes_train_oct_with_p)
    print('maes benders_oct_with_p', maes_train_benders_oct_with_p)
    print('\n')
    print('maes light test', maes_test_light)
    print('maes light_e_n test', maes_test_light_e_n)
    print('maes oct_with_p test', maes_test_oct_with_p)
    print('maes benders_oct_with_p test', maes_test_benders_oct_with_p)

    row_4 = [approach_name_4, input_file, train_len, features_count, depth, n_k_folds, time_limit,
             np.average(mip_gaps_benders_oct_with_p) * 100,
             np.average(solving_times_benders_oct_with_p), np.average(maes_train_benders_oct_with_p),
             np.average(r2_lads_train_benders_oct_with_p),
             np.average(maes_test_benders_oct_with_p), np.average(r2_lads_test_benders_oct_with_p)]

    row_6 = [approach_name_6, input_file, train_len, features_count, depth, n_k_folds, time_limit,
             np.average(mip_gaps_light) * 100,
             np.average(solving_times_light), np.average(maes_train_light), np.average(r2_lads_train_light),
             np.average(maes_test_light), np.average(r2_lads_test_light)]

    row_7 = [approach_name_7, input_file, train_len, features_count, depth, n_k_folds, time_limit,
             np.average(mip_gaps_oct_with_p) * 100,
             np.average(solving_times_oct_with_p), np.average(maes_train_oct_with_p),
             np.average(r2_lads_train_oct_with_p),
             np.average(maes_test_oct_with_p), np.average(r2_lads_test_oct_with_p)]

    row_8 = [approach_name_8, input_file, train_len, features_count, depth, n_k_folds, time_limit,
             np.average(mip_gaps_light_e_n) * 100,
             np.average(solving_times_light_e_n), np.average(maes_train_light_e_n), np.average(r2_lads_train_light_e_n),
             np.average(maes_test_light_e_n), np.average(r2_lads_test_light_e_n)]

    # writing info to the file
    result_file_benders_oct_with_p = out_put_name_4 + '.csv'
    with open(out_put_path + result_file_benders_oct_with_p, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_4)

    result_file_light = out_put_name_6 + '.csv'
    with open(out_put_path + result_file_light, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_6)

    result_file_tot = out_put_name + '.csv'

    # writing info to the file
    result_file_oct_with_p = out_put_name_7 + '.csv'
    with open(out_put_path + result_file_oct_with_p, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_7)

    result_file_light_e_n = out_put_name_8 + '.csv'
    with open(out_put_path + result_file_light_e_n, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_8)

    result_file_tot = out_put_name + '.csv'
    with open(out_put_path + result_file_tot, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_4)
        results_writer.writerow(row_6)
        results_writer.writerow(row_7)
        results_writer.writerow(row_8)

    with open(out_put_path + f'res_{date.today()}.csv', mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_4)
        results_writer.writerow(row_6)
        results_writer.writerow(row_7)
        results_writer.writerow(row_8)


if __name__ == "__main__":
    main(sys.argv[1:])
