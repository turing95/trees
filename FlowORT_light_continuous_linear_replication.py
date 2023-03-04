#!/usr/bin/python
from gurobipy import *
import pandas as pd
from datetime import date
import sys
import time
from Tree import Tree
from FlowORT_light_continuous_linear import FlowORT as FlowORT_light_continuous
import logger
import getopt
import csv
import numpy as np
from logger import logger
from sklearn.model_selection import KFold
from max_cut_node_means_pca import max_cut_node_means_pca,max_cut_node_means_pca_bottom_up
from initial_solution import get_initial_solution

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
    input_file = 'yacht_hydrodynamics_reg.csv'
    depth = 2
    time_limit = 3600
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
        time_limit) + '_continuous_constant_cross_validation'
    # Using logger we log the output of the console in a text file
    out_put_path = os.getcwd() + '/Results/'
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    approach_name_1 = 'FlowORT_light_continuous'
    out_put_name_1 = input_file + '_' + approach_name_1 + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_continuous_constant_cross_validation'

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
    maes_train_light = []

    maes_test_light = []

    mip_gaps_light = []

    solving_times_light = []

    r2_lads_train_light = []

    r2_lads_test_light = []
    r2_train = []
    r2_test = []

    n_k_folds = kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        ##########################################################
        # Creating and Solving the problem
        ##########################################################
        # We create the MIP problem by passing the required arguments

        start_time = time.time()
        primal_light = FlowORT_light_continuous(data_train, label, tree, time_limit)
        primal_light.create_primal_problem()
        primal_light.model.update()
        primal_light.model.optimize()
        primal_light.model.write(f'model_{time.time()}.mps')

        end_time = time.time()

        solving_time_light = end_time - start_time

        ##########################################################
        # Preparing the output

        lower_bound_reference = primal_light.model.getAttr("ObjBound")

        print("\n\n")
        primal_light.print_results(solving_time_light)

        print(f'\n\nlower_bound_reference {lower_bound_reference}')

        reg_res_light, mae_light, mse_light, r2_light, r2_lad_alt_light = primal_light.get_accuracy(data_train)
        reg_res_light_test, mae_light_test, mse_light_test, r2_light_test, r2_lad_alt_light_test = primal_light.get_accuracy(
            data_test)

        maes_train_light.append(mae_light)

        maes_test_light.append(mae_light_test)

        mip_gaps_light.append(primal_light.model.getAttr("MIPGap"))

        solving_times_light.append(solving_time_light)

        r2_lads_train_light.append(r2_lad_alt_light)

        r2_lads_test_light.append(r2_lad_alt_light_test)

        r2_train.append(r2_light)
        r2_test.append(r2_light_test)

    print('\n')
    print('mip gaps light', mip_gaps_light)
    print('\n')
    print('solving times light', solving_times_light)
    print('\n')
    print('maes light test', maes_test_light)

    print('\n')
    print('maes light train', maes_train_light)

    row_1 = [approach_name_1, input_file, train_len, features_count, depth, n_k_folds, time_limit,
             np.average(mip_gaps_light) * 100,
             np.average(solving_times_light), np.average(maes_train_light), np.average(r2_lads_train_light),
             np.average(r2_train),
             np.average(maes_test_light), np.average(r2_lads_test_light), np.average(r2_test)]

    result_file_light = out_put_name_1 + '.csv'
    with open(out_put_path + result_file_light, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_1)

    result_file_tot = out_put_name + '.csv'
    with open(out_put_path + result_file_tot, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_1)

    with open(out_put_path + f'res_continuous_{date.today()}.csv', mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(row_1)


if __name__ == "__main__":
    main(sys.argv[1:])
