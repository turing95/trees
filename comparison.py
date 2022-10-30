#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from FlowORT import FlowORT
from FlowORT_v2 import FlowORT as FlowORT_v2
import logger
import getopt
import csv
from sklearn.model_selection import train_test_split
from utils import *
from logger import logger


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
        time_limit)
    # Using logger we log the output of the console in a text file
    out_put_path = os.getcwd() + '/Results/'
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    out_put_name_1 = input_file + '_' + approach_name_1 + '_d_' + str(depth) + '_t_' + str(
        time_limit)

    approach_name_2 = 'FlowORT_binary'
    out_put_name_2 = input_file + '_' + approach_name_2 + '_d_' + str(depth) + '_t_' + str(
        time_limit)

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
    data_train = data
    train_len = len(data_train.index)
    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    # We create the MIP problem by passing the required arguments
    start_time = time.time()
    primal_v1 = FlowORT(data, label, tree, time_limit)

    primal_v1.create_primal_problem()
    primal_v1.model.update()
    primal_v1.model.optimize()

    end_time = time.time()

    solving_time_v1 = end_time - start_time
    start_time = time.time()

    primal_v2 = FlowORT_v2(data, label, tree, time_limit)

    primal_v2.create_primal_problem()
    primal_v2.model.update()
    primal_v2.model.optimize()
    end_time = time.time()

    solving_time_v2 = end_time - start_time

    ##########################################################
    # Preparing the output
    ##########################################################
    b_value_v1 = primal_v1.model.getAttr("X", primal_v1.b)
    b_value_v2 = primal_v2.model.getAttr("X", primal_v2.b)
    beta_zero_v1 = primal_v1.model.getAttr("x", primal_v1.beta_zero)
    beta_zero_v2 = primal_v2.model.getAttr("x", primal_v2.beta_zero)
    # zeta = primal.model.getAttr("x", primal.zeta)
    # p = primal.model.getAttr("x", primal.p)
    z_v1 = primal_v1.model.getAttr("x", primal_v1.z)
    z_v2 = primal_v2.model.getAttr("x", primal_v2.z)
    e_v1 = primal_v1.model.getAttr("x", primal_v1.e)
    e_v2 = primal_v2.model.getAttr("x", primal_v2.e)

    print("\n\n")
    print('\n\nTotal Solving Time v1', solving_time_v1)
    print('Total Solving Time v2', solving_time_v2)
    print("\n\nobj value v1", primal_v1.model.getAttr("ObjVal"))
    print("obj value v2", primal_v2.model.getAttr("ObjVal"))
    print('\n\nbnf_v1', b_value_v1)
    print('bnf_v2', b_value_v2)
    print('\n\npi_n v1')
    print(z_v1)
    print('#####')
    print('pi_n v2')
    print(z_v2)

    max_values = []
    max_values_v2 = []
    print(f'\n\nbeta_zero V1 {beta_zero_v1}')
    print(f'beta_zero V2 {beta_zero_v2}')
    beta_zeros_differing = False
    for key, value in beta_zero_v1.items():
        if value != beta_zero_v2[key]:
            print(f'beta_zero differring: V1 {key} : {value} V2  {key} : {beta_zero_v2[key]}')
            beta_zeros_differing = True

    for i in primal_v1.datapoints:
        max_value = -1
        node = None
        for t in range(1, np.power(2, depth + 1)):
            if max_value < z_v1[i, t]:
                node = t
                max_value = z_v1[i, t]
        max_values.append(node)
    for i in primal_v2.datapoints:
        max_value = -1
        node = None
        for t in range(1, np.power(2, depth + 1)):
            if max_value < z_v2[i, t]:
                node = t
                max_value = z_v2[i, t]
        max_values_v2.append(node)
    print('\n\nmax_values v1', max_values)
    print('max_values v1', max_values_v2)
    max_value_node_differring = False
    if max_values != max_values_v2:
        max_value_node_differring = True

    r2_v1, mse_v1, mae_v1, r2_lad_v1, r2_lad_alt_v1,mean_v1,median_v1 = get_model_accuracy(data, primal_v1.datapoints, z_v1, beta_zero_v1,
                                                                     depth, label)
    r2_v2, mse_v2, mae_v2, r2_lad_v2, r2_lad_alt_v2,mean_v2,median_v2 = get_model_accuracy(data, primal_v2.datapoints, z_v2, beta_zero_v2,
                                                                     depth, label)

    # writing info to the file
    result_file_v1 = out_put_name_1 + '.csv'
    with open(out_put_path + result_file_v1, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            [approach_name_1, input_file, train_len, depth, time_limit,
             primal_v1.model.getAttr("Status"), primal_v1.model.getAttr("ObjVal"),
             primal_v1.model.getAttr("MIPGap") * 100, primal_v1.model.getAttr("NodeCount"), solving_time_v1,
             r2_v1, mse_v1, mae_v1, r2_lad_v1, r2_lad_alt_v1, max_value_node_differring, beta_zeros_differing,mean_v1,median_v1])

    # writing info to the file
    result_file_v2 = out_put_name_2 + '.csv'
    with open(out_put_path + result_file_v2, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            [approach_name_2, input_file, train_len, depth, time_limit,
             primal_v2.model.getAttr("Status"), primal_v2.model.getAttr("ObjVal"),
             primal_v2.model.getAttr("MIPGap") * 100, primal_v2.model.getAttr("NodeCount"), solving_time_v2,
             r2_v2, mse_v2, mae_v2, r2_lad_v2, r2_lad_alt_v2, max_value_node_differring, beta_zeros_differing,mean_v2,median_v2])


if __name__ == "__main__":
    main(sys.argv[1:])
