#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from FlowORT_v2 import FlowORT as FlowORT_v2
from FlowORT import FlowORT
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
    input_sample = None
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:m:",
                                   ["input_file=", "depth=", "timelimit=",
                                    "input_sample="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--input_file"):
            input_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-i", "--input_sample"):
            input_sample = int(arg)

    start_time = time.time()
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
    approach_name = 'FlowORT'
    out_put_name = input_file + '_' + str(input_sample) + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
        time_limit)
    out_put_path = os.getcwd() + '/Results/'
    # Using logger we log the output of the console in a text file
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

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

    train_len = len(data_train.index)'''
    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    # We create the MIP problem by passing the required arguments
    primal = FlowORT(data, label, tree, time_limit)
    primal_v2 = FlowORT_v2(data, label, tree, time_limit)

    primal.create_primal_problem()
    primal.model.update()
    primal.model.optimize()

    primal_v2.create_primal_problem()
    primal_v2.model.update()
    primal_v2.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Preparing the output
    ##########################################################
    b_value = primal.model.getAttr("X", primal.b)
    beta_zero = primal.model.getAttr("x", primal.beta_zero)
    beta_zero_v2 = primal_v2.model.getAttr("x", primal_v2.beta_zero)
    # zeta = primal.model.getAttr("x", primal.zeta)
    # p = primal.model.getAttr("x", primal.p)
    z = primal.model.getAttr("x", primal.z)
    z_v2 = primal_v2.model.getAttr("x", primal_v2.z)
    e = primal.model.getAttr("x", primal.e)

    print("\n\n")
    print_tree(primal, b_value, 0)

    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.model.getAttr("ObjVal"))
    print(primal.datapoints)
    print(primal_v2.datapoints)
    print('pi_n v1')
    print(z)
    print('#####')
    print('pi_n v2')
    print(z_v2)


    max_values = []
    max_values_v2 = []
    print(f'V1 {beta_zero}')
    print(f'V2 {beta_zero_v2}')
    for key, value in beta_zero.items():
        if value != beta_zero_v2[key]:
            print(f'beta_zero differring: V1 {key} : {value} V2  {key} : {beta_zero_v2[key]}')

    for i in primal.datapoints:
        max_value = -1
        node = None
        for t in range(1, np.power(2, depth + 1)):
            if max_value < z[i, t]:
                node = t
                max_value = z[i, t]
        max_values.append(node)
    for i in primal.datapoints:
        max_value = -1
        node = None
        for t in range(1, np.power(2, depth + 1)):
            if max_value < z_v2[i, t]:
                node = t
                max_value = z_v2[i, t]
        max_values_v2.append(node)
    print(max_values)
    print(max_values_v2)
    assert max_values == max_values_v2


if __name__ == "__main__":
    main(sys.argv[1:])
