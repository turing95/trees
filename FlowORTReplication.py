#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
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
    _lambda = None
    input_sample = None
    calibration = None
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:m:",
                                   ["input_file=", "depth=", "timelimit=", "lambda=",
                                    "input_sample=",
                                    "calibration=", "mode="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--input_file"):
            input_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-l", "--lambda"):
            _lambda = float(arg)
        elif opt in ("-i", "--input_sample"):
            input_sample = int(arg)
        elif opt in ("-c", "--calibration"):
            calibration = int(arg)

    start_time = time.time()
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
    approach_name = 'FlowOCT'
    out_put_name = input_file + '_' + str(input_sample) + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_lambda_' + str(
        _lambda) + '_c_' + str(calibration)
    out_put_path = os.getcwd() + '/Results/'
    # Using logger we log the output of the console in a text file
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # data splitting
    ##########################################################
    '''
    Creating  train, test and calibration datasets
    We take 50% of the whole data as training, 25% as test and 25% as calibration

    When we want to calibrate _lambda, for a given value of _lambda we train the model on train and evaluate
    the accuracy on calibration set and at the end we pick the _lambda with the highest accuracy.

    When we got the calibrated _lambda, we train the mode on (train+calibration) which we refer to it as
    data_train_calibration and evaluate the accuracy on (test)

    '''
    data_train = data
    # Creating and Solving the problem
    ##########################################################
    # We create the MIP problem by passing the required arguments
    primal = FlowORT(data_train, label, tree, time_limit)

    primal.create_primal_problem()
    primal.model.update()
    primal.model.optimize()

    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Preparing the output
    ##########################################################
    b_value = primal.model.getAttr("X", primal.b)
    beta_zero = primal.model.getAttr("x", primal.beta_zero)
    zeta = primal.model.getAttr("x", primal.zeta)
    z = primal.model.getAttr("x", primal.z)
    e = primal.model.getAttr("x", primal.e)

    print("\n\n")
    print_tree(primal, b_value, 0)

    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.model.getAttr("ObjVal"))
    print('b value')
    print(b_value)
    print('#####')
    print('beta_zero value')
    print(beta_zero)
    print('#####')
    print('zeta value')
    print(zeta)
    print('#####')
    print('z value')
    print(z)
    print('#####')
    print('e value')
    print(e)

    print("obj value", primal.model.getAttr("ObjVal"))
    # sklearn.metrics.r2_score(y_true, y_pred) y_true = y[i], y_pred = beta_zero at maximal_potential_node

if __name__ == "__main__":
    main(sys.argv[1:])
