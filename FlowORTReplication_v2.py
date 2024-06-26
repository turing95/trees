#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from FlowORT_v2 import FlowORT
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
    approach_name = 'FlowORT_binary'
    out_put_name = input_file + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
        time_limit)
    out_put_path = os.getcwd() + '/Results/'
    # Using logger we log the output of the console in a text file
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # data splitting
    ##########################################################
    data_train = data
    train_len = len(data_train.index)

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
    #zeta = primal.model.getAttr("x", primal.zeta)
    z = primal.model.getAttr("x", primal.z)
    e = primal.model.getAttr("x", primal.e)

    print("\n\n")
    print_tree(primal, b_value, 0)

    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.model.getAttr("ObjVal"))

    print('Total Callback counter (Integer)', primal.model._callback_counter_integer)
    print('Total Successful Callback counter (Integer)', primal.model._callback_counter_integer_success)

    print('Total Callback Time (Integer)', primal.model._total_callback_time_integer)
    print('Total Successful Callback Time (Integer)', primal.model._total_callback_time_integer_success)
    print('b value')
    print(b_value)
    print('#####')
    print('beta_zero value')
    print(beta_zero)
    print('#####')
    print('z value')
    print(z)
    print('#####')
    print('e value')
    print(e)


    print("obj value", primal.model.getAttr("ObjVal"))
    # def get_model_accuracy(data, datapoints, z, beta_zero, depth, label)
    acc = get_model_accuracy(data, primal.datapoints, z, beta_zero, depth, label)
    print('accuracy', acc)
    print('\n\nTotal Solving Time', solving_time)

    # writing info to the file
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            [approach_name, input_file, train_len, depth, time_limit,
             primal.model.getAttr("Status"), primal.model.getAttr("ObjVal"),
             primal.model.getAttr("MIPGap") * 100, primal.model.getAttr("NodeCount"), solving_time,
             acc])


if __name__ == "__main__":
    main(sys.argv[1:])
