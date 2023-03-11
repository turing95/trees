#!/usr/bin/python
from gurobipy import *
import pandas as pd
from datetime import date
import sys
import time
from Tree import Tree
from sklearn.linear_model import LinearRegression
import logger
import getopt
from logger import logger
from sklearn.model_selection import KFold
import numpy as np


def main(argv):
    print(argv)
    input_file = None
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    input_file = 'yacht_hydrodynamics_reg_stand.csv'
    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:m:",
                                   ["input_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--input_file"):
            input_file = arg

    data_path = os.getcwd() + '/DataSets/'
    data = pd.read_csv(data_path + input_file)



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
    x = data.iloc[:, :-1]
    # k_folds = 5 if train_len < 300 else 10
    k_folds = 5
    random_state = 1
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    tr = []
    ts = []
    for train_index, test_index in kf.split(x):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        lg = LinearRegression()
        lg.fit(data_train.iloc[:,:-1], data_train.iloc[:,-1:])
        tr.append(lg.score(data_train.iloc[:,:-1], data_train.iloc[:,-1:]))
        ts.append(lg.score(data_test.iloc[:,:-1], data_test.iloc[:,-1:]))

    print('average train',np.average(tr))
    print('average testing',np.average(ts))


if __name__ == "__main__":
    main(sys.argv[1:])
