'''
This module formulate the FlowOCT problem in gurobipy.
'''

from gurobipy import *
from utils.utils_oct_no_p import get_model_accuracy
import numpy as np


class FlowORT:
    def __init__(self, data, label, tree, time_limit):
        '''

        :param data: The training data
        :param label: Name of the column representing the class label
        :param tree: Tree object
        :param time_limit: The given time limit for solving the MIP
        '''

        self.data = data
        self.datapoints = data.index
        self.label = label

        '''if self.mode == "classification":
            self.labels = data[label].unique()
        elif self.mode == "regression":
            self.labels = [1]'''

        '''
        cat_features is the set of all categorical features. 
        reg_features is the set of all features used for the linear regression prediction model in the leaves.  
        '''
        self.cat_features = self.data.columns[self.data.columns != self.label]
        # self.reg_features = None
        # self.num_of_reg_features = 1

        self.tree = tree

        # parameters
        y_max = None
        y_min = None
        # M = maxyi- minyi
        for i in self.datapoints:
            y_i = self.data.at[i, self.label]
            if y_max is None or y_i > y_max:
                y_max = y_i
            if y_min is None or y_i < y_min:
                y_min = y_i
        self.big_m = y_max - y_min
        self.d = self.tree.depth
        # Decision Variables
        self.b = 0
        self.beta_zero = 0
        self.zeta = 0
        self.e = 0
        self.beta = None

        # Gurobi model
        self.model = Model('FlowORT')
        '''
        To compare all approaches in a fair setting we limit the solver to use only one thread to merely evaluate 
        the strength of the formulation.
        '''
        self.model.params.Threads = 1
        self.model.params.TimeLimit = time_limit

        '''
        The following variables are used for the Benders problem to keep track of the times we call the callback.
        They are not used for this formulation.
        '''
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0

        self.model._total_callback_time_general = 0
        self.model._total_callback_time_general_success = 0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0

    ###########################################################
    # Create the MIP formulation
    ###########################################################
    def create_primal_problem(self):
        '''
        This function create and return a gurobi model formulating the FlowORT problem
        :return:  gurobi model object with the FlowOCT formulation
        '''
        ############################### define variables
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes, self.cat_features, vtype=GRB.BINARY, name='b')

        # TODO discretize leaves variable
        # e[i,n] is the amount of flow through the edge connecting node n to sink node t for datapoint i
        self.e = self.model.addVars(self.datapoints, vtype=GRB.CONTINUOUS, lb=0,
                                    name='e')
        # beta_zero[n] is the constant of the regression
        self.beta_zero = self.model.addVars(self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                            name='beta_zero')

        ############################### define constraints

        def path_length(node, i):
            path_len = 0
            while node is not None:
                if node == 1:
                    break
                parent = self.tree.get_parent(node)
                for f in self.cat_features:
                    feature_value = self.data.at[i, f]
                    path_len += 1 * self.b[parent, f] * ((node % 2) == feature_value)

                node = parent
            return path_len

        # 1a) e[i,n] >=sum( beta[n,f]*x[i,f]) - y[i]  forall i, n in Leaves
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (self.e[i] + self.big_m * (self.d - path_length(n, i)) >= self.beta_zero[n] - self.data.at[
                    i, self.label]) for
                i in self.datapoints)
        #  1b) -e[i,n] <= sum( beta[n,f]*x[i,f]) - y[i]  forall i, n in Leaves
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (-self.e[i] - self.big_m * (self.d - path_length(n, i)) <= self.beta_zero[n] - self.data.at[
                    i, self.label]) for
                i in self.datapoints)


        # 7) sum(b[n,f], f) = 1   forall n in Nodes
        self.model.addConstrs(
            (quicksum(self.b[n, f] for f in self.cat_features) == 1) for n in
            self.tree.Nodes)

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add(self.e[i])

        self.model.setObjective(obj, GRB.MINIMIZE)

    def print_results(self, solving_time):
        print('Total Solving Time light', solving_time)
        print("obj value light", self.model.getAttr("ObjVal"))
        print('bnf_light', self.model.getAttr("X", self.b))
        print(f'beta_zero light', self.model.getAttr("x", self.beta_zero))

    def get_accuracy(self, data):

        return get_model_accuracy(self,
                                  data,
                                  self.model.getAttr("X", self.b),
                                  self.model.getAttr("x", self.beta_zero),
                                  None)
