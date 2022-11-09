'''
This module formulate the FlowOCT problem in gurobipy.
'''

from gurobipy import *

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
        self.beta = 0
        self.z = 0
        self.e = 0

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

        # zeta[i] potential at sink node t for point i
        self.zeta = self.model.addVars(self.datapoints, vtype=GRB.CONTINUOUS, lb=0,
                                       name='zeta')
        # z[i,n] potential at node n for point i
        self.z = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                    name='z')

        # TODO discretize leaves variable
        # e[i,n] is the amount of flow through the edge connecting node n to sink node t for datapoint i
        self.e = self.model.addVars(self.datapoints, self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                    name='e')
        # beta_zero[n] is the constant of the regression
        self.beta_zero = self.model.addVars(self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                            name='beta_zero')
        self.beta = self.model.addVars(self.tree.Leaves, self.cat_features, vtype=GRB.CONTINUOUS, name='beta')
        ############################### define constraints

        # 1a) e[i,n] >=sum( beta[n,f]*x[i,f]) - y[i]  forall i, n in Leaves
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (self.e[i, n] >= self.beta_zero[n] + quicksum(
                    self.beta[n, f] * self.data.at[i, f] for f in self.cat_features) - self.data.at[i, self.label]) for
                i in self.datapoints)
        self.model.addConstrs(self.z[i, 1] == 0 for i in self.datapoints)
        #  1b) -e[i,n] <= sum( beta[n,f]*x[i,f]) - y[i]  forall i, n in Leaves
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (-self.e[i, n] <= self.beta_zero[n] + quicksum(
                    self.beta[n, f] * self.data.at[i, f] for f in self.cat_features) - self.data.at[i, self.label]) for
                i in self.datapoints)

        # 2) zeta[i] - z[i,n] >= e[i,n] - M*(D-z[i,n])  forall i, n in Leaves
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (self.zeta[i] - self.z[i, n] >= self.e[i, n] - self.big_m * (self.d - self.z[i, n])) for i in
                self.datapoints)
        # 3) zeta[i] - d <= e[i,n]
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (self.zeta[i] - self.d <= self.e[i, n]) for i in
                self.datapoints)
        # 4) e[i,n] >= M*(D-z[i,n])  forall i, n in Leaves
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (self.e[i, n] >= self.big_m * (self.d - self.z[i, n])) for i in
                self.datapoints)

        # 5) z[i,l(n)] - z[i,n] == sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_left_children(n))] - self.z[i, n] == quicksum(
                self.b[n, f] for f in self.cat_features if self.data.at[i, f] == 0)) for n in self.tree.Nodes)

        # 6) z[i,r(n)] - z[i,n] == sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_right_children(n))] - self.z[i, n] == quicksum(
                self.b[n, f] for f in self.cat_features if self.data.at[i, f] == 1)) for n in self.tree.Nodes)

        # 7) sum(b[n,f], f) = 1   forall n in Nodes
        self.model.addConstrs(
            (quicksum(self.b[n, f] for f in self.cat_features) == 1) for n in
            self.tree.Nodes)

        # self.model.addConstrs(self.zeta[i] >= self.d for i in self.datapoints)

        # sum(z[i,n] forall n in L+N[2^l:2^(l+1))>= level
        '''for level in range(self.d):
            if level == 0:
                continue
            nodes = self.tree.Nodes+self.tree.Leaves
            self.model.addConstrs(
                quicksum(self.z[i, n] for n in nodes[(np.power(2, level)-1):(np.power(2, level + 1))-1])
                == level + (sum(np.power(2, level - l_ - 1) * l_ for l_ in range(level)))
                for i in self.datapoints)'''

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            for n in self.tree.Leaves:
                obj.add(self.e[i, n])

        self.model.setObjective(obj, GRB.MINIMIZE)
