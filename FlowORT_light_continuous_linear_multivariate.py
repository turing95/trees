'''
This module formulate the FlowOCT problem in gurobipy.
'''

from gurobipy import *
from utils.utils_oct_no_p import get_model_accuracy
import time


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
        self.features = self.data.columns
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
        self.big_m = len(data.index)
        self.w = 0.0005
        self.d = self.tree.depth
        # Decision Variables
        self.b = 0
        self.beta_zero = 0
        self.zeta = 0
        self.e = 0
        self.a = 0
        self.g = 0
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
    def create_primal_problem(self, initial_a_b=None, init_beta_beta_zero=None, init_e_i_n=None, init_g_i_n=None):
        '''
        This function create and return a gurobi model formulating the FlowORT problem
        :return:  gurobi model object with the FlowOCT formulation
        '''
        ############################### define variables
        # b[n,f] ==1 iff at node n we branch on feature f
        self.a = self.model.addVars(self.tree.Nodes, self.features, vtype=GRB.CONTINUOUS, name='a', lb=-1, ub=1)
        self.g = self.model.addVars(self.datapoints, self.tree.Leaves, vtype=GRB.BINARY, name='g')

        self.e = self.model.addVars(self.datapoints, vtype=GRB.CONTINUOUS, lb=0,
                                    name='e')

        self.b = self.model.addVars(self.tree.Nodes, vtype=GRB.CONTINUOUS, name='b',lb=-1, ub=1)
        self.beta_zero = self.model.addVars(self.tree.Leaves, vtype=GRB.CONTINUOUS, name='beta_zero',lb=-GRB.INFINITY,ub=GRB.INFINITY)
        self.beta = self.model.addVars(self.tree.Leaves, self.cat_features, vtype=GRB.CONTINUOUS, name='beta',lb=-GRB.INFINITY,ub=GRB.INFINITY)

        ############################### define constraints
        if initial_a_b is not None:
            for n in self.tree.Nodes:
                v = initial_a_b[n]
                self.b[n].Start = -v[1]
                for idx, f in enumerate(self.features):
                    self.a[n, f].Start = v[0][idx]
        for n in self.tree.Leaves:
            if init_beta_beta_zero is not None:
                v = init_beta_beta_zero[n]
                try:
                    self.beta_zero[n].Start = v['intercept']
                except Exception as e:
                    print(v)
                    continue
                for idx, f in enumerate(self.cat_features):
                    self.beta[n, f].Start = v['coef'][idx]

            if init_e_i_n is not None:
                for i in self.datapoints:
                    self.g[i, n].Start = init_g_i_n[i][n]

                    try:
                        e_value = init_e_i_n[i][n]
                        self.e[i].Start = abs(e_value)
                    except KeyError:
                        continue

        # 1a) e[i,n] >=sum( beta[n,f]*x[i,f]) - y[i]  forall i, n in Leaves'''
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (self.e[i] + self.big_m * (1 - self.g[i, n]) >= self.beta_zero[n] + quicksum(
                    self.beta[n, f] * self.data.at[i, f] for f in self.cat_features) - self.data.at[
                     i, self.label]) for
                i in self.datapoints)
        #  1b) -e[i,n] <= sum( beta[n,f]*x[i,f]) - y[i]  forall i, n in Leaves
        for n in self.tree.Leaves:
            self.model.addConstrs(
                (-self.e[i] - self.big_m * (1 - self.g[i, n]) <= self.beta_zero[n] + quicksum(
                    self.beta[n, f] * self.data.at[i, f] for f in self.cat_features) - self.data.at[
                     i, self.label]) for
                i in self.datapoints)

        for n in self.tree.Nodes:
            left_leaves = self.tree.get_left_leaves(n)
            # right_leaves = self.tree.get_right_leaves(n)
            # no_reach = [x for x in self.tree.Leaves if x not in right_leaves + left_leaves]

            self.model.addConstrs(
                (quicksum(self.a[n, f] * self.data.at[i, f] for f in
                          self.features) + self.w <=
                 self.b[n] + (2 + self.w) * (1 - quicksum(self.g[i, x] for x in left_leaves))
                 )
                for i in
                self.datapoints)

        for n in self.tree.Nodes:
            # left_leaves = self.tree.get_left_leaves(n)
            right_leaves = self.tree.get_right_leaves(n)
            # no_reach = [x for x in self.tree.Leaves if x not in right_leaves + left_leaves]
            self.model.addConstrs(
                (quicksum(self.a[n, f] * (self.data.at[i, f]) for f in
                          self.features) >=
                 self.b[n] - 2 * (1 - quicksum(self.g[i, x] for x in right_leaves))
                 ) for i in
                self.datapoints)

        '''for n in self.tree.Nodes:
            left_leaves = self.tree.get_left_leaves(n)
            # right_leaves = self.tree.get_right_leaves(n)
            # no_reach = [x for x in self.tree.Leaves if x not in right_leaves + left_leaves]

            self.model.addConstrs(
                quicksum(self.a[n, f] * self.data.at[i, f] for f in
                         self.cat_features)+self.w <=
                self.b[n] + (1 + len(self.cat_features)+self.w) * (1 - quicksum(self.g[i, x] for x in left_leaves))
                for i in
                self.datapoints)

        for n in self.tree.Nodes:
            # left_leaves = self.tree.get_left_leaves(n)
            right_leaves = self.tree.get_right_leaves(n)
            # no_reach = [x for x in self.tree.Leaves if x not in right_leaves + left_leaves]
            self.model.addConstrs(
                (quicksum(self.a[n, f] * (self.data.at[i, f]) for f in
                          self.cat_features) >=
                 self.b[n] - (1 + len(self.cat_features)) * (1 - quicksum(self.g[i, x] for x in right_leaves))
                 ) for i in
                self.datapoints)'''

        self.model.addConstrs(
            (quicksum(self.g[i, n] for n in self.tree.Leaves) == 1) for i in self.datapoints)
        self.model.addConstrs(
            (quicksum(self.a[n, f] for f in self.features) <= 1) for n in self.tree.Nodes)
        self.model.addConstrs(
            (quicksum(self.a[n, f] for f in self.features) >= -1) for n in self.tree.Nodes)

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add(self.e[i])

        self.model.setObjective(obj, GRB.MINIMIZE)

    def print_results(self, solving_time):
        print('Total Solving Time light', solving_time)
        print("obj value light", self.model.getAttr("ObjVal"))
        print('bnf_light', self.model.getAttr("X", self.a))
        print(f'beta_zero light', self.model.getAttr("x", self.beta_zero))
        print(f'beta', self.model.getAttr("x", self.beta))

    def get_accuracy(self, data):

        return get_model_accuracy(self,
                                  data,
                                  self.model.getAttr("X", self.a),
                                  self.model.getAttr("x", self.beta_zero),
                                  self.model.getAttr("x", self.beta))
