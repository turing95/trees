'''
This module formulate the FlowOCT problem in gurobipy.
'''

from gurobipy import *
from utils.utils_oct_no_p import get_model_accuracy


class FlowOCT:
    def __init__(self, data, label, tree, time_limit, mode='regression', _lambda=0):
        '''

        :param data: The training data
        :param label: Name of the column representing the class label
        :param tree: Tree object
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param mode: Regression vs Classification
        '''
        self.mode = mode

        self.data = data
        self.datapoints = data.index
        self.label = label

        self.labels = [1]

        '''
        cat_features is the set of all categorical features. 
        reg_features is the set of all features used for the linear regression prediction model in the leaves.  
        '''
        self.cat_features = self.data.columns[self.data.columns != self.label]
        # self.reg_features = None
        # self.num_of_reg_features = 1

        self.tree = tree
        self._lambda = _lambda

        # parameters
        self.m = {}
        for i in self.datapoints:
            self.m[i] = 1

        for i in self.datapoints:
            y_i = self.data.at[i, self.label]
            self.m[i] = max(y_i, 1 - y_i)

        # Decision Variables
        self.b = 0
        self.beta = 0
        self.zeta = 0
        self.z = 0

        # Gurobi model
        self.model = Model('FlowOCT')
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
        This function create and return a gurobi model formulating the FlowOCT problem
        :return:  gurobi model object with the FlowOCT formulation
        '''
        ############################### define variables
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes, self.cat_features, vtype=GRB.BINARY, name='b')
        '''
        For classification beta[n,k]=1 iff at node n we predict class k
        For the case regression beta[n,1] is the prediction value for node n
        '''
        self.beta = self.model.addVars(self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                       name='beta')
        # zeta[i,n] is the amount of flow through the edge connecting node n to sink node t for datapoint i
        #TODO remove nodes
        self.zeta = self.model.addVars(self.datapoints, self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                       name='zeta')
        # z[i,n] is the incoming flow to node n for datapoint i
        self.z = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                    name='z')

        ############################### define constraints
        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            self.model.addConstrs(
                (self.z[i, n] == self.z[i, n_left] + self.z[i, n_right]) for i in self.datapoints)

        # z[i,l(n)] <= m[i] * sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_left_children(n))] <= self.m[i] * quicksum(
                self.b[n, f] for f in self.cat_features if self.data.at[i, f] == 0)) for n in self.tree.Nodes)

        # z[i,r(n)] <= m[i] * sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_right_children(n))] <= self.m[i] * quicksum(
                self.b[n, f] for f in self.cat_features if self.data.at[i, f] == 1)) for n in self.tree.Nodes)

        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (quicksum(self.b[n, f] for f in self.cat_features) == 1) for n in
            self.tree.Nodes)

        # sum(sum(b[n,f], f), n) <= branching_limit
        # self.model.addConstr(
        #     (quicksum(
        #         quicksum(self.b[n, f] for f in self.cat_features) for n in self.tree.Nodes)) <= self.branching_limit)

        # beta[n,k] = 1
        for n in self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n] <= self.m[i] - self.data.at[i, self.label] + self.beta[n]
                for i in self.datapoints)

            self.model.addConstrs(
                self.zeta[i, n] <= self.m[i] + self.data.at[i, self.label] - self.beta[n]
                for i in self.datapoints)

        self.model.addConstrs(
            (self.beta[n] <= 1) for n in self.tree.Leaves)

        for n in self.tree.Leaves:
            self.model.addConstrs(self.zeta[i, n] == self.z[i, n] for i in self.datapoints)

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add((1 - self._lambda) * (self.z[i, 1] - self.m[i]))

        for n in self.tree.Nodes:
            for f in self.cat_features:
                obj.add(-1 * self._lambda * self.b[n, f])

        self.model.setObjective(obj, GRB.MAXIMIZE)

    def print_results(self, solving_time):
        print('Total Solving Time oct_no_p', solving_time)
        print("obj value oct_no_p", self.model.getAttr("ObjVal"))
        print('bnf_oct_no_p', self.model.getAttr("X", self.b))
        print(f'oct_beta_zero light', self.model.getAttr("x", self.beta))

    def get_accuracy(self, data):

        return get_model_accuracy(self,
                                  data,
                                  self.model.getAttr("X", self.b),
                                  self.model.getAttr("x", self.beta_zero),
                                  None)