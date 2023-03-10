import pandas as pd
import time
from encoding import class_encoding_jobs, class_encoding_jobs_alternative
import numpy as np
from sklearn.svm import LinearSVC
import numpy as np


def linear_svc(dataframe, depth):
    clusters = class_encoding_jobs(dataframe, depth)

    nodes_a_b = [([], 0) for i in range(1, np.power(2, depth))]

    def recursive_linear_svc_bottom_up(level=0, node=1):
        # nodes_a_b = [([], 0) for i in range(1, np.power(2, depth))]
        if level == depth:
            df = clusters[node - 2 ** depth]
            # df = df.drop('target',axis=1)

            return df
        cluster_sx = recursive_linear_svc_bottom_up(level + 1, node * 2)
        cluster_dx = recursive_linear_svc_bottom_up(level + 1, node * 2 + 1)
        df = pd.concat([cluster_sx, cluster_dx], ignore_index=True)
        lsvc = LinearSVC(random_state=0, tol=1e-5)
        lsvc.fit(df.drop(['class'], axis=1), np.ravel(df[['class']]))
        nodes_a_b[node - 1] = (lsvc.coef_[0], lsvc.intercept_[0])
        return df

    recursive_linear_svc_bottom_up()
    return nodes_a_b, clusters


def linear_svc_alternative(dataframe, depth):
    clusters = class_encoding_jobs_alternative(dataframe, depth)

    nodes_a_b = {}

    def recursive_linear_svc_bottom_up(level=0, node=1):
        # nodes_a_b = [([], 0) for i in range(1, np.power(2, depth))]
        if level == depth:
            df = clusters[node]['cluster']
            # df = df.drop('target',axis=1)
        else:
            cluster_sx = recursive_linear_svc_bottom_up(level + 1, node * 2)
            cluster_dx = recursive_linear_svc_bottom_up(level + 1, node * 2 + 1)
            df = pd.concat([cluster_sx, cluster_dx])
            lsvc = LinearSVC(random_state=0, tol=1e-5, C=100)
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
            lsvc.fit(X, y)
            score = lsvc.score(X, y)
            '''while score != 1:
                for _i in df.index:
                    predicted_class = lsvc.predict([df.loc[_i, [c for c in df.columns if c != df.columns[-1]]]])[0]
                    df.loc[_i]['class'] = predicted_class
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
                lsvc = LinearSVC(random_state=0, tol=1e-5, C=100)
                lsvc.fit(X, y)
                score = lsvc.score(X, y)'''
            a = [lsvc.coef_[0], lsvc.intercept_[0], score, X, y, df, lsvc]
            nodes_a_b[node] = a
        df['class'] = node
        return df

    recursive_linear_svc_bottom_up()
    return nodes_a_b, clusters


if __name__ == '__main__':
    _df = pd.read_csv('./DataSets/yacht_hydrodynamics_reg.csv')  # [f1,f2,f3,target]
    _dp = 3
    # init_sol, clusters = linear_svc(dataframe.copy(), depth)
    init_sol_alternative, clusters_alternative = linear_svc_alternative(_df.copy(), _dp)
    print('\n')
