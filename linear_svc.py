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
            #df = df.drop('target',axis=1)

            return df
        cluster_sx = recursive_linear_svc_bottom_up(level + 1, node * 2)
        cluster_dx = recursive_linear_svc_bottom_up(level + 1, node * 2 + 1)
        df = pd.concat([cluster_sx, cluster_dx], ignore_index=True)
        lsvc = LinearSVC(random_state=0, tol=1e-5)
        lsvc.fit(df.drop(['class'], axis=1),np.ravel(df[['class']]))
        nodes_a_b[node - 1] = (lsvc.coef_[0], lsvc.intercept_[0])
        return df

    recursive_linear_svc_bottom_up()
    return nodes_a_b, clusters


def linear_svc_alternative(dataframe, depth):
    clusters = class_encoding_jobs_alternative(dataframe, depth)

    nodes_a_b = [([], 0) for i in range(1, np.power(2, depth))]

    def recursive_linear_svc_bottom_up(level=0, node=1):
        # nodes_a_b = [([], 0) for i in range(1, np.power(2, depth))]
        if level == depth:
            df = clusters[node]
            df['class'] = node
            #df = df.drop('target',axis=1)
        else:
            cluster_sx = recursive_linear_svc_bottom_up(level + 1, node * 2)
            cluster_dx = recursive_linear_svc_bottom_up(level + 1, node * 2 + 1)
            df = pd.concat([cluster_sx, cluster_dx], ignore_index=True)
            lsvc = LinearSVC(random_state=0, tol=1e-5)
            y = df[['class']]
            X = df.drop(['class'], axis=1)
            lsvc.fit(X, np.ravel(y))
            nodes_a_b[node - 1] = (lsvc.coef_[0], lsvc.intercept_[0])
        return df

    recursive_linear_svc_bottom_up()
    return nodes_a_b, clusters


if __name__ == '__main__':
    dataframe = pd.read_csv('./DataSets/airfoil_self_noise_reg.csv')  # [f1,f2,f3,target]
    depth = 1
    #init_sol, clusters = linear_svc(dataframe.copy(), depth)
    init_sol_alternative, clusters_alternative = linear_svc_alternative(dataframe.copy(), depth)
    print('\n')
