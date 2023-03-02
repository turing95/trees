import pandas as pd
import time
from encoding import class_encoding_jobs
from node_means_pca import node_means_pca
from max_cut import max_cut
import numpy as np

dataframe = pd.read_csv('./DataSets/airfoil_self_noise_reg.csv')  # [f1,f2,f3,target]
depth = 3

start = time.time()
clusters = class_encoding_jobs(dataframe, depth)

class_dataframe = pd.concat([df for df in clusters], ignore_index=True)
class_dataframe = class_dataframe.drop('target', axis=1)
nodes_a_b = [([], 0) for i in range(1, np.power(2, depth))]


def max_cut_node_means_pca(df,d, node=1, level=0):
    if level == d:
        return
    df = df.reset_index(drop=True)
    n_m_pca = node_means_pca(df, 'class', logging=True)
    n_m_pca[1] = df['class']
    th, th_index = max_cut(n_m_pca)
    nodes_a_b[node - 1] = (n_m_pca, th)
    max_cut_node_means_pca(df.iloc[:th_index+1],d, node * 2, level=level + 1)
    max_cut_node_means_pca(df.iloc[th_index+1:],d, node * 2 + 1, level=level + 1)


max_cut_node_means_pca(class_dataframe,depth)
print(nodes_a_b)
