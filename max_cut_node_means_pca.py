import pandas as pd
from encoding import class_encoding_jobs, class_encoding_jobs_alternative
from node_means_pca import node_means_pca
from max_cut import max_cut


def max_cut_node_means_pca_bottom_up(dataframe, depth):
    clusters = class_encoding_jobs_alternative(dataframe, depth)

    nodes_a_b = {}

    def iterative_process(level=0, node=1):
        # nodes_a_b = [([], 0) for i in range(1, np.power(2, depth))]
        if level == depth:
            df = clusters[node]['cluster']
            df = df.drop('target', axis=1)
        else:
            cluster_sx = iterative_process(level + 1, node * 2)
            cluster_dx = iterative_process(level + 1, node * 2 + 1)
            df = pd.concat([cluster_sx, cluster_dx])
            n_m_pca, pca_component, pca_score = node_means_pca(df, 'class')
            n_m_pca[1] = df['class']
            th, th_index = max_cut(n_m_pca)
            nodes_a_b[node] = [pca_component, th, pca_score]
        df['class'] = node

        return df

    iterative_process()
    return nodes_a_b, clusters


if __name__ == "__main__":
    dataframe = pd.read_csv('./DataSets/airfoil_self_noise_reg.csv')
    init_sol, clusters = max_cut_node_means_pca_bottom_up(dataframe, 2)
    print(init_sol)
