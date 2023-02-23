import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import time


# linear svc
# make blobs, an example of k means ++ initialization

def func(df):
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(df)
    df['class'] = km.labels_


def class_encoding(df, depth, level=0,node=0):
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(df)
    cluster_labels = km.predict(df)
    df_0 = df[cluster_labels == 0]
    df_1 = df[cluster_labels == 1]
    if level == 2:
        print(df_0.shape)
        print(df_1.shape)
        exit()


    if level == depth:
        return [df_0, df_1]
    else:
        try:
            return class_encoding(df_0, depth, level + 1,node+1) + class_encoding(df_1, level + 1,node+2)
        except Exception as e:
            print('level',level)
            print('rows 0',df_0.shape[0])
            print('rows 1',df_1.shape[0])
            raise e


def node_means_pca(df: pd.DataFrame, logging=False):
    means = []
    # TODO looppare su classi e non su punti, usare dataset con classi. Alla fine avrai un dataframe con una riga per ogni classe, rifare codice qui sotto
    # usare una sola componente per pca
    # fare dataframe * autovettore
    rows = []
    for cl in df['target'].unique():
        new_row = {}
        for col in df.columns:
            if col != 'target':
                new_row[col] = df.loc[df['target'] != cl, col].mean()
                rows.append(new_row)

    pca = PCA(n_components=1)
    pca.fit(pd.DataFrame(rows))
    mat_2 = pca.components_[0]
    df = df.drop('target', axis=1)
    mat_1 = df.to_numpy()

    new_numpy_df = np.matmul(mat_1, mat_2)
    new_df = pd.DataFrame(new_numpy_df, columns=[0])
    if logging is True:
        print(f'pca input is {rows}\n')
        print(f'matrix 1 (df - label) is {mat_1} with shape {mat_1.shape}\n')
        print(f'matrix 2 (pca component) is {mat_2} with shape {mat_2.shape}\n')
        print(f'projected dataframe is {new_df}')
    return new_df


def max_cut(d: pd.DataFrame, logging=False):
    new_d = d.sort_values(by=0)
    temp_gb = d.groupby(1, sort=False, observed=True, dropna=False)[0]
    Nc = temp_gb.size()
    Sc = temp_gb.sum()
    first_class = new_d.iloc[0, 1]
    t_s_Nc = sum(Nc.values)
    t_s_Sc = sum(Sc.values)
    f_Nc = {k: t_s_Nc - Nc[k] for k in Nc.index}
    f_Sc = {k: t_s_Sc - Sc[k] for k in Nc.index}
    theta0 = f_Sc[first_class] - new_d.iloc[0, 0] * f_Nc[first_class]
    threshold = new_d.iloc[0, 0]
    if logging is True:
        print(f'new_d {new_d}\n')
        print(f'temp_gb {temp_gb}\n')
        print(f't_s_Nc {t_s_Nc}\n')
        print(f't_s_Sc {t_s_Sc}\n')
        print(f'f_Nc {f_Nc}\n')
        print(f'f_Sc {f_Sc}\n')
        print(f'theta0 {theta0}\n')
        print(f'threshold {threshold}\n')
    for i in range(t_s_Nc - 1):
        theta1 = theta0 + f_Sc[first_class] - new_d.iloc[i + 1, 0] * f_Nc[first_class]
        if theta1 > theta0:
            threshold = new_d.iloc[i + 1, 0]
            theta0 = theta1
        elif theta1 == theta0:
            threshold = (new_d.iloc[i + 1, 0] + theta0) / 2
    return threshold


def max_cut_v_2(df, feature, label):
    df = df.astype(float)

    pred_teta_i = 0
    min_teta_i, min_idx = None, None
    for idx, row in df.iterrows():
        sum_a = df.loc[df[label] != row[label], feature].sum()
        sum_b = row[feature] * df[df[label] == row[label]].size
        teta_i = pred_teta_i + sum_a - sum_b
        pred_teta_i = teta_i
        if min_teta_i is None or teta_i < min_teta_i:
            min_teta_i = teta_i
            min_idx = idx
    return min_idx


dataframe = pd.read_csv('./DataSets/car_evaluation_enc_reg.csv')
clusters = class_encoding(dataframe,2)

start = time.time()
new_df = node_means_pca(dataframe)
new_df[1] = dataframe['target']
print(f'time pca {time.time() - start}')
start = time.time()
th = max_cut(new_df)
print(f'time max_cut {time.time() - start}')
print(f'threshold {th}')

# th = max_cut(new_df,logging=True)
