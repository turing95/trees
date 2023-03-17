from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def get_means(df, label):
    rows = []
    for cl in df[label].unique():
        new_row = {}
        for col in df.columns:
            if col != label:
                new_row[col] = df.loc[df[label] != cl, col].mean()
                rows.append(new_row)
    return rows


def node_means_pca(df: pd.DataFrame, label, logging=False):
    rows = get_means(df, label)

    pca = PCA(n_components=1)
    pca.fit(pd.DataFrame(rows))
    mat_2 = pca.components_[0]
    #df = df.drop(label, axis=1)
    mat_1 = df.iloc[:,:-1].to_numpy()

    new_numpy_df = np.matmul(mat_1, mat_2)
    new_df = pd.DataFrame(new_numpy_df, columns=[0],index=df.index)
    if logging is True:
        print(f'pca input is {rows}\n')
        print(f'matrix 1 (df - label) is {mat_1} with shape {mat_1.shape}\n')
        print(f'matrix 2 (pca component) is {mat_2} with shape {mat_2.shape}\n')
        print(f'projected dataframe is {new_df}')
    return new_df, mat_2
