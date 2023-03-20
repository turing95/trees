from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def get_means(df, label):
    n_means = []
    for cl in df[label].unique():
        n_mean = []
        for col in df.columns:
            if col != label:
                n_mean.append(df.loc[df[label] != cl, col].mean())
        n_means.append(n_mean)
    return n_means


def node_means_pca(df: pd.DataFrame, label):
    n_means = get_means(df, label)
    pca = PCA(n_components=1)
    pca.fit(n_means)
    mat_2 = pca.components_[0]
    X = df.iloc[:, :-1]
    # df = df.drop(label, axis=1)
    # mat_1 = X.to_numpy()

    new_numpy_df = pca.transform(X)
    # new_numpy_df = np.matmul(mat_1, mat_2)
    new_df = pd.DataFrame(new_numpy_df, columns=[0], index=df.index)
    return new_df, mat_2, pca.score(X)
