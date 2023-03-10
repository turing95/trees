from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score,silhouette_score
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

num_cores = multiprocessing.cpu_count()


def class_encoding_parallel(df, depth, level=0, node=1):
    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    km.fit(df)
    best_k = km
    best_score = davies_bouldin_score(df, km.labels_)
    for i in range(1, 100):
        km = KMeans(n_clusters=2, random_state=i, n_init=10)
        km.fit(df)
        score = davies_bouldin_score(df, km.labels_)
        if score > best_score:
            best_score = score
            best_k = km
    cluster_labels = best_k.predict(df)
    df_0 = df[cluster_labels == 0]
    df_1 = df[cluster_labels == 1]
    if level == depth:
        return [df_0, df_1]
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_0 = executor.submit(class_encoding_parallel, df_0, depth, level=level + 1, node=node * 2)
            future_1 = executor.submit(class_encoding_parallel, df_1, depth, level=level + 1, node=node * 2 + 1)
            return future_0.result() + future_1.result()


def class_encoding_jobs(df, depth, level=0, node=1):
    def find_best_k(d, random_state):
        km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        km.fit(df)
        score = davies_bouldin_score(d, km.labels_)
        return km, score

    # set up parallelization
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(find_best_k)(df, i) for i in range(1, 100)
    )

    # find best KMeans model
    best_k, best_score = results[0]
    for km, score in results[1:]:
        if score > best_score:
            best_score = score
            best_k = km
    cluster_labels = best_k.predict(df)

    df_0 = df[cluster_labels == 0]
    df_1 = df[cluster_labels == 1]

    if level == depth - 1:
        class_1 = node * 2 - 2 ** depth + 1
        class_2 = node * 2 - 2 ** depth + 2
        df_0['class'] = class_1
        df_1['class'] = class_2
        return [df_0, df_1]
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(class_encoding_jobs)(
                df_i, depth, level + 1, node * 2 + i)
            for i, df_i in enumerate([df_0, df_1]))
        return [item for sublist in results for item in sublist]


def get_best_k_means(df):
    def find_best_k(d, random_state):
        km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        km.fit(df)
        score = davies_bouldin_score(d, km.labels_)
        return km, score

    # set up parallelization
    results = Parallel(n_jobs=num_cores)(
        delayed(find_best_k)(df, i) for i in range(1, 100)
    )

    # find best KMeans model
    best_k, best_score = results[0]
    for km, score in results[1:]:
        if score < best_score:
            best_score = score
            best_k = km
    cluster_labels = best_k.predict(df)

    df_0 = df[cluster_labels == 0]
    df_1 = df[cluster_labels == 1]
    return df_0, df_1,best_score


def class_encoding_jobs_alternative(df, depth):
    clusters = {}

    def iterative_function(d,score=None,level=0,node=1):
        clusters[node] = {}
        clusters[node]['cluster'] = d
        clusters[node]['score'] = score
        if level == depth:
            return
        df_0, df_1,score = get_best_k_means(d)
        iterative_function(df_0,score,level+1,node*2)
        iterative_function(df_1,score,level+1,node*2+1)

    iterative_function(df)
    return clusters

'''    if level == depth - 1:
        class_1 = node * 2
        class_2 = node * 2 + 1
        df_0['class'] = class_1
        df_1['class'] = class_2
        print(node)
        print(class_1)
        print(class_2)
        return [df_0, df_1]
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(class_encoding_jobs_alternative)(
                df_i, depth, level + 1, node * 2 + i)
            for i, df_i in enumerate([df_0, df_1]))
        return [item for sublist in results for item in sublist]'''


def class_encoding(df, depth, level=0, node=1):
    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    km.fit(df)
    best_k = km
    best_score = davies_bouldin_score(df, km.labels_)
    for i in range(1, 100):
        km = KMeans(n_clusters=2, random_state=i, n_init=10)
        km.fit(df)
        score = davies_bouldin_score(df, km.labels_)
        if score < best_score:
            best_score = score
            best_k = km
    cluster_labels = best_k.predict(df)
    df_0 = df[cluster_labels == 0]
    df_1 = df[cluster_labels == 1]
    if level == depth - 1:
        print(node)
        class_1 = node * 2 - 2 ** depth + 1
        class_2 = node * 2 - 2 ** depth + 2
        print(class_1)
        print(class_2)
        df_0['class'] = class_1
        df_1['class'] = class_2
        return [df_0, df_1]
    else:
        return class_encoding(df_0, depth, level + 1, node * 2) + class_encoding(df_1, depth, level + 1, node * 2 + 1)


if __name__ == "__main__":
    dataframe = pd.read_csv('./DataSets/airfoil_self_noise_reg.csv')
    cl = class_encoding_jobs_alternative(dataframe, 2)
    print('\n')
