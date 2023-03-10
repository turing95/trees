import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def synthetic_dataset():
    # Set the number of iterations to run the loop
    n_iter = 10

    # Set the number of rows and columns for the synthetic dataset
    n_rows = 10000
    n_cols = 14

    # Initialize variables to store the best silhouette score and corresponding dataset
    best_score = 0
    best_data = None

    # Loop through the number of iterations
    for i in range(n_iter):

        # Generate a synthetic dataset with random values between 0 and 1
        data = np.random.rand(n_rows, n_cols)

        # Set the range of clusters to try

        # Fit the KMeans model to the synthetic dataset
        kmeans = KMeans(n_clusters=20, random_state=0,n_init=10,init='k-means++',max_iter=300, tol=0.0001,)
        kmeans.fit(data)

        # Compute the silhouette score for the KMeans model
        score = silhouette_score(data, kmeans.labels_)

        # If the score is higher than the previous best score, update the best score and dataset
        if score > best_score:
            best_score = score
            best_data = data
        print(f"Iteration {i + 1}: Best silhouette score = {best_score}")

    # Print the best silhouette score and corresponding dataset
    print("Best silhouette score:", best_score)
    print("Best dataset:")
    print(best_data)


if __name__ == '__main__':
    synthetic_dataset()
