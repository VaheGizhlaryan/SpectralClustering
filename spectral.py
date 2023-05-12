import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class SpectralCluster:

    def __init__(self, k, kernel='gaussian'):
        self.kernel = kernel
        self.transformed = None
        self.centroids = None
        self.k = k
        self.kmeans = KMeans(n_clusters=self.k, random_state=0)
        self.labels = None

    def fit(self, X):
        if self.kernel == 'gaussian':
            # Calculate graph Laplacian matrix
            W = gaussian_kernel(X)
            D = np.diag(np.sum(W, axis=1))
            L = D - W

            # Calculate eigenvectors and eigenvalues of Laplacian matrix using eigsh
            eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=self.k, which="SM")

            # normalize the eigenvectors
            self.transformed = eigenvectors / np.linalg.norm(eigenvectors, axis=1)[:, np.newaxis]

            # perform k-means clustering on the normalized eigenvectors
            self.kmeans.fit(self.transformed)

        elif self.kernel == 'knn':
            # Calculate graph Laplacian matrix
            W = construct_knn_graph(X)
            D = sp.diags(np.asarray(W.sum(axis=1)).flatten(), 0)
            L = D - W

            # compute the first n eigenvectors of the Laplacian matrix using eigsh
            eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=self.k, which="SM")

            # normalize the eigenvectors
            self.transformed = eigenvectors / np.linalg.norm(eigenvectors, axis=1)[:, np.newaxis]

            # perform k-means clustering on the normalized eigenvectors
            self.kmeans.fit(self.transformed)

    def predict(self, X):
        self.labels = self.kmeans.predict(self.transformed)
        self.centroids = self.kmeans.cluster_centers_
        return self.labels


def gaussian_kernel(X, sigma=0.1):
    """Sigma=0.1 gives the best performance"""
    """Compute the Gaussian kernel matrix for a given dataset X and bandwidth sigma."""
    pairwise_dists = pairwise_distances(X, metric='euclidean')
    K = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    return K


# X is the data matrix, k is the number of nearest neighbors
def construct_knn_graph(X, k=5):
    # construct the k-NN graph
    knn_graph = kneighbors_graph(X, k, mode='connectivity')
    # make the k-NN graph symmetric
    knn_graph = knn_graph.maximum(knn_graph.T)
    return knn_graph


#  For testing

X_circles, y_circles = make_circles(n_samples=1000, noise=0.02, factor=0.6, random_state=0)

spectral = SpectralCluster(k=2, kernel='gaussian')
spectral.fit(X_circles)
labels = spectral.predict(X_circles)

plt.scatter(X_circles[:, 0], X_circles[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
