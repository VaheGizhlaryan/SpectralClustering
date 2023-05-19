from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt


def calculate_dissimilarity(data):
    # Calculate pairwise Euclidean distances
    pairwise_distances = pdist(data, metric='euclidean')

    # Convert the pairwise distances to a square dissimilarity matrix
    dissimilarity_matrix = squareform(pairwise_distances)

    return dissimilarity_matrix


def mds_gradient_descent(data, n_dimensions, learning_rate=0.01, max_iterations=1000):
    # Step 1: Compute the squared distance matrix
    D = calculate_dissimilarity(data)
    D_squared = D ** 2

    # Step 2: Initialize the configuration randomly
    n_samples = D.shape[0]
    X = np.random.rand(n_samples, n_dimensions)

    # Step 3: Perform gradient descent
    for iteration in range(max_iterations):
        # Compute pairwise squared Euclidean distances in the current configuration
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        pairwise_distances = X_squared + X_squared.T - 2 * X.dot(X.T)

        # Step 4: Compute the gradient
        gradient = np.zeros_like(X)
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    gradient[i] += (pairwise_distances[i, j] - D_squared[i, j]) * (X[i] - X[j])

        # Step 5: Update the configuration
        X -= learning_rate * gradient

    return X


# Generate synthetic data
np.random.seed(42)
n_samples = 100
n_features = 3
data = np.random.rand(n_samples, n_features)

# Apply MDS with gradient descent
n_dimensions = 2
learning_rate = 0.01
max_iterations = 1000
embedding = mds_gradient_descent(data, n_dimensions, learning_rate, max_iterations)

# Calculate distances from the origin
distances = np.linalg.norm(data, axis=1)

# Plot the original and embedded data with color
fig = plt.figure(figsize=(12, 6))

# Original data
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c=distances, cmap='viridis')
ax1.set_title('Original Data')

# Embedded data
ax2 = fig.add_subplot(1, 2, 2)
scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], c=distances, cmap='viridis')
ax2.set_title('Embedded Data')

# Add color-bar
cbar = plt.colorbar(scatter)
cbar.set_label('Distance from Origin')

plt.tight_layout()
plt.show()
