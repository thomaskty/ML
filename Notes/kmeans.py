import numpy as np
from typing import Union, Callable, Tuple, Optional
import matplotlib.pyplot as plt


class BaseKClustering:
    """Base class for K-clustering algorithms (K-Means, K-Modes, K-Medoids, K-Prototypes)"""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids randomly from data points"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute distance matrix between points and centroids - OVERRIDE THIS"""
        raise NotImplementedError
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids based on current labels - OVERRIDE THIS"""
        raise NotImplementedError

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid"""
        distances = self._compute_distance(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def fit(self, X: np.ndarray):
        """Fit the clustering model"""
        self.centroids = self._initialize_centroids(X)
        
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            self.labels_ = self._assign_labels(X)
            self.centroids = self._update_centroids(X, self.labels_)
            
            # Check convergence
            shift = np.sum(np.abs(self.centroids - old_centroids))
            if shift < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        # Calculate inertia
        distances = self._compute_distance(X, self.centroids)
        self.inertia_ = np.sum(np.min(distances, axis=1))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        return self._assign_labels(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_
    

class KMeans(BaseKClustering):
    """K-Means clustering for continuous data"""
    
    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Euclidean distance"""
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Mean of points in each cluster"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                new_centroids[k] = self.centroids[k]
        return new_centroids


class KModes(BaseKClustering):
    """K-Modes clustering for categorical data"""
    
    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Hamming distance (count of mismatches)"""
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sum(X != centroid, axis=1)
        return distances
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Mode (most frequent value) of points in each cluster"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_data = X[mask]
                for j in range(X.shape[1]):
                    values, counts = np.unique(cluster_data[:, j], return_counts=True)
                    new_centroids[k, j] = values[np.argmax(counts)]
            else:
                new_centroids[k] = self.centroids[k]
        return new_centroids


class KMedoids(BaseKClustering):
    """K-Medoids clustering - uses actual data points as centroids"""
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize by selecting random points"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.medoid_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[self.medoid_indices].copy()
    
    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Euclidean distance"""
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Select point with minimum total distance to others in cluster"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        new_medoid_indices = np.zeros(self.n_clusters, dtype=int)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_points = X[mask]
                cluster_indices = np.where(mask)[0]
                
                # Compute pairwise distances within cluster
                distances = np.zeros(len(cluster_points))
                for i, point in enumerate(cluster_points):
                    distances[i] = np.sum(np.sqrt(np.sum((cluster_points - point) ** 2, axis=1)))
                
                # Select point with minimum total distance
                medoid_idx = np.argmin(distances)
                new_medoid_indices[k] = cluster_indices[medoid_idx]
                new_centroids[k] = cluster_points[medoid_idx]
            else:
                new_medoid_indices[k] = self.medoid_indices[k]
                new_centroids[k] = self.centroids[k]
        
        self.medoid_indices = new_medoid_indices
        return new_centroids


class KPrototypes(BaseKClustering):
    """K-Prototypes clustering for mixed numerical and categorical data"""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, tol: float = 1e-4, 
                 categorical_indices: list = None, gamma: float = 1.0, random_state: Optional[int] = None):
        super().__init__(n_clusters, max_iter, tol, random_state)
        self.categorical_indices = categorical_indices if categorical_indices else []
        self.numerical_indices = None
        self.gamma = gamma  # Weight for categorical distance
        
    def fit(self, X: np.ndarray):
        """Set up indices and fit"""
        self.numerical_indices = [i for i in range(X.shape[1]) if i not in self.categorical_indices]
        return super().fit(X)
    
    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Combined Euclidean (numerical) + Hamming (categorical) distance"""
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        
        for i, centroid in enumerate(centroids):
            # Numerical distance (Euclidean)
            if self.numerical_indices:
                num_dist = np.sqrt(np.sum((X[:, self.numerical_indices] - centroid[self.numerical_indices]) ** 2, axis=1))
            else:
                num_dist = 0
            
            # Categorical distance (Hamming)
            if self.categorical_indices:
                cat_dist = np.sum(X[:, self.categorical_indices] != centroid[self.categorical_indices], axis=1)
            else:
                cat_dist = 0
            
            distances[:, i] = num_dist + self.gamma * cat_dist
        
        return distances
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Mean for numerical, mode for categorical"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_data = X[mask]
                
                # Numerical: mean
                if self.numerical_indices:
                    new_centroids[k, self.numerical_indices] = np.mean(cluster_data[:, self.numerical_indices], axis=0)
                
                # Categorical: mode
                if self.categorical_indices:
                    for j in self.categorical_indices:
                        values, counts = np.unique(cluster_data[:, j], return_counts=True)
                        new_centroids[k, j] = values[np.argmax(counts)]
            else:
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    

# Demo and testing
if __name__ == "__main__":
    print("=== K-Means Demo ===")
    np.random.seed(42)
    X_numeric = np.vstack([
        np.random.randn(50, 2) + [2, 2],
        np.random.randn(50, 2) + [-2, -2],
        np.random.randn(50, 2) + [2, -2]
    ])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_numeric)
    print(f"Converged in {kmeans.n_iter_} iterations")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(X_numeric[:, 0], X_numeric[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, edgecolors='black')
    plt.title('K-Means')
    
    print("\n=== K-Modes Demo ===")
    X_categorical = np.random.choice([0, 1, 2], size=(150, 3))
    X_categorical[:50, 0] = 0
    X_categorical[50:100, 0] = 1
    X_categorical[100:, 0] = 2
    
    kmodes = KModes(n_clusters=3, random_state=42)
    labels_modes = kmodes.fit_predict(X_categorical)
    print(f"Converged in {kmodes.n_iter_} iterations")
    print(f"Centroids:\n{kmodes.centroids}")
    
    print("\n=== K-Medoids Demo ===")
    kmedoids = KMedoids(n_clusters=3, random_state=42)
    labels_medoids = kmedoids.fit_predict(X_numeric)
    print(f"Converged in {kmedoids.n_iter_} iterations")
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_numeric[:, 0], X_numeric[:, 1], c=labels_medoids, cmap='viridis', alpha=0.6)
    plt.scatter(kmedoids.centroids[:, 0], kmedoids.centroids[:, 1], c='red', marker='X', s=200, edgecolors='black')
    plt.title('K-Medoids')
    
    print("\n=== K-Prototypes Demo ===")
    X_mixed = np.hstack([X_numeric[:150], X_categorical])
    kprototypes = KPrototypes(n_clusters=3, categorical_indices=[2, 3, 4], gamma=0.5, random_state=42)
    labels_proto = kprototypes.fit_predict(X_mixed)
    print(f"Converged in {kprototypes.n_iter_} iterations")
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_mixed[:, 0], X_mixed[:, 1], c=labels_proto, cmap='viridis', alpha=0.6)
    plt.scatter(kprototypes.centroids[:, 0], kprototypes.centroids[:, 1], c='red', marker='X', s=200, edgecolors='black')
    plt.title('K-Prototypes')
    
    plt.tight_layout()
    print("\nVisualization saved as 'clustering_comparison.png'")
    plt.show()


    