# Silhouette Score for Clustering Evaluation

The **Silhouette score** is a widely used metric for evaluating the quality of clusters in a clustering algorithm. It measures how similar a point is to its own cluster compared to other clusters. The higher the Silhouette score, the better defined the clusters are.

The Silhouette score combines both cohesion (how close points in a cluster are to each other) and separation (how well-separated the clusters are). For each data point, the score compares the average distance to other points within the same cluster (cohesion) to the average distance to points in the nearest cluster (separation). The Silhouette score is calculated for all points in the dataset, and the overall score is the average of individual point scores.

The **Silhouette score** for an individual point \( i \) is defined as:
$$\Large
\begin{align*}
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\end{align*}
$$

- $ a(i)$ is the **average distance** between point \( i \) and all other points in the same cluster. This measures cohesion.
- $ b(i) $ is the **minimum average distance** between point \( i \) and all points in any other cluster. This measures separation.
- $ \max(a(i), b(i)) $ is used to normalize the score, ensuring that it falls between -1 and 1.

- **Cohesion** $a(i)$ measures how well the point \( i \) fits within its cluster. A lower value of $a(i)$ means that the point is close to other points in the cluster.
- **Separation** $b(i)$ measures how well-separated point $i$ is from the other clusters. A higher value of $b(i)$ means that the point is far from the nearest neighboring cluster.
\\
- The **Silhouette score** $s(i)$ can take values between -1 and 1:
  - A score close to **1** indicates that the point is well clustered (both close to its own cluster and far from others).
  - A score close to **0** indicates that the point lies on or near the boundary between two clusters.
  - A score close to **-1** indicates that the point may have been assigned to the wrong cluster.

The **Silhouette score for the entire dataset** is the average of the Silhouette scores of all individual points:

- **Assumes Convex Clusters**: The Silhouette score assumes that clusters are convex and isotropic, which may not always be the case, especially for non-globular clusters.
- **Sensitive to Outliers**: The score can be sensitive to outliers or noise, as these points can distort the cohesion and separation calculations.

# Davies-Bouldin Score
The **Davies-Bouldin Score** evaluates clustering quality by measuring the average similarity between clusters. Lower scores indicate better clustering.

$$\Large
\begin{align*}
DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d_{ij}} \right)
\end{align*}
$$

* $k$: The number of clusters
* $\sigma_i$: Scatter within cluster
* $i$, defined as the average distance of all points in cluster $i$ to the cluster centroid $c_i$.
* $d_{ij}$: Distance between centroids of clusters $i$ and $j$, typically computed as: $d_{ij} = \text{dist}(c_i, c_j)$

* $\sigma_i = \frac{1}{|C_i|} \sum_{x \in C_i} \text{dist}(x, c_i)$.

**Interpretation**:
- **Lower $DB$**: Clusters are compact (low $\sigma$) and well-separated (high $d_{ij}$).
- **Higher $DB$**: Indicates poor clustering (overlapping or dispersed clusters).


