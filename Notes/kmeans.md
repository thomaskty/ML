# KMeans Clustering

**Formal Definition**: K-means clustering is an unsupervised learning algorithm that partitions a dataset $X={x_1,x_2,...,x_n}$ into $K$ disjoint clusters $C={c_1,c_2,...,c_k}$, where each clusters is represented by its centroid (mean) $\mu_{k}$, with the objective of minimizing the within-cluster sum of squares.

-------------

In KMeans we parition data into K groups such that ;
* Each data point belongs to exactly one cluster ( hard clustering)
* Clusters are compact ( within cluster sum of squares should be as less as possible)
* Each cluster is represented by a centroid( mean of the cluster points)

**Group data into K clusters so that points are close to their own cluster center and far from others**

## Algorithm 

* Inputs : Dataset with $n$ data points and the number of Clusters : $K$
* Initialize centroids ( randomly or K-Means ++ )
* Assisgnment Step : For each data point, compute its distance to each centroid and assign the point to the nearest centroid.
* For each cluster - recalculate the centroid as the mean of all points assigned to that cluster.
* Repeat this process, until the centroids do not change significantly or assignments do not change or maximum number of iterations is reached.
* Final Outputs : The data points and the associated cluster labels.

$$
\begin{align*}\Large
\underset{C_1,\dots,C_K}{\min}
\sum_{k=1}^{K}
\sum_{x_i \in C_k}
\left\| x_i - \mu_k \right\|^2
\end{align*}
$$

KMeans is **Coordinate Descent**- assignment step is there for optimize the cluster labels, and update step is there for optimizing the centroids

**Concerns**- regarding cluster shape, selection of distance metric, selection of evaluation metric, selection of number of clusters, scaling of data points, centroid initializations, 

**Computational Complexity**


**Question**- Given a set of points assigned to a cluster, where should the centroid be placed so that the total squared distance from the points to the centroid is minimized? The centroid that minimizes the sum of squared euclidean distances is the mean of the points. 


**Why K-Means has local minima( Not guaranteed Global minimum)?**
 * The objective is non-convex
 * Cluster assignments are discrete (combinatorial)
 * The algorithm uses greedy alternating optimization
 * Initialization fixes the "basin of attraction"

Once K-Means makes early assignment decisions, it cannot undo them globally - it only makes locally improving moves.

* The number of possible clustering grows exponentially with data size.
* Even if the objective is convex with respect to centroids, it is not convex with respect to assignments.
* This is why K-Means ++ helps ( but doesn't solve it): improves the initialization, reduces the probability of bad local minima.

**Pairwise distance to centroid based kmeans objective**
 * Pairwise View : All points inside a cluster should be close to each other
 * All points should be close to the cluster center
 * Minimizing the sum of sqaured distances between all pairs of points inside a cluster is equivalent to minimizing the sum of squared distances from points to the cluster centroid.

**Variance Decomposition Implications**

$TSS=WCSS+BCSS$. Since $TSS$ is constant for any given dataset, minimizing $WCSS$ is mathematically equivalent to maximizing $BCSS$. In other words, creating tight clusters automatically means creating well-separated clusters. - complementary

**Computational Complexity**
In the assignment step, we must compute the distance from each of the n points to each of the K centroids. For data in d dimensions, each distance computation requires O(d)a operations. Thus, the assignment step has complexity O(nKd)

In the update step, we must compute the mean of each cluster. This requires summing all points in each cluster and dividing by the cluster size, which takes O(nd) operations in total.

If the algorithm converges in I iterations, the total time complexity is ; $O(I.n.K.d)$


## K-Means ++
**K-means++ Initialization**- The key idea is to choose initial centroids that are far apart from each other. 

* Pick one data point uniformly at random.
* For each data point , compute distance to nearest already chosen centroid. 
$$\Large
\begin{align*}
D(x_i)=\min_{1 \le j \le t}
\|x_i - \mu_j\|
\end{align*}
$$

 * When computing this distance, we are asking the question 'Is this point already well represented by an existing centroid'. if yes then the $D(x_i)$ will be small, otherwise large. Points with large distance are good candidates for new centroids.

* Instead of always picking the farthest point (deterministic) K means ++ uses probabilistic sampling.

$$\Large
\begin{align*}
P(x_i)=\frac{D(x_i)^2}{\sum_{j=1}^{n} D(x_j)^2}
\end{align*}
$$

**Why does kmeans ++ reduce sensitivity to outliers?**- Because it samples points proportitional to squared distance, so dense regions collectively outweigh isolated extream points, unlike deterministic farthest point selection.

## K-Means Through the Expectation-Maximization Lens
Recalling K-Means objective; 
$$
\begin{align*}\Large
\min_{\{z_{ik}, \mu_k\}}
\sum_{i=1}^{n} \sum_{k=1}^{K}
z_{ik} \|x_i - \mu_k\|^2
\end{align*}
$$
Here $z_{ik}$ is 1 if $x_i$ belongs to cluster k and 0 otherwise.

**Expectation Step**: Given current centroids $\mu_{k}$;
$$
\Large
\begin{align*}
z_{ik} =
\begin{cases}
1 & k = \arg\min_j \|x_i - \mu_j\|^2 \\
0 & \text{otherwise}
\end{cases}
\end{align*}
$$
This is hard assignment ; Each point belongs to exactly one cluster( Probabilities are 0 or 1) 

**Maximization Step**: Given assignments $z_{ik}$; 
$$
\Large
\begin{align*}
\mu_{k}=
\frac{\sum_{i=1}^{n} z_{ik} x_i}
{\sum_{i=1}^{n} z_{ik}}
\end{align*}
$$
This is called the mean update 

## Comparison of K-Means & GMM in terms of EM Algorithm 
KMeans and Gmm both aim to cluster data, but they differ in what they assume about the data. 

Kmeans focuses directly on partitioning the data into groups by minimizing distances. It does not try to model how the data was generated; it only cares about assigning each point to the closest cluster center. In this sense, it behaves like a discriminative method. 

GMM assumes that the data was generated by a mixture of several underlying sources, and each data point is produced by one of these sources. The goal is to learn these sources and estimate how likely each data point cam efrom each one. 
Because of this, GMM are generative ; instead of just assigning clusters, they try to model the full data distribution and explain the data-generation process. 

## Discussion about the use of Euclidean distance in Kmeans.
Kmeans is build around a very simple idea ; Represent a group of points by a single repesentative point ; what does central even mean . The answers depends on the distance we use. Under Euclidean distance squared, something magical happens; The point that minimize the total squared distance to all points is the mean. This is not true for most other distances. 

**What breaks if we don't use euclidean distance**
 * Mean is no longer optimal centroid
 * No variance interpretation
 * No EM interpretation

## What happens if we use Manhattan distance (L1) 
Under manhattan distance, the point that minimizes total distance is the median, not the mean. Manhattan distance cares about absolute deviations. The median minimizes absolute error. This is why medians are robust to outliers. So clustering with Manhattan distance naturally leads to k-medians. 


## K-Medoid : When the Center Must be a real point
In many real-world cases, distances are not euclidean, data will be mixed or categorical. You want a real data point as the representative. You want robustness to outliers. 
Medoid is an actual data point whose total distance to other points in the cluster is minimal. 
$$
\Large
\begin{align*}
\min_{\{m_k\}} \sum_{k=1}^{K} \sum_{x_i \in C_k} d(x_i, m_k)
\end{align*}
$$
 * Choose K data points as initial medoids ( random or heuristic)
 * Assign each point to the nearest medoid using the chosen distance
 * For each cluster;
    * Try every point in the cluster as a candidate medoid
    * Choose the one that minimizes total distance to others. ( new medoidd)
 * Repeat until convergence
 * We dont need any geometry for the distance metric ( as long as smaller distance = more similar)
 * Expensive : Updating medoids requires pairwise distances. So in practice, people use; PAM (Partitioning Around Medoids) , CLARA ( sampling-based), CLARANS (randomized)

## K-Modes ( For categorical data) 
* For categorical attribute, distance is total Number of mismatched attributes. 
* The centroid is the mode;- Which is computed for each attribute.

## K-Prototype( Numeric + Categorical) 
* Combines k-means and k-modes in a single objective
* Use Euclidean distance for numeric features
* Use Mismatch distance for categorical features
* Balance then with a weight

$$
\Large
\begin{align*}
d(x, \mu) =
\sum_{j \in \text{num}} (x_j - \mu_j)^2
\;+\;
\gamma \sum_{j \in \text{cat}} \mathbf{1}(x_j \neq \mu_j)
\end{align*}
$$

* The centroid has two parts: Numeric attributes will be updated using the mean. Categorical part will be updated using the mode.
* Think of $\gamma$ as answering : How much numeric difference is equivalent to one categorical difference?

## Assumptions of KMeans 
 * Clusters are compact and well separated
 * Variance within clusters is roughly similar
 * Features are on comparable scales
 * Euclidean geometry is meaninful
 * Clusters are convex

## Scaling and High-Dimensional Behavior
 * Curse of dimensionality effect on Euclidean distance
 * PCA can be applied

## KMeans as Matrix Factorization

## Online / Mini-Batch kmeans
In online kmeans, we take one data point, assign it to the nearest centroid and update that centroid slightly toward that data point. 

In mini-batch the update happens based on a batch of data ( not the entire dataset) - similar to stochastic gradient descent. 

