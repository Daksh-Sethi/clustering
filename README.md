# Clustering Algorithms using PyCaret

## Overview

This repository contains implementations and examples of various clustering algorithms using the PyCaret library. Below is a brief overview of each algorithm, along with code snippets on how to use them:

### KMeans Clustering

A simple and popular clustering algorithm that partitions the data into K distinct clusters.

```python
from pycaret.clustering import setup, create_model

clustering_setup = setup(data, normalize=True)
kmeans_model = create_model('kmeans', num_clusters=3)
```

### Hierarchical Clustering (Agglomerative)
A hierarchical clustering algorithm that builds a tree of clusters by recursively merging the most similar clusters.

```python
from pycaret.clustering import setup, create_model

clustering_setup = setup(data, normalize=True)
agglomerative_model = create_model('agglomerative', num_clusters=3)
```

### DBSCAN Clustering
A density-based clustering algorithm that groups together points that are close to each other and leaves out points that are in low-density regions.

```python
from pycaret.clustering import setup, create_model

clustering_setup = setup(data, normalize=True)
dbscan_model = create_model('dbscan', eps=0.5, min_samples=5)
```

### KModes Clustering
A clustering algorithm that can handle categorical data and uses modes (most frequent values) instead of means or medians.

```python
from pycaret.clustering import setup, create_model

clustering_setup = setup(data, normalize=True)
kmodes_model = create_model('kmodes', num_clusters=3)
```

### Plots and Evaluation Metrics
#### Here are some common plots and evaluation measures for clustering algorithms:

Plots:

Elbow Plot: Shows the number of clusters (K) on the x-axis and the within-cluster sum of squares (WCSS) on the y-axis. Helps to determine the optimal number of clusters.


Silhouette Plot: Shows the silhouette coefficient for each data point, which measures how similar a point is to its own cluster compared to other clusters. Values range from -1 to 1, where a higher value indicates better clustering.


Evaluation Measures:

Silhouette Score: A single value that represents the average silhouette coefficient for all data points. Ranges from -1 to 1, where a higher value indicates better clustering.

Davies-Bouldin Score: Compares the average distance between clusters to the average distance within clusters. A lower value indicates better clustering.

These plots and metrics can help you evaluate the performance of your clustering algorithms and choose the optimal number of clusters for your dataset.
