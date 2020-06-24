#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:11:47 2020

@author: dawei
"""

"""
kmeans clusterinf selection
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import numpy as np

#%%
'''elbow method & silhouette score'''
def elbow_and_silhouette(embeddings):
    range_n_clusters = np.arange(2,10)
    distortions = []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        clusterer.fit(embeddings)
        cluster_labels = clusterer.predict(embeddings)
        #print(cluster_labels)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(embeddings, cluster_labels)
        # save elbow score
        distortions.append(clusterer.inertia_)
    # plot elbow scores
    plt.figure(figsize=(16,8))
    plt.plot(range_n_clusters, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    
    plt.show()

elbow_and_silhouette(embeddings)

clusterer = KMeans(n_clusters=2, random_state=10)
clusterer.fit(embeddings)
cluster_labels = clusterer.predict(embeddings)

#%%
'''gap statistics for kmeans'''
def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)

def compute_gap(clustering, data, k_max=5, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(data)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))
    
    ondata_inertia = []
    for k in range(1, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)

k_max = 10
gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), embeddings, k_max)

# plot reference pair-wise distance
plt.plot(range(1, k_max+1), reference_inertia,
         '-o', label='reference')
plt.plot(range(1, k_max+1), ondata_inertia,
         '-o', label='data')
plt.xlabel('k')
plt.ylabel('log(inertia)')
plt.show()
# plot gap distance
plt.plot(range(1, k_max+1), gap, '-o')
plt.ylabel('gap')
plt.xlabel('k')