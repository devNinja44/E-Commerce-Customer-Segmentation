"""
Clustering module for customer segmentation.

This module contains functions for applying various clustering
techniques to customer data.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
import matplotlib.pyplot as plt


def find_optimal_k(X, k_range=range(2, 11), random_state=42):
    """
    Find the optimal number of clusters for KMeans using the elbow method and silhouette score.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    k_range : range, default=range(2, 11)
        Range of k values to try.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (optimal_k_elbow, optimal_k_silhouette, inertia_values, silhouette_values)
    """
    inertia_values = []
    silhouette_values = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score requires at least 2 clusters
            labels = kmeans.labels_
            silhouette_values.append(silhouette_score(X, labels))
        else:
            silhouette_values.append(0)
    
    # Find optimal k using elbow method
    try:
        kl = KneeLocator(
            list(k_range),
            inertia_values,
            curve='convex',
            direction='decreasing'
        )
        optimal_k_elbow = kl.elbow
    except:
        # If KneeLocator fails, use a simple heuristic
        diffs = np.diff(inertia_values)
        optimal_k_elbow = k_range[np.argmax(diffs) + 1]
    
    # Find optimal k using silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_values)]
    
    return optimal_k_elbow, optimal_k_silhouette, inertia_values, silhouette_values


def apply_kmeans(X, n_clusters=3, random_state=42):
    """
    Apply KMeans clustering.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (labels, model, inertia, silhouette, davies_bouldin)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calculate evaluation metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(X, labels) if n_clusters > 1 else float('inf')
    
    return labels, kmeans, inertia, silhouette, davies_bouldin


def find_optimal_dbscan_params(X, eps_range=None, min_samples_range=None):
    """
    Find optimal parameters for DBSCAN.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    eps_range : list, default=None
        Range of eps values to try. If None, a default range is used.
    min_samples_range : list, default=None
        Range of min_samples values to try. If None, a default range is used.
        
    Returns
    -------
    tuple
        (optimal_eps, optimal_min_samples, silhouette_values)
    """
    if eps_range is None:
        # Calculate a reasonable range for eps based on data
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, 1])
        
        # Use percentiles to define a range
        eps_min = np.percentile(distances, 10)
        eps_max = np.percentile(distances, 90)
        eps_range = np.linspace(eps_min, eps_max, 10)
    
    if min_samples_range is None:
        # Rule of thumb: min_samples should be at least 2*dimensions
        dim = X.shape[1]
        min_samples_range = list(range(2*dim, 5*dim + 1, dim))
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Count number of clusters (excluding noise points labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Calculate silhouette score if there are at least 2 clusters
            if n_clusters >= 2:
                # Filter out noise points for silhouette calculation
                mask = labels != -1
                if sum(mask) > n_clusters:  # Ensure we have enough points
                    silhouette = silhouette_score(X[mask], labels[mask])
                    noise_ratio = 1 - sum(mask) / len(labels)
                    
                    results.append((eps, min_samples, silhouette, n_clusters, noise_ratio))
    
    if not results:
        return None, None, []
    
    # Sort by silhouette score (higher is better)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Filter results with reasonable noise ratio (e.g., less than 30%)
    filtered_results = [r for r in results if r[4] < 0.3]
    
    if filtered_results:
        optimal_eps, optimal_min_samples, _, _, _ = filtered_results[0]
    else:
        optimal_eps, optimal_min_samples, _, _, _ = results[0]
    
    return optimal_eps, optimal_min_samples, results


def apply_dbscan(X, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns
    -------
    tuple
        (labels, model, n_clusters, noise_ratio, silhouette, davies_bouldin)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Calculate noise ratio
    noise_ratio = np.sum(labels == -1) / len(labels)
    
    # Calculate evaluation metrics if there are at least 2 clusters
    if n_clusters >= 2:
        # Filter out noise points for silhouette calculation
        mask = labels != -1
        if sum(mask) > n_clusters:  # Ensure we have enough points
            silhouette = silhouette_score(X[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
        else:
            silhouette = 0
            davies_bouldin = float('inf')
    else:
        silhouette = 0
        davies_bouldin = float('inf')
    
    return labels, dbscan, n_clusters, noise_ratio, silhouette, davies_bouldin


def apply_agglomerative(X, n_clusters=3, linkage='ward'):
    """
    Apply Agglomerative Clustering.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_clusters : int, default=3
        Number of clusters.
    linkage : str, default='ward'
        Linkage criterion. Options: 'ward', 'complete', 'average', 'single'.
        
    Returns
    -------
    tuple
        (labels, model, silhouette, davies_bouldin)
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)
    
    # Calculate evaluation metrics
    silhouette = silhouette_score(X, labels) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(X, labels) if n_clusters > 1 else float('inf')
    
    return labels, agg, silhouette, davies_bouldin


def cluster_data(X, method='kmeans', **kwargs):
    """
    Apply clustering using the specified method.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    method : str, default='kmeans'
        Clustering method. Options: 'kmeans', 'dbscan', 'agglomerative'.
    **kwargs : dict
        Additional parameters for the specific method.
        
    Returns
    -------
    tuple
        (labels, model, metrics)
        metrics is a dictionary with method-specific evaluation metrics.
    """
    if method == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 3)
        random_state = kwargs.get('random_state', 42)
        
        labels, model, inertia, silhouette, davies_bouldin = apply_kmeans(
            X, n_clusters=n_clusters, random_state=random_state
        )
        
        metrics = {
            'inertia': inertia,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
    
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        
        labels, model, n_clusters, noise_ratio, silhouette, davies_bouldin = apply_dbscan(
            X, eps=eps, min_samples=min_samples
        )
        
        metrics = {
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
    
    elif method == 'agglomerative':
        n_clusters = kwargs.get('n_clusters', 3)
        linkage = kwargs.get('linkage', 'ward')
        
        labels, model, silhouette, davies_bouldin = apply_agglomerative(
            X, n_clusters=n_clusters, linkage=linkage
        )
        
        metrics = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return labels, model, metrics


def compare_clustering_methods(X, methods=None, **kwargs):
    """
    Compare different clustering methods.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    methods : list, default=None
        List of methods to compare. If None, all methods are used.
        Options: 'kmeans', 'dbscan', 'agglomerative'.
    **kwargs : dict
        Additional parameters for the specific methods.
        
    Returns
    -------
    dict
        Dictionary with method names as keys and (labels, model, metrics) as values.
    """
    if methods is None:
        methods = ['kmeans', 'dbscan', 'agglomerative']
    
    results = {}
    
    for method in methods:
        print(f"Applying {method}...")
        method_kwargs = kwargs.get(method, {})
        results[method] = cluster_data(X, method=method, **method_kwargs)
    
    return results
