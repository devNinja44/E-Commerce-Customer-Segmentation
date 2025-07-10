"""
Evaluation module for customer segmentation.

This module contains functions for evaluating clustering results
using various metrics and techniques.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_silhouette_score(X, labels):
    """
    Calculate the silhouette score for clustering results.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    labels : array-like of shape (n_samples,)
        Cluster labels.
        
    Returns
    -------
    float
        Silhouette score.
    """
    # Check if there are at least 2 clusters and each cluster has at least 1 sample
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or -1 in unique_labels:
        # Filter out noise points (label -1) if present
        mask = labels != -1
        if sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            return silhouette_score(X[mask], labels[mask])
        return 0
    
    return silhouette_score(X, labels)


def calculate_davies_bouldin_score(X, labels):
    """
    Calculate the Davies-Bouldin index for clustering results.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    labels : array-like of shape (n_samples,)
        Cluster labels.
        
    Returns
    -------
    float
        Davies-Bouldin index.
    """
    # Check if there are at least 2 clusters and each cluster has at least 1 sample
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or -1 in unique_labels:
        # Filter out noise points (label -1) if present
        mask = labels != -1
        if sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            return davies_bouldin_score(X[mask], labels[mask])
        return float('inf')
    
    return davies_bouldin_score(X, labels)


def calculate_calinski_harabasz_score(X, labels):
    """
    Calculate the Calinski-Harabasz index for clustering results.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    labels : array-like of shape (n_samples,)
        Cluster labels.
        
    Returns
    -------
    float
        Calinski-Harabasz index.
    """
    # Check if there are at least 2 clusters and each cluster has at least 1 sample
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or -1 in unique_labels:
        # Filter out noise points (label -1) if present
        mask = labels != -1
        if sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            return calinski_harabasz_score(X[mask], labels[mask])
        return 0
    
    return calinski_harabasz_score(X, labels)


def evaluate_clustering(X, labels, method_name=None):
    """
    Evaluate clustering results using multiple metrics.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    method_name : str, default=None
        Name of the clustering method.
        
    Returns
    -------
    dict
        Dictionary with evaluation metrics.
    """
    metrics = {}
    
    # Calculate number of clusters (excluding noise points labeled as -1)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    metrics['n_clusters'] = n_clusters
    
    # Calculate noise ratio if applicable
    if -1 in unique_labels:
        noise_ratio = np.sum(labels == -1) / len(labels)
        metrics['noise_ratio'] = noise_ratio
    
    # Calculate evaluation metrics
    metrics['silhouette'] = calculate_silhouette_score(X, labels)
    metrics['davies_bouldin'] = calculate_davies_bouldin_score(X, labels)
    metrics['calinski_harabasz'] = calculate_calinski_harabasz_score(X, labels)
    
    # Print evaluation results
    print(f"\nEvaluation results for {method_name if method_name else 'clustering'}:")
    print(f"Number of clusters: {n_clusters}")
    if 'noise_ratio' in metrics:
        print(f"Noise ratio: {metrics['noise_ratio']:.4f}")
    print(f"Silhouette score: {metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin index: {metrics['davies_bouldin']:.4f}")
    print(f"Calinski-Harabasz index: {metrics['calinski_harabasz']:.4f}")
    
    return metrics


def compare_clustering_results(X, results_dict):
    """
    Compare clustering results from different methods.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    results_dict : dict
        Dictionary with method names as keys and (labels, model, metrics) as values.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with comparison of evaluation metrics.
    """
    comparison = {}
    
    for method_name, (labels, _, _) in results_dict.items():
        metrics = evaluate_clustering(X, labels, method_name)
        comparison[method_name] = metrics
    
    # Convert to DataFrame for easy comparison
    comparison_df = pd.DataFrame(comparison).T
    
    return comparison_df


def analyze_clusters(X, labels, feature_names=None):
    """
    Analyze clusters to understand their characteristics.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    feature_names : list, default=None
        Names of the features. If None, generic names are used.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with cluster profiles.
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrame with data and cluster labels
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = labels
    
    # Calculate cluster profiles (mean of each feature for each cluster)
    cluster_profiles = df.groupby('Cluster').mean()
    
    # Calculate cluster sizes
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    cluster_profiles['Size'] = cluster_sizes
    
    return cluster_profiles


def generate_cluster_labels(cluster_profiles):
    """
    Generate human-readable labels for clusters based on their profiles.
    
    Parameters
    ----------
    cluster_profiles : pandas.DataFrame
        DataFrame with cluster profiles.
        
    Returns
    -------
    dict
        Dictionary mapping cluster IDs to human-readable labels.
    """
    # Drop the Size column for feature analysis
    profiles = cluster_profiles.drop('Size', axis=1, errors='ignore')
    
    # Normalize profiles to identify distinguishing features
    normalized_profiles = (profiles - profiles.mean()) / profiles.std()
    
    cluster_labels = {}
    
    for cluster_id in profiles.index:
        # Skip noise cluster
        if cluster_id == -1:
            cluster_labels[cluster_id] = "Noise"
            continue
        
        # Get top distinguishing features (both high and low values)
        cluster_profile = normalized_profiles.loc[cluster_id]
        top_features_high = cluster_profile.nlargest(2)
        top_features_low = cluster_profile.nsmallest(2)
        
        # Generate label based on distinguishing features
        label_parts = []
        
        # Add high-value features
        for feature, value in top_features_high.items():
            if value > 0.5:  # Only include if significantly above average
                label_parts.append(f"High {feature}")
        
        # Add low-value features
        for feature, value in top_features_low.items():
            if value < -0.5:  # Only include if significantly below average
                label_parts.append(f"Low {feature}")
        
        # Create final label
        if label_parts:
            cluster_labels[cluster_id] = " & ".join(label_parts[:2])  # Limit to 2 features
        else:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
    
    return cluster_labels
