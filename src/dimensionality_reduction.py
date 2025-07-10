"""
Dimensionality reduction module for customer segmentation.

This module contains functions for applying various dimensionality reduction
techniques to high-dimensional customer data.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS
import umap
from sklearn.preprocessing import StandardScaler


def apply_pca(X, n_components=2, random_state=42):
    """
    Apply Principal Component Analysis (PCA) for dimensionality reduction.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_components : int, default=2
        Number of components to keep.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (transformed_data, pca_model, explained_variance_ratio)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca, pca.explained_variance_ratio_


def apply_kernel_pca(X, n_components=2, kernel='rbf', gamma=None, random_state=42):
    """
    Apply Kernel PCA for dimensionality reduction.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_components : int, default=2
        Number of components to keep.
    kernel : str, default='rbf'
        Kernel type to be used. Options: 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'.
    gamma : float, default=None
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (transformed_data, kpca_model)
    """
    kpca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state
    )
    X_kpca = kpca.fit_transform(X)
    
    # Return None as the third value to maintain consistent return signature with other methods
    return X_kpca, kpca, None


def apply_mds(X, n_components=2, metric=True, n_jobs=-1, random_state=42):
    """
    Apply Multidimensional Scaling (MDS) for dimensionality reduction.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_components : int, default=2
        Number of components to keep.
    metric : bool, default=True
        If True, perform metric MDS; otherwise, perform nonmetric MDS.
    n_jobs : int, default=-1
        The number of jobs to use. Use -1 to use all processors.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (transformed_data, mds_model, stress)
    """
    mds = MDS(
        n_components=n_components,
        metric=metric,
        n_jobs=n_jobs,
        random_state=random_state
    )
    X_mds = mds.fit_transform(X)
    
    return X_mds, mds, mds.stress_


def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """
    Apply Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_components : int, default=2
        Number of components to keep.
    n_neighbors : int, default=15
        Number of neighbors to consider for each point.
    min_dist : float, default=0.1
        Minimum distance between points in the low-dimensional representation.
    metric : str, default='euclidean'
        Distance metric to use.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (transformed_data, umap_model)
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    # Handle compatibility with different versions of UMAP
    try:
        X_umap = reducer.fit_transform(X)
    except TypeError as e:
        if 'ensure_all_finite' in str(e):
            # For newer versions of UMAP that don't accept ensure_all_finite
            print("Handling UMAP compatibility issue with ensure_all_finite parameter")
            X_umap = reducer.fit_transform(X.copy())  # Use a copy to avoid modifying the original
        else:
            raise e
    
    # Return None as the third value to maintain consistent return signature with other methods
    return X_umap, reducer, None


def reduce_dimensions(X, method='pca', n_components=2, **kwargs):
    """
    Apply dimensionality reduction using the specified method.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    method : str, default='pca'
        Dimensionality reduction method.
        Options: 'pca', 'kernel_pca', 'mds', 'umap'.
    n_components : int, default=2
        Number of components to keep.
    **kwargs : dict
        Additional parameters for the specific method.
        
    Returns
    -------
    tuple
        (transformed_data, model, additional_info)
        additional_info depends on the method:
        - PCA: explained_variance_ratio
        - Kernel PCA: None
        - MDS: stress
        - UMAP: None
    """
    # Ensure data is scaled for methods that are sensitive to scale
    if method in ['mds', 'umap']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    if method == 'pca':
        return apply_pca(X, n_components=n_components, **kwargs)
    
    elif method == 'kernel_pca':
        return apply_kernel_pca(X, n_components=n_components, **kwargs)
    
    elif method == 'mds':
        return apply_mds(X, n_components=n_components, **kwargs)
    
    elif method == 'umap':
        return apply_umap(X, n_components=n_components, **kwargs)
    
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def compare_dimensionality_reduction_methods(X, methods=None, n_components=2):
    """
    Compare different dimensionality reduction methods.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    methods : list, default=None
        List of methods to compare. If None, all methods are used.
        Options: 'pca', 'kernel_pca', 'mds', 'umap'.
    n_components : int, default=2
        Number of components to keep.
        
    Returns
    -------
    dict
        Dictionary with method names as keys and (transformed_data, model, additional_info) as values.
    """
    if methods is None:
        methods = ['pca', 'kernel_pca', 'mds', 'umap']
    
    results = {}
    
    for method in methods:
        print(f"Applying {method}...")
        results[method] = reduce_dimensions(X, method=method, n_components=n_components)
    
    return results
