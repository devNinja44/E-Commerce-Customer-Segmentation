"""
Configuration module for customer segmentation.

This module contains configuration options for the customer segmentation pipeline.
"""

import argparse


def get_default_config():
    """
    Get default configuration for the customer segmentation pipeline.
    
    Returns
    -------
    dict
        Default configuration dictionary.
    """
    return {
        'data': {
            'use_synthetic': True,
            'n_samples': 1000,
            'random_state': 42
        },
        'preprocessing': {
            'missing_values': {
                'strategy': 'mean',
                'categorical_strategy': 'most_frequent'
            },
            'outliers': {
                'method': 'iqr',
                'threshold': 1.5
            },
            'scaling': {
                'method': 'standard'
            },
            'encoding': {
                'method': 'onehot'
            }
        },
        'dimensionality_reduction': {
            'method': 'pca',  # Options: 'pca', 'kernel_pca', 'mds', 'umap'
            'n_components': 2,
            'random_state': 42,
            'pca': {},
            'kernel_pca': {
                'kernel': 'rbf',
                'gamma': None
            },
            'mds': {
                'metric': True,
                'n_jobs': -1
            },
            'umap': {
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean'
            }
        },
        'clustering': {
            'method': 'kmeans',  # Options: 'kmeans', 'dbscan', 'agglomerative'
            'kmeans': {
                'n_clusters': 3,
                'random_state': 42
            },
            'dbscan': {
                'eps': 0.5,
                'min_samples': 5
            },
            'agglomerative': {
                'n_clusters': 3,
                'linkage': 'ward'
            }
        },
        'evaluation': {
            'metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        },
        'visualization': {
            'figsize': (10, 8),
            'alpha': 0.7,
            's': 50
        }
    }


def parse_args():
    """
    Parse command-line arguments for the customer segmentation pipeline.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Customer Segmentation Pipeline')
    
    # Data options
    parser.add_argument('--use-synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples for synthetic data')
    parser.add_argument('--data-path', type=str, help='Path to data file')
    
    # Preprocessing options
    parser.add_argument('--missing-strategy', type=str, default='mean', help='Strategy for handling missing values')
    parser.add_argument('--outlier-method', type=str, default='iqr', help='Method for outlier detection')
    parser.add_argument('--scaling-method', type=str, default='standard', help='Method for feature scaling')
    
    # Dimensionality reduction options
    parser.add_argument('--dim-reduction', type=str, default='pca', help='Dimensionality reduction method')
    parser.add_argument('--n-components', type=int, default=2, help='Number of components for dimensionality reduction')
    
    # Clustering options
    parser.add_argument('--clustering', type=str, default='kmeans', help='Clustering method')
    parser.add_argument('--n-clusters', type=int, default=3, help='Number of clusters for KMeans and Agglomerative')
    parser.add_argument('--eps', type=float, default=0.5, help='Epsilon parameter for DBSCAN')
    parser.add_argument('--min-samples', type=int, default=5, help='Min samples parameter for DBSCAN')
    
    # Random state
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """
    Update configuration dictionary from command-line arguments.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    args : argparse.Namespace
        Parsed command-line arguments.
        
    Returns
    -------
    dict
        Updated configuration dictionary.
    """
    # Update data options
    if args.use_synthetic is not None:
        config['data']['use_synthetic'] = args.use_synthetic
    if args.n_samples is not None:
        config['data']['n_samples'] = args.n_samples
    if args.data_path is not None:
        config['data']['path'] = args.data_path
    
    # Update preprocessing options
    if args.missing_strategy is not None:
        config['preprocessing']['missing_values']['strategy'] = args.missing_strategy
    if args.outlier_method is not None:
        config['preprocessing']['outliers']['method'] = args.outlier_method
    if args.scaling_method is not None:
        config['preprocessing']['scaling']['method'] = args.scaling_method
    
    # Update dimensionality reduction options
    if args.dim_reduction is not None:
        config['dimensionality_reduction']['method'] = args.dim_reduction
    if args.n_components is not None:
        config['dimensionality_reduction']['n_components'] = args.n_components
    
    # Update clustering options
    if args.clustering is not None:
        config['clustering']['method'] = args.clustering
    if args.n_clusters is not None:
        config['clustering']['kmeans']['n_clusters'] = args.n_clusters
        config['clustering']['agglomerative']['n_clusters'] = args.n_clusters
    if args.eps is not None:
        config['clustering']['dbscan']['eps'] = args.eps
    if args.min_samples is not None:
        config['clustering']['dbscan']['min_samples'] = args.min_samples
    
    # Update random state
    if args.random_state is not None:
        config['data']['random_state'] = args.random_state
        config['dimensionality_reduction']['random_state'] = args.random_state
        config['clustering']['kmeans']['random_state'] = args.random_state
    
    return config


def get_config():
    """
    Get configuration for the customer segmentation pipeline.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    config = get_default_config()
    try:
        args = parse_args()
        config = update_config_from_args(config, args)
    except:
        # If running in a notebook or without command-line arguments
        pass
    
    return config
