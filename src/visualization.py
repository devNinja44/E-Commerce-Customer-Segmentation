"""
Visualization module for customer segmentation.

This module contains functions for visualizing clustering results
and dimensionality reduction outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap


def plot_elbow_method(k_range, inertia_values, silhouette_values=None, figsize=(12, 5)):
    """
    Plot the elbow method results for KMeans clustering.
    
    Parameters
    ----------
    k_range : array-like
        Range of k values.
    inertia_values : array-like
        Inertia values for each k.
    silhouette_values : array-like, default=None
        Silhouette scores for each k.
    figsize : tuple, default=(12, 5)
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2 if silhouette_values is not None else 1, figsize=figsize)
    
    # Plot inertia
    if silhouette_values is not None:
        ax1 = axes[0]
    else:
        ax1 = axes
    
    ax1.plot(k_range, inertia_values, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for KMeans')
    ax1.grid(True)
    
    # Plot silhouette score if provided
    if silhouette_values is not None:
        ax2 = axes[1]
        ax2.plot(k_range, silhouette_values, 'ro-')
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score for KMeans')
        ax2.grid(True)
    
    plt.tight_layout()
    return fig


def plot_clusters_2d(X_2d, labels, centroids=None, title=None, figsize=(10, 8), alpha=0.7, s=50):
    """
    Plot clusters in 2D.
    
    Parameters
    ----------
    X_2d : array-like of shape (n_samples, 2)
        2D data points.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    centroids : array-like of shape (n_clusters, 2), default=None
        Cluster centroids.
    title : str, default=None
        Plot title.
    figsize : tuple, default=(10, 8)
        Figure size.
    alpha : float, default=0.7
        Alpha value for scatter points.
    s : int, default=50
        Marker size for scatter points.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Create colormap
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    # Plot each cluster
    for label in unique_labels:
        if label == -1:
            # Plot noise points in black
            mask = labels == label
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=s,
                c='black',
                alpha=alpha,
                marker='x',
                label='Noise'
            )
        else:
            # Plot cluster points
            mask = labels == label
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=s,
                c=[cmap(label % n_clusters)],
                alpha=alpha,
                label=f'Cluster {label}'
            )
    
    # Plot centroids if provided
    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=s*3,
            c='red',
            marker='X',
            label='Centroids'
        )
    
    # Add labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title if title else f'Clustering Results ({n_clusters} clusters)')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_clusters_3d(X_3d, labels, centroids=None, title=None, figsize=(12, 10), alpha=0.7, s=50):
    """
    Plot clusters in 3D.
    
    Parameters
    ----------
    X_3d : array-like of shape (n_samples, 3)
        3D data points.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    centroids : array-like of shape (n_clusters, 3), default=None
        Cluster centroids.
    title : str, default=None
        Plot title.
    figsize : tuple, default=(12, 10)
        Figure size.
    alpha : float, default=0.7
        Alpha value for scatter points.
    s : int, default=50
        Marker size for scatter points.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Create colormap
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    # Plot each cluster
    for label in unique_labels:
        if label == -1:
            # Plot noise points in black
            mask = labels == label
            ax.scatter(
                X_3d[mask, 0],
                X_3d[mask, 1],
                X_3d[mask, 2],
                s=s,
                c='black',
                alpha=alpha,
                marker='x',
                label='Noise'
            )
        else:
            # Plot cluster points
            mask = labels == label
            ax.scatter(
                X_3d[mask, 0],
                X_3d[mask, 1],
                X_3d[mask, 2],
                s=s,
                c=[cmap(label % n_clusters)],
                alpha=alpha,
                label=f'Cluster {label}'
            )
    
    # Plot centroids if provided
    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            s=s*3,
            c='red',
            marker='X',
            label='Centroids'
        )
    
    # Add labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title if title else f'Clustering Results ({n_clusters} clusters)')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_cluster_profiles(cluster_profiles, figsize=(14, 8)):
    """
    Plot cluster profiles as a heatmap.
    
    Parameters
    ----------
    cluster_profiles : pandas.DataFrame
        DataFrame with cluster profiles.
    figsize : tuple, default=(14, 8)
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    # Drop the Size column if it exists
    profiles = cluster_profiles.drop('Size', axis=1, errors='ignore')
    
    # Normalize profiles for better visualization
    normalized_profiles = (profiles - profiles.mean()) / profiles.std()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(normalized_profiles, annot=True, cmap='coolwarm', center=0, ax=ax)
    
    ax.set_title('Cluster Profiles (Normalized)')
    ax.set_ylabel('Cluster')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(cluster_profiles, figsize=(12, 8)):
    """
    Plot feature importance for each cluster.
    
    Parameters
    ----------
    cluster_profiles : pandas.DataFrame
        DataFrame with cluster profiles.
    figsize : tuple, default=(12, 8)
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    # Drop the Size column if it exists
    profiles = cluster_profiles.drop('Size', axis=1, errors='ignore')
    
    # Normalize profiles
    normalized_profiles = (profiles - profiles.mean()) / profiles.std()
    
    # Get number of clusters and features
    n_clusters = normalized_profiles.shape[0]
    n_features = normalized_profiles.shape[1]
    
    # Create subplots
    fig, axes = plt.subplots(n_clusters, 1, figsize=figsize)
    
    # Handle case with only one cluster
    if n_clusters == 1:
        axes = [axes]
    
    # Plot feature importance for each cluster
    for i, (cluster_id, profile) in enumerate(normalized_profiles.iterrows()):
        # Sort features by absolute importance
        sorted_features = profile.abs().sort_values(ascending=False)
        
        # Plot horizontal bar chart
        ax = axes[i]
        bars = ax.barh(
            sorted_features.index,
            sorted_features.values,
            color=['red' if v < 0 else 'green' for v in profile[sorted_features.index]]
        )
        
        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                    va='center')
        
        ax.set_title(f'Cluster {cluster_id} - Feature Importance')
        ax.set_xlabel('Normalized Value')
        ax.grid(axis='x')
    
    plt.tight_layout()
    return fig


def create_interactive_scatter(X_2d, labels, hover_data=None, title=None):
    """
    Create an interactive scatter plot using Plotly.
    
    Parameters
    ----------
    X_2d : array-like of shape (n_samples, 2)
        2D data points.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    hover_data : pandas.DataFrame, default=None
        Additional data to show on hover.
    title : str, default=None
        Plot title.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object.
    """
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'cluster': [f'Cluster {label}' if label != -1 else 'Noise' for label in labels]
    })
    
    # Add hover data if provided
    if hover_data is not None:
        for col in hover_data.columns:
            df_plot[col] = hover_data[col].values
    
    # Create interactive scatter plot
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='cluster',
        title=title if title else 'Interactive Cluster Visualization',
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        hover_data=df_plot.columns if hover_data is not None else None,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Update layout
    fig.update_layout(
        legend_title_text='Cluster',
        plot_bgcolor='white',
        width=900,
        height=700
    )
    
    # Update traces
    fig.update_traces(
        marker=dict(size=10, opacity=0.7),
        selector=dict(mode='markers')
    )
    
    return fig


def compare_dimensionality_reduction_methods(X_dict, labels_dict, figsize=(15, 10)):
    """
    Compare different dimensionality reduction methods with clustering results.
    
    Parameters
    ----------
    X_dict : dict
        Dictionary with method names as keys and 2D data as values.
    labels_dict : dict
        Dictionary with method names as keys and cluster labels as values.
    figsize : tuple, default=(15, 10)
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    n_methods = len(X_dict)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Plot each method
    for i, (method_name, X_2d) in enumerate(X_dict.items()):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        # Get labels for this method
        labels = labels_dict.get(method_name, None)
        
        # If no specific labels for this method, use the first available
        if labels is None and labels_dict:
            labels = next(iter(labels_dict.values()))
        
        # Skip if no labels available
        if labels is None:
            continue
        
        # Get unique labels
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Create colormap
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        
        # Plot each cluster
        for label in unique_labels:
            if label == -1:
                # Plot noise points in black
                mask = labels == label
                ax.scatter(
                    X_2d[mask, 0],
                    X_2d[mask, 1],
                    s=30,
                    c='black',
                    alpha=0.7,
                    marker='x',
                    label='Noise'
                )
            else:
                # Plot cluster points
                mask = labels == label
                ax.scatter(
                    X_2d[mask, 0],
                    X_2d[mask, 1],
                    s=30,
                    c=[cmap(label % n_clusters)],
                    alpha=0.7,
                    label=f'Cluster {label}'
                )
        
        # Add title
        ax.set_title(f'{method_name} ({n_clusters} clusters)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        # Add legend to the first plot only
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(X_dict), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
