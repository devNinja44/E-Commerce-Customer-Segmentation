"""
Interactive dashboard for customer segmentation using Streamlit.

This module provides an interactive dashboard for exploring customer segments.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_online_retail_data, preprocess_data
from src.dimensionality_reduction import reduce_dimensions
from src.clustering import cluster_data, find_optimal_k
from src.evaluation import analyze_clusters, generate_cluster_labels
from src.config import get_default_config


def run_dashboard():
    """
    Run the Streamlit dashboard for customer segmentation.
    """
    st.set_page_config(
        page_title="Customer Segmentation Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("E-Commerce Customer Segmentation Dashboard")
    st.write("""
    This dashboard allows you to explore customer segments based on behavioral and transactional data.
    You can adjust various parameters to see how they affect the clustering results.
    """)

    # Sidebar for configuration
    st.sidebar.title("Configuration")

    # Data configuration
    st.sidebar.header("Data")
    data_path = st.sidebar.text_input("Data path", value="../data/raw/online_retail.xlsx")
    random_state = st.sidebar.slider("Random seed", 0, 100, 42)

    # Preprocessing configuration
    st.sidebar.header("Preprocessing")
    missing_strategy = st.sidebar.selectbox(
        "Missing values strategy",
        ["mean", "median", "most_frequent"],
        index=0
    )
    outlier_method = st.sidebar.selectbox(
        "Outlier removal method",
        ["iqr", "zscore"],
        index=0
    )
    scaling_method = st.sidebar.selectbox(
        "Scaling method",
        ["standard", "minmax"],
        index=0
    )

    # Dimensionality reduction configuration
    st.sidebar.header("Dimensionality Reduction")
    dim_reduction_method = st.sidebar.selectbox(
        "Method",
        ["pca", "kernel_pca", "mds", "umap"],
        index=0
    )
    n_components = st.sidebar.slider("Number of components", 2, 5, 2)

    # Additional parameters for specific methods
    dim_reduction_params = {}
    if dim_reduction_method == "kernel_pca":
        kernel = st.sidebar.selectbox(
            "Kernel",
            ["rbf", "poly", "sigmoid", "cosine"],
            index=0
        )
        dim_reduction_params["kernel"] = kernel
    elif dim_reduction_method == "umap":
        n_neighbors = st.sidebar.slider("Number of neighbors", 5, 50, 15)
        min_dist = st.sidebar.slider("Minimum distance", 0.0, 1.0, 0.1, 0.05)
        dim_reduction_params["n_neighbors"] = n_neighbors
        dim_reduction_params["min_dist"] = min_dist

    # Clustering configuration
    st.sidebar.header("Clustering")
    clustering_method = st.sidebar.selectbox(
        "Method",
        ["kmeans", "dbscan", "agglomerative"],
        index=0
    )

    # Additional parameters for specific methods
    clustering_params = {}
    if clustering_method == "kmeans":
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        clustering_params["n_clusters"] = n_clusters
    elif clustering_method == "dbscan":
        eps = st.sidebar.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Minimum samples", 2, 20, 5)
        clustering_params["eps"] = eps
        clustering_params["min_samples"] = min_samples
    elif clustering_method == "agglomerative":
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        linkage = st.sidebar.selectbox(
            "Linkage",
            ["ward", "complete", "average", "single"],
            index=0
        )
        clustering_params["n_clusters"] = n_clusters
        clustering_params["linkage"] = linkage

    # Run button
    if st.sidebar.button("Run Segmentation"):
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Load data
        status_text.text("Loading and processing data...")
        try:
            df, customer_ids = load_online_retail_data(file_path=data_path)
            if df is None:
                st.error(f"Failed to load data from {data_path}")
                return
            progress_bar.progress(20)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

        # Preprocess data
        status_text.text("Preprocessing data...")
        config = get_default_config()
        config['preprocessing']['missing_values']['strategy'] = missing_strategy
        config['preprocessing']['outliers']['method'] = outlier_method
        config['preprocessing']['scaling']['method'] = scaling_method
        df_processed = preprocess_data(df, config=config['preprocessing'])
        progress_bar.progress(40)

        # Apply dimensionality reduction
        status_text.text(f"Applying {dim_reduction_method} dimensionality reduction...")
        X_reduced, model, additional_info = reduce_dimensions(
            df_processed.values,
            method=dim_reduction_method,
            n_components=n_components,
            random_state=random_state,
            **dim_reduction_params
        )
        progress_bar.progress(60)
        
        feature_names = df_processed.columns.tolist()

        # Apply clustering
        status_text.text(f"Applying {clustering_method} clustering...")
        labels, model, metrics = cluster_data(
            X_reduced,
            method=clustering_method,
            **clustering_params
        )
        progress_bar.progress(80)

        # Analyze clusters
        status_text.text("Analyzing clusters...")
        # Create a DataFrame with reduced dimensions for analysis
        reduced_df = pd.DataFrame(X_reduced, columns=[f"Component_{i+1}" for i in range(X_reduced.shape[1])])
        # Use the reduced data for cluster analysis
        cluster_profiles = analyze_clusters(X_reduced, labels, reduced_df.columns.tolist())
        cluster_labels = generate_cluster_labels(cluster_profiles)
        progress_bar.progress(100)
        status_text.text("Done!")

        # Display results
        st.header("Clustering Results")

        # Display metrics
        st.subheader("Clustering Metrics")
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
        metrics_df.columns = ["Value"]
        st.dataframe(metrics_df)

        # Display cluster visualization
        st.subheader("Cluster Visualization")
        
        # Create a DataFrame for plotting
        df_plot = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1] if X_reduced.shape[1] > 1 else np.zeros(X_reduced.shape[0]),
            'z': X_reduced[:, 2] if X_reduced.shape[1] > 2 else np.zeros(X_reduced.shape[0]),
            'cluster': [f'Cluster {label}' if label != -1 else 'Noise' for label in labels]
        })
        
        # Only add customer_ids if they match the length of X_reduced
        if len(customer_ids) == len(X_reduced):
            df_plot['customer_id'] = customer_ids
        else:
            # Generate placeholder IDs if lengths don't match
            df_plot['customer_id'] = [f'Customer_{i}' for i in range(len(X_reduced))]
            st.warning(f"Note: Using placeholder customer IDs due to length mismatch (X_reduced: {len(X_reduced)}, customer_ids: {len(customer_ids)})")
        
        # We don't add original features for hover data since they might have different lengths
        # Instead, we'll use the reduced dimensions and cluster information
        
        # Create interactive scatter plot
        if n_components == 2:
            fig = px.scatter(
                df_plot,
                x='x',
                y='y',
                color='cluster',
                hover_data=['customer_id'],  # Only use columns that exist in df_plot
                title=f'{clustering_method.upper()} Clustering Results',
                labels={'x': 'Component 1', 'y': 'Component 2'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        else:  # 3D plot
            fig = px.scatter_3d(
                df_plot,
                x='x',
                y='y',
                z='z',
                color='cluster',
                hover_data=['customer_id'],  # Only use columns that exist in df_plot
                title=f'{clustering_method.upper()} Clustering Results',
                labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        
        # Update layout
        fig.update_layout(
            legend_title_text='Cluster',
            plot_bgcolor='white',
            height=600
        )
        
        # Update traces
        fig.update_traces(
            marker=dict(size=8, opacity=0.7),
            selector=dict(mode='markers')
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Display cluster profiles
        st.subheader("Cluster Profiles")
        st.dataframe(cluster_profiles)

        # Display cluster labels
        st.subheader("Cluster Labels")
        labels_df = pd.DataFrame({
            'Cluster': list(cluster_labels.keys()),
            'Label': list(cluster_labels.values()),
            'Size': [cluster_profiles.loc[cluster_id, 'Size'] if 'Size' in cluster_profiles.columns else 'N/A' 
                    for cluster_id in cluster_labels.keys()]
        })
        st.dataframe(labels_df)

        # Display customer segments
        st.subheader("Customer Segments")
        
        # Check if lengths match before creating DataFrame
        if len(customer_ids) == len(labels):
            customer_clusters = pd.DataFrame({
                'customer_id': customer_ids,
                'cluster': labels,
                'cluster_name': [cluster_labels[label] if label in cluster_labels else 'Unknown' for label in labels]
            })
            st.dataframe(customer_clusters)
        else:
            st.warning(f"Cannot display customer segments: length mismatch between customer IDs ({len(customer_ids)}) and cluster labels ({len(labels)})")
            # Create a simplified DataFrame with just the cluster information
            cluster_summary = pd.DataFrame({
                'cluster': sorted(list(set(labels))),
                'count': [list(labels).count(label) for label in sorted(list(set(labels)))],
                'cluster_name': [cluster_labels[label] if label in cluster_labels else 'Unknown' for label in sorted(list(set(labels)))]
            })
            st.dataframe(cluster_summary)

        # Feature importance by cluster
        st.subheader("Feature Importance by Cluster")
        
        # Drop the Size column if it exists
        profiles = cluster_profiles.drop('Size', axis=1, errors='ignore')
        
        # Normalize profiles
        normalized_profiles = (profiles - profiles.mean()) / profiles.std()
        
        # Create tabs for each cluster
        tabs = st.tabs([f"Cluster {cluster_id}" for cluster_id in normalized_profiles.index])
        
        for i, (cluster_id, profile) in enumerate(normalized_profiles.iterrows()):
            with tabs[i]:
                # Sort features by absolute importance
                sorted_features = profile.abs().sort_values(ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    x=sorted_features.values,
                    y=sorted_features.index,
                    orientation='h',
                    color=profile[sorted_features.index],
                    color_continuous_scale='RdBu_r',
                    title=f'Feature Importance for Cluster {cluster_id}',
                    labels={'x': 'Normalized Value', 'y': 'Feature', 'color': 'Value'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Cluster {cluster_id}**: {cluster_labels[cluster_id]}")


if __name__ == "__main__":
    run_dashboard()
