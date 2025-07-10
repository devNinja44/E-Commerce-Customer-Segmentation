"""
Data preprocessing module for customer segmentation.

This module contains functions for loading, cleaning, and preprocessing
e-commerce customer data for segmentation analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(file_path):
    """
    Load data from a CSV or Excel file.
    
    Parameters
    ----------
    file_path : str
        Path to the data file (CSV or Excel).
        
    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_online_retail_data(file_path=None):
    """
    Load and preprocess the Online Retail dataset.
    
    Parameters
    ----------
    file_path : str, default=None
        Path to the Online Retail dataset. If None, will use the default path.
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed Online Retail dataset with customer features.
    tuple
        (DataFrame, customer_ids) if customer_ids are extracted, otherwise just DataFrame
    """
    # Use absolute path if file_path is not provided
    if file_path is None:
        # Get the absolute path to the data directory
        import os
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(base_dir, 'data', 'raw', 'online_retail.xlsx')
        print(f"Using default data path: {file_path}")
    # Load the data
    df = load_data(file_path)
    if df is None:
        print(f"Failed to load data from {file_path}")
        return None
    
    # Basic cleaning
    print(f"Original data shape: {df.shape}")
    
    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    print(f"Data shape after dropping rows with missing CustomerID: {df.shape}")
    
    # Convert CustomerID to integer and then to string
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Filter out rows with negative quantities or prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print(f"Data shape after filtering out negative quantities/prices: {df.shape}")
    
    # Calculate total purchase amount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Extract customer features (RFM analysis)
    # Get the maximum date to calculate recency
    max_date = df['InvoiceDate'].max()
    
    # Group by customer
    customer_features = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency (days since last purchase)
        'InvoiceNo': 'nunique',  # Frequency (number of purchases)
        'TotalAmount': 'sum',  # Monetary (total spend)
        'Quantity': ['mean', 'sum'],  # Average and total items purchased
        'UnitPrice': 'mean',  # Average price per item
        'StockCode': 'nunique'  # Number of unique items purchased
    })
    
    # Flatten the column names
    customer_features.columns = ['recency', 'frequency', 'monetary', 'avg_items_per_purchase', 
                               'total_items', 'avg_price', 'unique_items']
    
    # Calculate additional features
    customer_features['avg_basket_value'] = customer_features['monetary'] / customer_features['frequency']
    customer_features['avg_item_value'] = customer_features['monetary'] / customer_features['total_items']
    
    # Reset index to make CustomerID a column
    customer_features = customer_features.reset_index()
    
    # Store customer IDs separately
    customer_ids = customer_features['CustomerID'].values
    
    print(f"Final customer features shape: {customer_features.shape}")
    
    return customer_features, customer_ids


def handle_missing_values(df, strategy='mean', categorical_strategy='most_frequent'):
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    strategy : str, default='mean'
        Strategy for imputing numerical missing values.
        Options: 'mean', 'median', 'most_frequent', 'constant'.
    categorical_strategy : str, default='most_frequent'
        Strategy for imputing categorical missing values.
        
    Returns
    -------
    pandas.DataFrame
        Data with missing values handled.
    """
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create imputers
    num_imputer = SimpleImputer(strategy=strategy)
    cat_imputer = SimpleImputer(strategy=categorical_strategy)
    
    # Apply imputation
    if numerical_cols:
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    if categorical_cols:
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df


def remove_outliers(df, method='iqr', threshold=1.5):
    """
    Remove outliers from the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    method : str, default='iqr'
        Method for outlier detection. Options: 'iqr', 'zscore'.
    threshold : float, default=1.5
        Threshold for outlier detection.
        For IQR method: points outside Q1 - threshold*IQR and Q3 + threshold*IQR are outliers.
        For Z-score method: points with |z| > threshold are outliers.
        
    Returns
    -------
    pandas.DataFrame
        Data with outliers removed.
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in numerical_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    elif method == 'zscore':
        for col in numerical_cols:
            z_scores = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
            df_clean = df_clean[abs(z_scores) <= threshold]
    
    return df_clean


def scale_features(df, method='standard', numerical_cols=None):
    """
    Scale numerical features in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    method : str, default='standard'
        Scaling method. Options: 'standard', 'minmax'.
    numerical_cols : list, default=None
        List of numerical columns to scale. If None, all numerical columns are scaled.
        
    Returns
    -------
    pandas.DataFrame
        Data with scaled features.
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    if numerical_cols:
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    
    return df_scaled


def encode_categorical_features(df, categorical_cols=None, method='onehot'):
    """
    Encode categorical features in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    categorical_cols : list, default=None
        List of categorical columns to encode. If None, all object columns are encoded.
    method : str, default='onehot'
        Encoding method. Options: 'onehot', 'label'.
        
    Returns
    -------
    pandas.DataFrame
        Data with encoded categorical features.
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df_encoded = df.copy()
    
    if not categorical_cols:
        return df_encoded
    
    if method == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df_encoded[categorical_cols])
        
        # Create DataFrame with encoded data
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df_encoded.index
        )
        
        # Drop original categorical columns and concatenate encoded columns
        df_encoded = pd.concat([df_encoded.drop(categorical_cols, axis=1), encoded_df], axis=1)
    
    elif method == 'label':
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
    
    return df_encoded


def preprocess_data(df, config=None):
    """
    Preprocess the data using a configurable pipeline.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    config : dict, default=None
        Configuration dictionary with preprocessing options.
        If None, default options are used.
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed data.
    """
    if config is None:
        config = {
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
        }
    
    # Handle missing values
    df_processed = handle_missing_values(
        df,
        strategy=config['missing_values']['strategy'],
        categorical_strategy=config['missing_values']['categorical_strategy']
    )
    
    # Remove outliers
    df_processed = remove_outliers(
        df_processed,
        method=config['outliers']['method'],
        threshold=config['outliers']['threshold']
    )
    
    # Encode categorical features
    df_processed = encode_categorical_features(
        df_processed,
        method=config['encoding']['method']
    )
    
    # Scale features
    df_processed = scale_features(
        df_processed,
        method=config['scaling']['method']
    )
    
    return df_processed
