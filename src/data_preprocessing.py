"""
Data preprocessing for WSN Packet Delivery Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(data_path, save_path=None):
    """
    Load, clean, encode, scale, and split the dataset.
    Args:
        data_path (str): Path to CSV file.
        save_path (str): Path to save preprocessed data (optional).
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(data_path)
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    categorical_columns = ['Node_ID', 'Congestion_Status', 'Traffic_Class', 'Routing_Algorithm', 'Latency_Category']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Features and target
    X = df.drop(['Latency_Category'], axis=1)
    y = df['Latency_Category']
    # Scale numeric features
    num_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Save if needed
    if save_path:
        joblib.dump((X_train, X_test, y_train, y_test, label_encoders, scaler), save_path)
    return X_train, X_test, y_train, y_test
