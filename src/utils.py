"""
Utility functions for WSN Packet Delivery Prediction
"""
import pandas as pd
import joblib

def load_data(path):
    """Load CSV data."""
    return pd.read_csv(path)

def save_pickle(obj, path):
    """Save object to pickle file."""
    joblib.dump(obj, path)

def load_pickle(path):
    """Load object from pickle file."""
    return joblib.load(path)
