"""
Visualization functions for WSN Packet Delivery Prediction
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_feature_importance(model, X_train, save_dir='reports'):
    """
    Plot feature importances from Random Forest.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    importances = model.feature_importances_
    features = X_train.columns
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances[indices], y=features[indices])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def plot_relationships(data_path, save_dir='reports'):
    """
    Plot relationships between features and targets.
    """
    df = pd.read_csv(data_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Boxplot: Packet_Size vs Latency_Category
    plt.figure(figsize=(7,5))
    sns.boxplot(x='Latency_Category', y='Packet_Size', data=df)
    plt.title('Packet Size by Latency Category')
    plt.savefig(os.path.join(save_dir, 'boxplot_packet_size_latency.png'))
    plt.close()
    # Scatter: Energy_Level vs PDR
    plt.figure(figsize=(7,5))
    sns.scatterplot(x='Energy_Level', y='PDR', hue='Latency_Category', data=df)
    plt.title('Energy Level vs PDR')
    plt.savefig(os.path.join(save_dir, 'scatter_energy_pdr.png'))
    plt.close()
    # Scatter: Link_Quality vs PDR
    plt.figure(figsize=(7,5))
    sns.scatterplot(x='Link_Quality', y='PDR', hue='Latency_Category', data=df)
    plt.title('Link Quality vs PDR')
    plt.savefig(os.path.join(save_dir, 'scatter_linkquality_pdr.png'))
    plt.close()
