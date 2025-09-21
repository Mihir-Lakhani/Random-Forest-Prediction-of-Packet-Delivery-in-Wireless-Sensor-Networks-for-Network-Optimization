"""
Main pipeline for WSN Packet Delivery Prediction
"""
from src.data_preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_feature_importance, plot_relationships
import os

DATA_PATH = os.path.join('data', 'wsn_dataset.csv')
PREPROCESSED_PATH = os.path.join('data', 'preprocessed_data.pkl')
MODEL_PATH = os.path.join('reports', 'best_rf_model.pkl')

if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH, PREPROCESSED_PATH)

    # Train model
    best_model = train_model(X_train, y_train, MODEL_PATH)

    # Evaluate model
    evaluate_model(best_model, X_test, y_test)

    # Visualizations
    plot_feature_importance(best_model, X_train)
    plot_relationships(DATA_PATH)
