"""
Random Forest model training and hyperparameter tuning
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_model(X_train, y_train, save_path=None):
    """
    Train Random Forest with hyperparameter tuning.
    Args:
        X_train: Training features
        y_train: Training labels
        save_path: Path to save model
    Returns:
        best_model: Trained model
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    if save_path:
        joblib.dump(best_model, save_path)
    return best_model
