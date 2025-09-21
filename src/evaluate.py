"""
Model evaluation and plotting
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os

def evaluate_model(model, X_test, y_test, save_dir='reports'):
    """
    Evaluate model and generate plots.
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save outputs
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=None)
    rec = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    # ROC curves
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        n_classes = y_score.shape[1]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
        plt.close()
    # Classification report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print(f"Accuracy: {acc:.3f}")
    print(report)
