"""
Utility functions for visualization and reporting
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve
import io
from datetime import datetime

def plot_feature_importance(model, X, top_n=20):
    """Plot feature importance for tree-based models"""
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n), ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    return fig

def plot_roc_curve(y_test, y_prob, model_name="Model"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix_heatmap(y_test, y_pred, model_name="Model"):
    """Plot confusion matrix as heatmap"""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=['No Attrition', 'Attrition'],
                yticklabels=['No Attrition', 'Attrition'])
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1%})',
                   ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    return fig

def plot_precision_recall_curve(y_test, y_prob, model_name="Model"):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_learning_curve(estimator, X, y, cv=5, scoring='roc_auc'):
    """Plot learning curve to diagnose bias-variance"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel(f'{scoring.upper()} Score', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_correlation_matrix(X, figsize=(12, 10)):
    """Plot correlation matrix heatmap"""
    fig, ax = plt.subplots(figsize=figsize)
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, center=0, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix of Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_model_comparison(comparison_df):
    """Plot model performance comparison"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            ax = axes[idx]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax,
                             color='skyblue', legend=False)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=9)

    # Hide the last subplot if we have fewer than 6 metrics
    if len(metrics) < 6:
        axes[-1].axis('off')

    plt.tight_layout()
    return fig

def plot_feature_importance_comparison(models_dict, feature_names, top_n=15):
    """Compare feature importance across multiple models"""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(8*len(models_dict), 6))

    if len(models_dict) == 1:
        axes = [axes]

    for idx, (model_name, model) in enumerate(models_dict.items()):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        axes[idx].barh(importance_df['feature'], importance_df['importance'], color='teal')
        axes[idx].set_title(f'{model_name}\nTop {top_n} Features', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Importance', fontsize=10)
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_training_history(history_df):
    """Plot training history over time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(history_df['timestamp'], history_df['roc_auc'], marker='o', label='ROC-AUC', linewidth=2)
    ax.plot(history_df['timestamp'], history_df['f1_score'], marker='s', label='F1-Score', linewidth=2)
    ax.plot(history_df['timestamp'], history_df['accuracy'], marker='^', label='Accuracy', linewidth=2)

    ax.set_xlabel('Training Date', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def generate_classification_report_df(y_test, y_pred):
    """Generate classification report as DataFrame"""
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    return df

def plot_distribution_comparison(df1, df2, column, labels=['Dataset 1', 'Dataset 2']):
    """Compare distributions of a column across two datasets"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df1[column].dropna(), alpha=0.5, label=labels[0], bins=30, color='blue')
    ax.hist(df2[column].dropna(), alpha=0.5, label=labels[1], bins=30, color='orange')

    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution Comparison: {column}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def create_prediction_report_df(predictions, probabilities, feature_values):
    """Create a detailed prediction report DataFrame"""
    report_df = pd.DataFrame({
        'Prediction': ['Attrition' if p == 1 else 'No Attrition' for p in predictions],
        'Attrition_Probability': probabilities,
        'Confidence': ['High' if (p > 0.7 or p < 0.3) else 'Medium' if (p > 0.6 or p < 0.4) else 'Low'
                      for p in probabilities]
    })

    # Add feature values
    for col, val in feature_values.items():
        report_df[col] = val

    return report_df

def format_timestamp():
    """Get formatted timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
