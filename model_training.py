"""
Model training, evaluation, and hyperparameter tuning module
"""
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import streamlit as st
from config import PARAM_GRIDS, DEFAULT_CV_FOLDS

# Check if xgboost is available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def get_model_instance(model_type, **hyperparameters):
    """
    Get model instance based on type

    Args:
        model_type: Type of model
        **hyperparameters: Model hyperparameters

    Returns:
        Model instance
    """
    if model_type == 'Random Forest':
        return RandomForestClassifier(**hyperparameters)
    elif model_type == 'Gradient Boosting':
        return GradientBoostingClassifier(**hyperparameters)
    elif model_type == 'Extra Trees':
        return ExtraTreesClassifier(**hyperparameters)
    elif model_type == 'XGBoost' and XGBOOST_AVAILABLE:
        return XGBClassifier(**hyperparameters, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_random_forest(X_train, y_train, **hyperparameters):
    """Train Random Forest model"""
    if not hyperparameters:
        hyperparameters = {'n_estimators': 100, 'random_state': 42}

    rf_model = RandomForestClassifier(**hyperparameters)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_gradient_boosting(X_train, y_train, **hyperparameters):
    """Train Gradient Boosting model"""
    if not hyperparameters:
        hyperparameters = {'n_estimators': 100, 'random_state': 42}

    gb_model = GradientBoostingClassifier(**hyperparameters)
    gb_model.fit(X_train, y_train)
    return gb_model

def train_extra_trees(X_train, y_train, **hyperparameters):
    """Train Extra Trees model"""
    if not hyperparameters:
        hyperparameters = {'n_estimators': 100, 'random_state': 42}

    et_model = ExtraTreesClassifier(**hyperparameters)
    et_model.fit(X_train, y_train)
    return et_model

def train_xgboost(X_train, y_train, **hyperparameters):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")

    if not hyperparameters:
        hyperparameters = {'n_estimators': 100, 'random_state': 42}

    xgb_model = XGBClassifier(**hyperparameters, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_model(model_type, X_train, y_train, **hyperparameters):
    """
    Train model based on type

    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training labels
        **hyperparameters: Model hyperparameters

    Returns:
        Trained model
    """
    if model_type == 'Random Forest':
        return train_random_forest(X_train, y_train, **hyperparameters)
    elif model_type == 'Gradient Boosting':
        return train_gradient_boosting(X_train, y_train, **hyperparameters)
    elif model_type == 'Extra Trees':
        return train_extra_trees(X_train, y_train, **hyperparameters)
    elif model_type == 'XGBoost':
        return train_xgboost(X_train, y_train, **hyperparameters)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_model(model, X_test, y_test, return_metrics=False):
    """
    Evaluate model performance

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        return_metrics: If True, return metrics dict instead of displaying

    Returns:
        Dictionary of metrics if return_metrics=True
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    if return_metrics:
        return metrics

    # Display metrics in Streamlit
    st.write("**Classification Report:**")
    st.code(classification_report(y_test, y_pred))

    st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
    st.write(f"**Precision:** {metrics['precision']:.4f}")
    st.write(f"**Recall:** {metrics['recall']:.4f}")
    st.write(f"**F1-Score:** {metrics['f1_score']:.4f}")
    st.write(f"**ROC AUC Score:** {metrics['roc_auc']:.4f}")

    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    return metrics

def hyperparameter_tuning(model_type, X_train, y_train, search_type='grid', cv=DEFAULT_CV_FOLDS,
                         n_iter=20, custom_param_grid=None):
    """
    Perform hyperparameter tuning using Grid Search or Random Search

    Args:
        model_type: Type of model
        X_train: Training features
        y_train: Training labels
        search_type: 'grid' or 'random'
        cv: Number of cross-validation folds
        n_iter: Number of iterations for random search
        custom_param_grid: Custom parameter grid (optional)

    Returns:
        best_model: Best model from search
        best_params: Best hyperparameters
        cv_results: Cross-validation results
    """
    # Get parameter grid
    param_grid = custom_param_grid or PARAM_GRIDS.get(model_type, {})

    if not param_grid:
        raise ValueError(f"No parameter grid defined for {model_type}")

    # Get base model
    base_model = get_model_instance(model_type)

    # Perform search
    if search_type == 'grid':
        search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1, return_train_score=True
        )
    else:  # random search
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=n_iter, cv=cv,
            scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42,
            return_train_score=True
        )

    # Fit search
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.cv_results_

def handle_class_imbalance(X, y, strategy='smote', random_state=42):
    """
    Handle class imbalance using SMOTE or other techniques

    Args:
        X: Features
        y: Labels
        strategy: Resampling strategy ('smote', 'over', 'under')
        random_state: Random state

    Returns:
        X_resampled, y_resampled: Resampled data
    """
    if strategy == 'smote':
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        # Default to SMOTE if strategy not recognized
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

def save_model(model, filename):
    """Save model to file"""
    joblib.dump(model, filename)

def load_model(filename):
    """Load model from file"""
    return joblib.load(filename)

def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models

    Args:
        models_dict: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels

    Returns:
        DataFrame with comparison results
    """
    import pandas as pd

    comparison_results = []

    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, return_metrics=True)
        metrics['Model'] = model_name
        comparison_results.append(metrics)

    return pd.DataFrame(comparison_results)[
        ['Model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    ]

def get_feature_importance(model):
    """Get feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    else:
        return None

def cross_validate_model(model_type, X, y, cv=5, **hyperparameters):
    """
    Perform cross-validation on model

    Args:
        model_type: Type of model
        X: Features
        y: Labels
        cv: Number of folds
        **hyperparameters: Model hyperparameters

    Returns:
        Dictionary with CV scores
    """
    from sklearn.model_selection import cross_val_score

    model = get_model_instance(model_type, **hyperparameters)

    scores = {
        'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
        'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc'),
        'f1': cross_val_score(model, X, y, cv=cv, scoring='f1')
    }

    return {
        'accuracy_mean': scores['accuracy'].mean(),
        'accuracy_std': scores['accuracy'].std(),
        'roc_auc_mean': scores['roc_auc'].mean(),
        'roc_auc_std': scores['roc_auc'].std(),
        'f1_mean': scores['f1'].mean(),
        'f1_std': scores['f1'].std()
    }
