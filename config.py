"""
Configuration file for Ola Driver Attrition Prediction Project
Contains paths, settings, and constants
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UPLOADS_DIR = DATA_DIR / "uploads"

# Model directories
MODELS_DIR = BASE_DIR / "models"

# Reports directory
REPORTS_DIR = BASE_DIR / "reports"

# Artifacts directory
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, UPLOADS_DIR,
                  MODELS_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model metadata file
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Dataset metadata file
DATASET_METADATA_FILE = DATA_DIR / "dataset_metadata.json"

# Default parameters
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# Feature columns
NUMERICAL_FEATURES = ['Age', 'Income', 'tenure', 'Quarterly Rating']
CATEGORICAL_FEATURES = ['City', 'Education_Level', 'Joining Designation', 'Grade']
BINARY_FEATURES = ['Gender', 'rating_increased', 'income_increased']

# Model options
AVAILABLE_MODELS = {
    'Random Forest': 'RandomForestClassifier',
    'Gradient Boosting': 'GradientBoostingClassifier',
    'XGBoost': 'XGBClassifier',
    'Extra Trees': 'ExtraTreesClassifier'
}

# Hyperparameter grids
PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0]
    },
    'Extra Trees': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

# Streamlit page config
PAGE_TITLE = "OLA Driver Attrition Prediction System"
PAGE_ICON = "🚗"
LAYOUT = "wide"

# Color scheme for visualizations
COLOR_PALETTE = "viridis"
