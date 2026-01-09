"""
Data preprocessing and feature engineering module
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from config import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, NUMERICAL_FEATURES

def load_data(file_path):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess data: feature engineering, encoding, imputation, scaling

    Args:
        df: Input DataFrame

    Returns:
        X: Preprocessed features
        y: Target variable
        imputer: Fitted imputer
        scaler: Fitted scaler
        feature_names: List of feature names
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Convert date columns to datetime
    date_columns = ['Dateofjoining', 'LastWorkingDate', 'MMM-YY']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%y', errors='coerce')

    # Create target variable
    df['target'] = np.where(df['LastWorkingDate'].notna(), 1, 0)

    # Feature engineering
    df['tenure'] = (df['LastWorkingDate'].fillna(pd.Timestamp.now()) - df['Dateofjoining']).dt.days

    # Rating and income trends
    if 'Driver_ID' in df.columns:
        df['rating_increased'] = df.groupby('Driver_ID')['Quarterly Rating'].diff() > 0
        df['income_increased'] = df.groupby('Driver_ID')['Income'].diff() > 0
    else:
        df['rating_increased'] = 0
        df['income_increased'] = 0

    # Fill boolean features
    df['rating_increased'] = df['rating_increased'].fillna(0).astype(int)
    df['income_increased'] = df['income_increased'].fillna(0).astype(int)

    # Select relevant features
    features = ['Age', 'Gender', 'City', 'Education_Level', 'Income', 'tenure',
                'Joining Designation', 'Grade', 'Quarterly Rating',
                'rating_increased', 'income_increased']

    # Keep only features that exist in the dataframe
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df['target']

    # Encode categorical variables
    categorical_cols = ['City', 'Education_Level', 'Joining Designation', 'Grade']
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Store feature names before imputation
    feature_names = X.columns.tolist()

    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale numerical features
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in X_imputed.columns]
    scaler = StandardScaler()
    X_imputed[numerical_cols] = scaler.fit_transform(X_imputed[numerical_cols])

    return X_imputed, y, imputer, scaler, feature_names

def split_data(X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preprocess_user_input(user_input, df_columns, imputer, scaler):
    """
    Preprocess user input for prediction

    Args:
        user_input: Dictionary of user input values
        df_columns: Expected column names
        imputer: Fitted imputer
        scaler: Fitted scaler

    Returns:
        user_df: Preprocessed DataFrame ready for prediction
    """
    # Initialize a dictionary to hold the processed input
    processed_input = {col: 0 for col in df_columns}

    # Process categorical variables
    categorical_columns = ['City', 'Education_Level', 'Joining Designation', 'Grade']
    for col in categorical_columns:
        if col in user_input:
            # The user_input might contain the encoded column name
            processed_input[user_input[col]] = 1

    # Process numerical and binary variables
    numerical_binary_cols = ['Age', 'Gender', 'Income', 'tenure', 'Quarterly Rating',
                             'rating_increased', 'income_increased']
    for col in numerical_binary_cols:
        if col in user_input:
            processed_input[col] = user_input[col]

    # Create a DataFrame with a single row
    user_df = pd.DataFrame([processed_input])

    # Ensure all columns from the training data are present
    for col in df_columns:
        if col not in user_df.columns:
            user_df[col] = 0

    # Reorder columns to match the training data
    user_df = user_df[df_columns]

    # Impute missing values
    user_df_imputed = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)

    # Scale numerical features
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in user_df_imputed.columns]
    user_df_imputed[numerical_cols] = scaler.transform(user_df_imputed[numerical_cols])

    return user_df_imputed

def preprocess_batch_input(batch_df, df_columns, imputer, scaler):
    """
    Preprocess batch input (multiple rows) for prediction

    Args:
        batch_df: DataFrame with multiple rows of input data
        df_columns: Expected column names
        imputer: Fitted imputer
        scaler: Fitted scaler

    Returns:
        processed_df: Preprocessed DataFrame ready for prediction
    """
    # Store original data for reference
    original_df = batch_df.copy()

    # Apply same preprocessing as training data
    # Encode categorical variables
    categorical_cols = ['City', 'Education_Level', 'Joining Designation', 'Grade']
    categorical_cols = [c for c in categorical_cols if c in batch_df.columns]

    if categorical_cols:
        batch_df = pd.get_dummies(batch_df, columns=categorical_cols, drop_first=True)

    # Ensure all columns from training data are present
    for col in df_columns:
        if col not in batch_df.columns:
            batch_df[col] = 0

    # Reorder columns to match training data
    batch_df = batch_df[df_columns]

    # Impute missing values
    batch_df_imputed = pd.DataFrame(imputer.transform(batch_df), columns=batch_df.columns)

    # Scale numerical features
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in batch_df_imputed.columns]
    batch_df_imputed[numerical_cols] = scaler.transform(batch_df_imputed[numerical_cols])

    return batch_df_imputed, original_df

def get_data_quality_report(df):
    """
    Generate data quality report

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with data quality metrics
    """
    report = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict()
    }

    # Check for columns with high missing percentage
    report['high_missing_cols'] = [
        col for col, pct in report['missing_percentage'].items() if pct > 50
    ]

    # Numerical column statistics
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        report['numerical_stats'] = df[numerical_cols].describe().to_dict()

    # Categorical column value counts
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        report['categorical_value_counts'] = {
            col: df[col].value_counts().to_dict() for col in categorical_cols
        }

    return report

def prepare_data_for_training(df):
    """
    Complete data preparation pipeline for training

    Args:
        df: Raw DataFrame

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, imputer, scaler, feature_names)
    """
    X, y, imputer, scaler, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test, imputer, scaler, feature_names
