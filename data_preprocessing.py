import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Convert date columns to datetime
    date_columns = ['Dateofjoining', 'LastWorkingDate', 'MMM-YY']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%y', errors='coerce')

    # Create target variable
    df['target'] = np.where(df['LastWorkingDate'].notna(), 1, 0)

    # Feature engineering
    df['tenure'] = (df['LastWorkingDate'].fillna(pd.Timestamp.now()) - df['Dateofjoining']).dt.days
    df['rating_increased'] = df.groupby('Driver_ID')['Quarterly Rating'].diff() > 0
    df['income_increased'] = df.groupby('Driver_ID')['Income'].diff() > 0

    # Select relevant features
    features = ['Age', 'Gender', 'City', 'Education_Level', 'Income', 'tenure', 
                'Joining Designation', 'Grade', 'Quarterly Rating', 
                'rating_increased', 'income_increased']
    
    X = df[features]
    y = df['target']

    # Encode categorical variables
    X = pd.get_dummies(X, columns=['City', 'Education_Level', 'Joining Designation', 'Grade'], drop_first=True)

    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['Age', 'Income', 'tenure', 'Quarterly Rating']
    X_imputed[numerical_columns] = scaler.fit_transform(X_imputed[numerical_columns])

    return X_imputed, y, imputer, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preprocess_user_input(user_input, df_columns, imputer, scaler):
    # Initialize a dictionary to hold the processed input
    processed_input = {col: 0 for col in df_columns}

    # Process categorical variables
    categorical_columns = ['City', 'Education_Level', 'Joining Designation', 'Grade']
    for col in categorical_columns:
        if col in user_input:
            processed_input[user_input[col]] = 1

    # Process numerical and binary variables
    for col in ['Age', 'Gender', 'Income', 'tenure', 'Quarterly Rating', 'rating_increased', 'income_increased']:
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
    numerical_columns = ['Age', 'Income', 'tenure', 'Quarterly Rating']
    user_df_imputed[numerical_columns] = scaler.transform(user_df_imputed[numerical_columns])

    return user_df_imputed