import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data, split_data, preprocess_user_input
from model_training import train_random_forest, train_gradient_boosting, evaluate_model, handle_class_imbalance, save_model, load_model
from utils import plot_feature_importance, plot_roc_curve, plot_correlation_matrix

@st.cache_data
def load_and_preprocess():
    df = load_data("ola_driver.csv")
    X, y, imputer, scaler = preprocess_data(df)
    return df, X, y, imputer, scaler

def main():
    st.set_page_config(layout="wide")
    st.title("Ola Driver Attrition Prediction")

    df, X, y, imputer, scaler = load_and_preprocess()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Visualization", "Model Training", "Prediction"])

    if page == "Data Visualization":
        data_visualization(df, X, y)
    elif page == "Model Training":
        model_training(X, y)
    else:
        prediction(X, y, imputer, scaler)

def data_visualization(df, X, y):
    st.header("Data Visualization")

    # Basic dataset information
    st.subheader("Dataset Overview")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of features: {X.shape[1]}")

    # Display first few rows
    st.subheader("Sample Data")
    st.write(df.head())

    # Univariate Analysis
    st.subheader("Univariate Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(X['Age'], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(X['Income'], kde=True, ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.write("Education Level Distribution")
        fig, ax = plt.subplots()
        X.filter(regex='^Education_Level_').sum().sort_values(ascending=False).plot(kind='bar', ax=ax)
        plt.title("Education Level Distribution")
        plt.xlabel("Education Level")
        plt.ylabel("Count")
        st.pyplot(fig)

    with col4:
        st.write("Target Distribution")
        fig, ax = plt.subplots()
        y.value_counts().plot(kind='bar', ax=ax)
        plt.title("Target Distribution")
        plt.xlabel("Target")
        plt.ylabel("Count")
        st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    selected_features = ['Age', 'Income', 'tenure', 'Quarterly Rating']
    X_selected = X[selected_features].copy()
    X_selected['target'] = y
    corr_matrix = plot_correlation_matrix(X_selected)
    st.pyplot(corr_matrix)

def model_training(X, y):
    st.header("Model Training")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Display class distribution
    st.subheader("Class Distribution")
    st.write("Training set:")
    st.write(pd.Series(y_train).value_counts(normalize=True))
    st.write("Test set:")
    st.write(pd.Series(y_test).value_counts(normalize=True))

    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X_train, y_train)

    st.write("Resampled training set:")
    st.write(pd.Series(y_resampled).value_counts(normalize=True))

    # Model selection
    model_option = st.selectbox(
        "Choose a model",
        ("Random Forest", "Gradient Boosting")
    )

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if model_option == "Random Forest":
                model = train_random_forest(X_resampled, y_resampled)
            else:
                model = train_gradient_boosting(X_resampled, y_resampled)
            
            save_model(model, f"{model_option.lower().replace(' ', '_')}_model.joblib")
            st.success("Model trained and saved successfully!")

        # Evaluate model
        st.subheader("Model Evaluation")
        evaluate_model(model, X_test, y_test)

        # Plot feature importance
        st.subheader("Feature Importance")
        fig_importance = plot_feature_importance(model, X)
        st.pyplot(fig_importance)

        # Plot ROC curve
        st.subheader("ROC Curve")
        y_prob = model.predict_proba(X_test)[:, 1]
        fig_roc = plot_roc_curve(y_test, y_prob)
        st.pyplot(fig_roc)

        # Display some predictions
        st.subheader("Sample Predictions")
        sample_size = min(10, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        sample_X = X_test.iloc[sample_indices]
        sample_y = y_test.iloc[sample_indices]
        sample_pred = model.predict(sample_X)
        sample_prob = model.predict_proba(sample_X)[:, 1]

        results_df = pd.DataFrame({
            'Actual': sample_y,
            'Predicted': sample_pred,
            'Probability': sample_prob
        })
        st.write(results_df)

def prediction(X, y, imputer, scaler):
    st.header("Attrition Prediction")

    # User input form
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.selectbox("City", X.filter(regex='^City_').columns)
    education_level = st.selectbox("Education Level", X.filter(regex='^Education_Level_').columns)
    income = st.number_input("Monthly Income", min_value=0, value=50000)
    joining_designation = st.selectbox("Joining Designation", X.filter(regex='^Joining Designation_').columns)
    grade = st.selectbox("Current Grade", X.filter(regex='^Grade_').columns)
    quarterly_rating = st.slider("Quarterly Rating", 1, 5, 3)
    tenure = st.number_input("Tenure (in days)", min_value=0, value=365)
    rating_increased = st.checkbox("Rating Increased")
    income_increased = st.checkbox("Income Increased")

    if st.button("Predict Attrition"):
        # Prepare user input
        user_input = {
            'Age': age,
            'Gender': 1 if gender == "Female" else 0,
            'Income': income,
            'tenure': tenure,
            'Quarterly Rating': quarterly_rating,
            'rating_increased': 1 if rating_increased else 0,
            'income_increased': 1 if income_increased else 0,
            city: 1,
            education_level: 1,
            joining_designation: 1,
            grade: 1
        }

        # Preprocess user input
        user_df = preprocess_user_input(user_input, X.columns, imputer, scaler)

        # Load the model and make prediction
        model = load_model("random_forest_model.joblib")  # You can change this to the desired model
        
        st.subheader("Preprocessed User Input")
        st.write(user_df)

        prediction = model.predict(user_df)
        probability = model.predict_proba(user_df)[0][1]

        st.subheader("Model Output")
        st.write(f"Raw prediction: {prediction}")
        st.write(f"Probability: {probability}")

        st.subheader("Final Prediction")
        st.write(f"Predicted Attrition: {'Yes' if prediction[0] == 1 else 'No'}")
        st.write(f"Attrition Probability: {probability:.2f}")

        # Display feature importances for this prediction
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        st.subheader("Feature Importances")
        st.write(feature_importance)

if __name__ == "__main__":
    main()