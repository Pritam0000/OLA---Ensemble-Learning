"""
OLA Driver Attrition Prediction System - Enhanced Version
Multi-page Streamlit application with comprehensive ML pipeline
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import json

# Import custom modules
from data_preprocessing import (
    load_data, preprocess_data, split_data, preprocess_user_input,
    preprocess_batch_input, get_data_quality_report, prepare_data_for_training
)
from model_training import (
    train_model, evaluate_model, handle_class_imbalance,
    hyperparameter_tuning, cross_validate_model
)
from model_manager import ModelManager
from dataset_manager import DatasetManager
from utils import *
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, RAW_DATA_DIR,
    REPORTS_DIR, AVAILABLE_MODELS, PARAM_GRIDS
)

# Initialize managers
model_manager = ModelManager()
dataset_manager = DatasetManager()

# Page configuration
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""

    # Sidebar navigation
    st.sidebar.title("🚗 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "📊 Data Management",
            "📈 Data Visualization",
            "🤖 Model Training",
            "⚙️ Hyperparameter Tuning",
            "📊 Model Comparison",
            "🔮 Single Prediction",
            "📁 Batch Prediction",
            "📚 Model History",
            "ℹ️ About"
        ]
    )

    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Data Management":
        data_management_page()
    elif page == "📈 Data Visualization":
        data_visualization_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "⚙️ Hyperparameter Tuning":
        hyperparameter_tuning_page()
    elif page == "📊 Model Comparison":
        model_comparison_page()
    elif page == "🔮 Single Prediction":
        single_prediction_page()
    elif page == "📁 Batch Prediction":
        batch_prediction_page()
    elif page == "📚 Model History":
        model_history_page()
    elif page == "ℹ️ About":
        about_page()

def home_page():
    """Home page with project overview"""
    st.markdown('<p class="main-header">🚗 OLA Driver Attrition Prediction System</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Enhanced OLA Driver Attrition Prediction System!

    This comprehensive machine learning application predicts driver attrition using ensemble learning techniques.

    #### 🎯 Key Features:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📊 Data Management**
        - Upload multiple datasets
        - Merge datasets intelligently
        - Data quality reports
        - Dataset versioning
        """)

    with col2:
        st.markdown("""
        **🤖 Advanced ML**
        - Multiple ensemble models
        - Hyperparameter tuning
        - Model versioning
        - Performance tracking
        """)

    with col3:
        st.markdown("""
        **🔮 Predictions**
        - Single predictions
        - Batch predictions
        - Download reports (CSV/PDF)
        - Confidence scores
        """)

    st.markdown("---")

    # Quick stats
    st.markdown("### 📈 System Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        datasets = dataset_manager.get_all_datasets()
        st.metric("Total Datasets", len(datasets))

    with col2:
        merged = dataset_manager.get_merged_datasets()
        st.metric("Merged Datasets", len(merged))

    with col3:
        models = model_manager.get_all_models()
        st.metric("Trained Models", len(models))

    with col4:
        best_model_id, best_model_info = model_manager.get_best_model('roc_auc')
        if best_model_info:
            st.metric("Best ROC-AUC", f"{best_model_info['metrics']['roc_auc']:.3f}")
        else:
            st.metric("Best ROC-AUC", "N/A")

    st.markdown("---")

    st.markdown("""
    ### 🚀 Getting Started
    1. **Upload Data**: Go to 'Data Management' to upload your driver data
    2. **Explore Data**: Visualize your data in 'Data Visualization'
    3. **Train Models**: Train and compare multiple models
    4. **Make Predictions**: Use trained models for predictions

    ### 🎓 BTech Project Information
    - **Project**: Driver Attrition Prediction using Ensemble Learning
    - **Techniques**: Random Forest, Gradient Boosting, XGBoost, Extra Trees
    - **Features**: SMOTE for class imbalance, hyperparameter tuning, model versioning
    """)

def data_management_page():
    """Data management page for uploading and merging datasets"""
    st.title("📊 Data Management")

    tab1, tab2, tab3 = st.tabs(["📤 Upload Dataset", "🔗 Merge Datasets", "📋 View Datasets"])

    with tab1:
        st.subheader("Upload New Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        dataset_name = st.text_input("Dataset Name (optional)")

        if uploaded_file:
            st.write("**Preview:**")
            df_preview = pd.read_csv(uploaded_file)
            st.write(df_preview.head())

            if st.button("Upload Dataset"):
                uploaded_file.seek(0)  # Reset file pointer
                dataset_id, df = dataset_manager.upload_dataset(uploaded_file, dataset_name)
                st.success(f"Dataset uploaded successfully! ID: {dataset_id}")
                st.write(f"Records: {len(df)}, Features: {len(df.columns)}")

    with tab2:
        st.subheader("Merge Datasets")
        datasets = dataset_manager.get_all_datasets()

        if len(datasets) < 2:
            st.warning("Upload at least 2 datasets to merge")
        else:
            dataset_options = {d['dataset_name']: d['dataset_id'] for d in datasets}
            selected_datasets = st.multiselect(
                "Select datasets to merge",
                options=list(dataset_options.keys())
            )

            merge_name = st.text_input("Merged Dataset Name")

            if st.button("Merge Datasets"):
                if len(selected_datasets) < 2:
                    st.error("Select at least 2 datasets to merge")
                else:
                    dataset_ids = [dataset_options[name] for name in selected_datasets]
                    merged_df, merged_id = dataset_manager.merge_datasets(dataset_ids, merge_name)
                    st.success(f"Datasets merged successfully! ID: {merged_id}")
                    st.write(f"Total records: {len(merged_df)}, Features: {len(merged_df.columns)}")
                    st.write(merged_df.head())

    with tab3:
        st.subheader("All Available Datasets")
        all_data_df = dataset_manager.get_all_available_data()

        if not all_data_df.empty:
            st.dataframe(all_data_df, use_container_width=True)

            # Data quality report
            st.subheader("Data Quality Report")
            selected_id = st.selectbox("Select dataset for quality report",
                                      options=all_data_df['id'].tolist())

            if st.button("Generate Quality Report"):
                try:
                    if selected_id.startswith('merged'):
                        df, _ = dataset_manager.load_merged_dataset(selected_id)
                    else:
                        df, _ = dataset_manager.load_dataset(selected_id)

                    quality_report = get_data_quality_report(df)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", quality_report['total_records'])
                    with col2:
                        st.metric("Total Features", quality_report['total_features'])
                    with col3:
                        st.metric("Duplicate Rows", quality_report['duplicate_rows'])

                    if quality_report['high_missing_cols']:
                        st.warning(f"High missing values (>50%) in: {', '.join(quality_report['high_missing_cols'])}")

                    st.write("**Missing Values:**")
                    missing_df = pd.DataFrame({
                        'Column': quality_report['missing_values'].keys(),
                        'Missing Count': quality_report['missing_values'].values(),
                        'Missing %': quality_report['missing_percentage'].values()
                    })
                    st.dataframe(missing_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        else:
            st.info("No datasets uploaded yet")

def data_visualization_page():
    """Data visualization and EDA page"""
    st.title("📈 Data Visualization & EDA")

    all_data_df = dataset_manager.get_all_available_data()

    if all_data_df.empty:
        st.warning("No datasets available. Please upload data first.")
        return

    selected_id = st.selectbox("Select dataset to visualize",
                              options=all_data_df['id'].tolist(),
                              format_func=lambda x: all_data_df[all_data_df['id']==x]['name'].values[0])

    try:
        if selected_id.startswith('merged'):
            df, _ = dataset_manager.load_merged_dataset(selected_id)
        else:
            df, _ = dataset_manager.load_dataset(selected_id)

        # Preprocess data
        X, y, imputer, scaler, feature_names = preprocess_data(df)

        # Basic information
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", df.shape[0])
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Target Classes", y.nunique())

        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "🎯 Target Analysis", "📈 Feature Analysis"])

        with tab1:
            st.subheader("Feature Distributions")
            col1, col2 = st.columns(2)

            with col1:
                if 'Age' in X.columns:
                    st.write("**Age Distribution**")
                    fig, ax = plt.subplots()
                    sns.histplot(X['Age'], kde=True, ax=ax, color='skyblue')
                    ax.set_title("Age Distribution")
                    st.pyplot(fig)

            with col2:
                if 'Income' in X.columns:
                    st.write("**Income Distribution**")
                    fig, ax = plt.subplots()
                    sns.histplot(X['Income'], kde=True, ax=ax, color='lightcoral')
                    ax.set_title("Income Distribution")
                    st.pyplot(fig)

            col3, col4 = st.columns(2)

            with col3:
                if 'tenure' in X.columns:
                    st.write("**Tenure Distribution**")
                    fig, ax = plt.subplots()
                    sns.histplot(X['tenure'], kde=True, ax=ax, color='lightgreen')
                    ax.set_title("Tenure Distribution")
                    st.pyplot(fig)

            with col4:
                if 'Quarterly Rating' in X.columns:
                    st.write("**Quarterly Rating Distribution**")
                    fig, ax = plt.subplots()
                    sns.histplot(X['Quarterly Rating'], kde=True, ax=ax, color='plum')
                    ax.set_title("Quarterly Rating Distribution")
                    st.pyplot(fig)

        with tab2:
            st.subheader("Correlation Matrix")
            numerical_features = ['Age', 'Income', 'tenure', 'Quarterly Rating']
            available_features = [f for f in numerical_features if f in X.columns]

            if available_features:
                X_corr = X[available_features].copy()
                X_corr['target'] = y.values
                fig = plot_correlation_matrix(X_corr)
                st.pyplot(fig)
            else:
                st.warning("No numerical features available for correlation matrix")

        with tab3:
            st.subheader("Target Variable Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Target Distribution**")
                fig, ax = plt.subplots()
                y.value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                ax.set_title("Target Distribution")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                ax.set_xticklabels(['No Attrition', 'Attrition'], rotation=0)
                st.pyplot(fig)

            with col2:
                st.write("**Class Balance**")
                class_dist = y.value_counts(normalize=True)
                st.write(f"No Attrition (0): {class_dist[0]:.2%}")
                st.write(f"Attrition (1): {class_dist[1]:.2%}")
                st.write(f"Imbalance Ratio: {class_dist[0]/class_dist[1]:.2f}:1")

        with tab4:
            st.subheader("Feature Statistics")
            st.dataframe(X.describe().T, use_container_width=True)

    except Exception as e:
        st.error(f"Error visualizing data: {str(e)}")

def model_training_page():
    """Model training page"""
    st.title("🤖 Model Training")

    all_data_df = dataset_manager.get_all_available_data()

    if all_data_df.empty:
        st.warning("No datasets available. Please upload data first.")
        return

    # Select dataset
    selected_id = st.selectbox("Select dataset for training",
                              options=all_data_df['id'].tolist(),
                              format_func=lambda x: all_data_df[all_data_df['id']==x]['name'].values[0])

    try:
        if selected_id.startswith('merged'):
            df, _ = dataset_manager.load_merged_dataset(selected_id)
        else:
            df, _ = dataset_manager.load_dataset(selected_id)

        # Preprocessing
        X_train, X_test, y_train, y_test, imputer, scaler, feature_names = prepare_data_for_training(df)

        # Display class distribution
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Training Set:**")
            st.write(pd.Series(y_train).value_counts(normalize=True))

        with col2:
            st.write("**Test Set:**")
            st.write(pd.Series(y_test).value_counts(normalize=True))

        # Handle class imbalance
        use_smote = st.checkbox("Use SMOTE for class imbalance", value=True)

        if use_smote:
            X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
            st.success(f"SMOTE applied: {len(X_train)} → {len(X_train_resampled)} samples")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Model selection
        st.subheader("Model Configuration")

        available_model_types = list(AVAILABLE_MODELS.keys())
        if 'XGBoost' in available_model_types:
            try:
                from xgboost import XGBClassifier
            except ImportError:
                available_model_types.remove('XGBoost')
                st.info("XGBoost not installed. Install with: pip install xgboost")

        model_type = st.selectbox("Choose model type", available_model_types)

        model_name = st.text_input("Model Name", value=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}")

        # Basic hyperparameters
        with st.expander("Configure Basic Hyperparameters"):
            n_estimators = st.slider("Number of Estimators", 50, 500, 100, 50)
            random_state = st.number_input("Random State", value=42)

            hyperparams = {
                'n_estimators': n_estimators,
                'random_state': random_state
            }

        # Train model
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                model = train_model(model_type, X_train_resampled, y_train_resampled, **hyperparams)

                st.success("Model trained successfully!")

                # Evaluate model
                st.subheader("Model Evaluation")
                metrics = evaluate_model(model, X_test, y_test, return_metrics=True)

                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                col2.metric("Precision", f"{metrics['precision']:.4f}")
                col3.metric("Recall", f"{metrics['recall']:.4f}")
                col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

                # Visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Confusion Matrix", "📈 ROC Curve", "📊 Feature Importance", "🔄 PR Curve"])

                with tab1:
                    y_pred = model.predict(X_test)
                    fig_cm = plot_confusion_matrix_heatmap(y_test, y_pred, model_name)
                    st.pyplot(fig_cm)

                with tab2:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fig_roc = plot_roc_curve(y_test, y_prob, model_name)
                    st.pyplot(fig_roc)

                with tab3:
                    X_full = pd.concat([X_train, X_test])
                    fig_importance = plot_feature_importance(model, X_full, top_n=20)
                    st.pyplot(fig_importance)

                with tab4:
                    fig_pr = plot_precision_recall_curve(y_test, y_prob, model_name)
                    st.pyplot(fig_pr)

                # Save model
                if st.button("Save Model"):
                    preprocessors = {'imputer': imputer, 'scaler': scaler}
                    dataset_info = {
                        'dataset_id': selected_id,
                        'num_records': len(df),
                        'num_features': len(feature_names),
                        'feature_names': feature_names
                    }

                    model_id = model_manager.save_model(
                        model, model_name, model_type, metrics,
                        hyperparameters=hyperparams,
                        dataset_info=dataset_info,
                        preprocessors=preprocessors
                    )

                    st.success(f"Model saved with ID: {model_id}")

    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def hyperparameter_tuning_page():
    """Hyperparameter tuning page"""
    st.title("⚙️ Hyperparameter Tuning")

    st.markdown("""
    Optimize model performance through Grid Search or Randomized Search.
    """)

    all_data_df = dataset_manager.get_all_available_data()

    if all_data_df.empty:
        st.warning("No datasets available. Please upload data first.")
        return

    # Select dataset
    selected_id = st.selectbox("Select dataset",
                              options=all_data_df['id'].tolist(),
                              format_func=lambda x: all_data_df[all_data_df['id']==x]['name'].values[0])

    try:
        if selected_id.startswith('merged'):
            df, _ = dataset_manager.load_merged_dataset(selected_id)
        else:
            df, _ = dataset_manager.load_dataset(selected_id)

        X_train, X_test, y_train, y_test, imputer, scaler, feature_names = prepare_data_for_training(df)

        # Apply SMOTE
        X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

        # Model selection
        available_model_types = list(PARAM_GRIDS.keys())
        model_type = st.selectbox("Choose model type", available_model_types)

        # Search type
        search_type = st.radio("Search Strategy", ['grid', 'random'])

        if search_type == 'random':
            n_iter = st.slider("Number of iterations", 10, 100, 20)
        else:
            n_iter = 20

        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)

        # Display parameter grid
        st.subheader("Parameter Grid")
        st.json(PARAM_GRIDS[model_type])

        model_name = st.text_input("Model Name", value=f"{model_type}_tuned_{datetime.now().strftime('%Y%m%d_%H%M')}")

        # Start tuning
        if st.button("Start Hyperparameter Tuning", type="primary"):
            with st.spinner(f"Performing {search_type} search... This may take a while..."):
                best_model, best_params, cv_results = hyperparameter_tuning(
                    model_type, X_train_resampled, y_train_resampled,
                    search_type=search_type, cv=cv_folds, n_iter=n_iter
                )

                st.success("Tuning completed!")

                # Display best parameters
                st.subheader("Best Parameters")
                st.json(best_params)

                # Evaluate best model
                st.subheader("Best Model Evaluation")
                metrics = evaluate_model(best_model, X_test, y_test, return_metrics=True)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                col2.metric("Precision", f"{metrics['precision']:.4f}")
                col3.metric("Recall", f"{metrics['recall']:.4f}")
                col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

                # Visualizations
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]

                col1, col2 = st.columns(2)

                with col1:
                    fig_cm = plot_confusion_matrix_heatmap(y_test, y_pred, model_name)
                    st.pyplot(fig_cm)

                with col2:
                    fig_roc = plot_roc_curve(y_test, y_prob, model_name)
                    st.pyplot(fig_roc)

                # Save model
                if st.button("Save Tuned Model"):
                    preprocessors = {'imputer': imputer, 'scaler': scaler}
                    dataset_info = {
                        'dataset_id': selected_id,
                        'num_records': len(df),
                        'num_features': len(feature_names),
                        'feature_names': feature_names
                    }

                    model_id = model_manager.save_model(
                        best_model, model_name, model_type, metrics,
                        hyperparameters=best_params,
                        dataset_info=dataset_info,
                        preprocessors=preprocessors
                    )

                    st.success(f"Tuned model saved with ID: {model_id}")

    except Exception as e:
        st.error(f"Error during tuning: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def model_comparison_page():
    """Model comparison page"""
    st.title("📊 Model Comparison")

    models_metadata = model_manager.get_all_models()

    if not models_metadata:
        st.warning("No trained models available. Train models first.")
        return

    st.subheader("All Trained Models")
    models_df = model_manager.get_models_dataframe()
    st.dataframe(models_df, use_container_width=True)

    # Select models to compare
    st.subheader("Compare Models")
    model_ids = list(models_metadata.keys())
    selected_models = st.multiselect(
        "Select models to compare",
        options=model_ids,
        default=model_ids[:min(3, len(model_ids))]
    )

    if selected_models:
        comparison_df = model_manager.get_model_comparison(selected_models)

        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True)

        # Visualization
        fig_comparison = plot_model_comparison(comparison_df)
        st.pyplot(fig_comparison)

        # Best model
        best_idx = comparison_df['ROC-AUC'].idxmax()
        best_model_name = comparison_df.loc[best_idx, 'Model']
        best_roc_auc = comparison_df.loc[best_idx, 'ROC-AUC']

        st.success(f"🏆 Best Model: **{best_model_name}** with ROC-AUC of **{best_roc_auc:.4f}**")

        # Detailed comparison
        with st.expander("Detailed Model Information"):
            for model_id in selected_models:
                model_info = models_metadata[model_id]
                st.markdown(f"### {model_info['model_name']}")
                st.write(f"**Type:** {model_info['model_type']}")
                st.write(f"**Timestamp:** {model_info['timestamp']}")
                st.write(f"**Hyperparameters:**")
                st.json(model_info['hyperparameters'])

def single_prediction_page():
    """Single prediction page"""
    st.title("🔮 Single Driver Attrition Prediction")

    models_metadata = model_manager.get_all_models()

    if not models_metadata:
        st.warning("No trained models available. Train a model first.")
        return

    # Select model
    model_ids = list(models_metadata.keys())
    model_names = {mid: models_metadata[mid]['model_name'] for mid in model_ids}

    selected_model_id = st.selectbox(
        "Select Model",
        options=model_ids,
        format_func=lambda x: model_names[x]
    )

    # Load model and preprocessors
    model, preprocessors, model_info = model_manager.load_model(selected_model_id)

    st.write(f"**Model Type:** {model_info['model_type']}")
    st.write(f"**ROC-AUC:** {model_info['metrics']['roc_auc']:.4f}")

    st.markdown("---")

    # Input form
    st.subheader("Enter Driver Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Monthly Income", min_value=0, value=50000)

    with col2:
        tenure = st.number_input("Tenure (in days)", min_value=0, value=365)
        quarterly_rating = st.slider("Quarterly Rating", 1, 5, 3)

    with col3:
        rating_increased = st.checkbox("Rating Increased")
        income_increased = st.checkbox("Income Increased")

    # Additional inputs (simplified)
    city = st.text_input("City (e.g., C1, C2)", value="C1")
    education_level = st.selectbox("Education Level", ["0", "1", "2", "3"])
    joining_designation = st.selectbox("Joining Designation", ["1", "2", "3"])
    grade = st.selectbox("Grade", ["1", "2", "3"])

    if st.button("Predict Attrition", type="primary"):
        try:
            # Prepare input
            feature_names = model_info['dataset_info']['feature_names']

            user_input = {
                'Age': age,
                'Gender': 1 if gender == "Female" else 0,
                'Income': income,
                'tenure': tenure,
                'Quarterly Rating': quarterly_rating,
                'rating_increased': 1 if rating_increased else 0,
                'income_increased': 1 if income_increased else 0
            }

            # Create dataframe with all required features
            input_df = pd.DataFrame([user_input])
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            input_df = input_df[feature_names]

            # Apply preprocessing if available
            if preprocessors:
                imputer = preprocessors['imputer']
                scaler = preprocessors['scaler']

                # Impute
                input_processed = pd.DataFrame(
                    imputer.transform(input_df),
                    columns=input_df.columns
                )

                # Scale numerical features
                numerical_cols = ['Age', 'Income', 'tenure', 'Quarterly Rating']
                numerical_cols = [c for c in numerical_cols if c in input_processed.columns]
                if numerical_cols:
                    input_processed[numerical_cols] = scaler.transform(input_processed[numerical_cols])
            else:
                input_processed = input_df

            # Make prediction
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1]

            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("⚠️ **High Risk of Attrition**")
                else:
                    st.success("✅ **Low Risk of Attrition**")

            with col2:
                st.metric("Attrition Probability", f"{probability:.2%}")

            # Confidence level
            if probability > 0.7 or probability < 0.3:
                confidence = "High"
                color = "green"
            elif probability > 0.6 or probability < 0.4:
                confidence = "Medium"
                color = "orange"
            else:
                confidence = "Low"
                color = "red"

            st.markdown(f"**Confidence Level:** :{color}[{confidence}]")

            # Recommendations
            st.subheader("Recommendations")
            if prediction == 1:
                st.markdown("""
                - **Immediate Action Required**
                - Schedule retention interview
                - Review compensation and benefits
                - Assess work-life balance
                - Provide career development opportunities
                """)
            else:
                st.markdown("""
                - **Regular Monitoring**
                - Maintain current engagement level
                - Continue performance recognition
                - Ensure career growth path
                """)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def batch_prediction_page():
    """Batch prediction page"""
    st.title("📁 Batch Prediction")

    st.markdown("""
    Upload a CSV file with multiple driver records to get predictions for all at once.
    """)

    models_metadata = model_manager.get_all_models()

    if not models_metadata:
        st.warning("No trained models available. Train a model first.")
        return

    # Select model
    model_ids = list(models_metadata.keys())
    model_names = {mid: models_metadata[mid]['model_name'] for mid in model_ids}

    selected_model_id = st.selectbox(
        "Select Model",
        options=model_ids,
        format_func=lambda x: model_names[x]
    )

    # Upload file
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("**Preview of uploaded data:**")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Generate Predictions", type="primary"):
            try:
                # Load model
                model, preprocessors, model_info = model_manager.load_model(selected_model_id)

                # Prepare batch input
                feature_names = model_info['dataset_info']['feature_names']

                # Ensure all required features exist
                for feature in feature_names:
                    if feature not in batch_df.columns:
                        batch_df[feature] = 0

                batch_input = batch_df[feature_names]

                # Apply preprocessing
                if preprocessors:
                    imputer = preprocessors['imputer']
                    scaler = preprocessors['scaler']

                    batch_processed = pd.DataFrame(
                        imputer.transform(batch_input),
                        columns=batch_input.columns
                    )

                    numerical_cols = ['Age', 'Income', 'tenure', 'Quarterly Rating']
                    numerical_cols = [c for c in numerical_cols if c in batch_processed.columns]
                    if numerical_cols:
                        batch_processed[numerical_cols] = scaler.transform(batch_processed[numerical_cols])
                else:
                    batch_processed = batch_input

                # Make predictions
                predictions = model.predict(batch_processed)
                probabilities = model.predict_proba(batch_processed)[:, 1]

                # Create results dataframe
                results_df = batch_df.copy()
                results_df['Prediction'] = ['Attrition' if p == 1 else 'No Attrition' for p in predictions]
                results_df['Attrition_Probability'] = probabilities
                results_df['Risk_Level'] = pd.cut(
                    probabilities,
                    bins=[0, 0.3, 0.7, 1.0],
                    labels=['Low', 'Medium', 'High']
                )

                st.success(f"Predictions generated for {len(results_df)} records!")

                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results_df, use_container_width=True)

                # Summary statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    attrition_count = (predictions == 1).sum()
                    st.metric("Predicted Attrition", attrition_count)

                with col2:
                    no_attrition_count = (predictions == 0).sum()
                    st.metric("Predicted No Attrition", no_attrition_count)

                with col3:
                    avg_probability = probabilities.mean()
                    st.metric("Avg Attrition Probability", f"{avg_probability:.2%}")

                # Download options
                st.subheader("Download Results")

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def model_history_page():
    """Model training history page"""
    st.title("📚 Model Training History")

    models_metadata = model_manager.get_all_models()

    if not models_metadata:
        st.warning("No trained models available.")
        return

    # Get training history
    history_df = model_manager.get_training_history()

    if not history_df.empty:
        # Performance over time
        st.subheader("Model Performance Over Time")
        fig_history = plot_training_history(history_df)
        st.pyplot(fig_history)

        # Best model
        best_model_id, best_model_info = model_manager.get_best_model('roc_auc')

        if best_model_info:
            st.subheader("🏆 Best Performing Model")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model Name", best_model_info['model_name'])
            with col2:
                st.metric("Model Type", best_model_info['model_type'])
            with col3:
                st.metric("ROC-AUC", f"{best_model_info['metrics']['roc_auc']:.4f}")
            with col4:
                st.metric("F1-Score", f"{best_model_info['metrics']['f1_score']:.4f}")

        # Model list
        st.subheader("All Models")
        models_df = model_manager.get_models_dataframe()
        st.dataframe(models_df, use_container_width=True)

        # Export functionality
        st.subheader("Export Model Information")
        export_model_id = st.selectbox("Select model to export", options=list(models_metadata.keys()))

        if st.button("Export Model Info as JSON"):
            model_info = models_metadata[export_model_id]
            json_str = json.dumps(model_info, indent=4, default=str)

            st.download_button(
                label="📥 Download Model Info",
                data=json_str,
                file_name=f"model_info_{export_model_id}.json",
                mime="application/json"
            )

def about_page():
    """About page"""
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 🚗 OLA Driver Attrition Prediction System

    ### 🎓 BTech Final Year Project

    This is a comprehensive machine learning system for predicting driver attrition in ride-sharing platforms like OLA.

    ### 🎯 Project Objectives
    - Predict driver attrition with high accuracy
    - Identify key factors contributing to attrition
    - Provide actionable insights for retention strategies
    - Demonstrate advanced ML techniques and software engineering practices

    ### 🔧 Technologies Used
    - **Language:** Python 3.x
    - **ML Libraries:** scikit-learn, XGBoost, imbalanced-learn
    - **Web Framework:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn

    ### 🤖 Machine Learning Techniques
    1. **Ensemble Methods:**
       - Random Forest Classifier
       - Gradient Boosting Classifier
       - Extra Trees Classifier
       - XGBoost (optional)

    2. **Data Preprocessing:**
       - KNN Imputation for missing values
       - Standard Scaling for numerical features
       - One-Hot Encoding for categorical features

    3. **Class Imbalance Handling:**
       - SMOTE (Synthetic Minority Over-sampling Technique)

    4. **Model Optimization:**
       - Grid Search CV
       - Randomized Search CV
       - Cross-validation

    ### 📊 Key Features
    - ✅ Multi-dataset support with merging capability
    - ✅ Model versioning and comparison
    - ✅ Hyperparameter tuning interface
    - ✅ Single and batch predictions
    - ✅ Comprehensive visualizations
    - ✅ Export functionality (CSV, JSON)
    - ✅ Data quality reports

    ### 👥 Team Information
    - **Project Type:** BTech Final Year Project
    - **Domain:** Machine Learning, Predictive Analytics
    - **Application:** Human Resource Management, Employee Retention

    ### 📚 References
    - scikit-learn Documentation
    - Streamlit Documentation
    - Research papers on employee attrition prediction
    - SMOTE paper by Chawla et al.

    ### 📧 Contact
    For queries and support, please refer to the project repository.

    ---

    **Version:** 2.0 Enhanced
    **Last Updated:** 2025
    **Status:** Production Ready
    """)

if __name__ == "__main__":
    main()
