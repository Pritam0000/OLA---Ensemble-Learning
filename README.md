# 🚗 OLA Driver Attrition Prediction System

## BTech Final Year Project - Machine Learning Application

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for predicting driver attrition in ride-sharing platforms using ensemble learning techniques with an interactive Streamlit interface.

![Project Banner](https://via.placeholder.com/1200x300/1f77b4/ffffff?text=OLA+Driver+Attrition+Prediction)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project addresses the critical business problem of driver attrition in ride-sharing companies like OLA. Using advanced machine learning techniques, the system:

- **Predicts** driver attrition with high accuracy (ROC-AUC > 0.85)
- **Identifies** key factors contributing to attrition
- **Provides** actionable insights for HR and management
- **Demonstrates** production-ready ML engineering practices

### Problem Statement

Driver attrition costs ride-sharing companies millions in recruitment and training. This system helps identify at-risk drivers early, enabling proactive retention strategies.

---

## ✨ Features

### 📊 Data Management
- **Multi-dataset Support**: Upload and manage multiple CSV datasets
- **Dataset Merging**: Intelligently merge datasets with deduplication
- **Data Quality Reports**: Comprehensive data quality analysis
- **Version Tracking**: Track all uploaded and merged datasets

### 🤖 Advanced Machine Learning
- **Multiple Models**: Random Forest, Gradient Boosting, Extra Trees, XGBoost
- **Hyperparameter Tuning**: Grid Search and Randomized Search CV
- **Model Versioning**: Store and compare all trained models
- **Performance Tracking**: Monitor model performance over time
- **Cross-Validation**: Robust model evaluation

### 📈 Data Visualization
- **Exploratory Data Analysis**: Interactive visualizations
- **Feature Distributions**: Histograms with KDE
- **Correlation Analysis**: Heatmaps for feature relationships
- **Target Analysis**: Class distribution and imbalance metrics

### 🔮 Predictions
- **Single Predictions**: Interactive form for individual driver predictions
- **Batch Predictions**: Upload CSV for bulk predictions
- **Confidence Scores**: High/Medium/Low confidence levels
- **Download Reports**: Export predictions as CSV

### 📊 Model Comparison
- **Side-by-side Comparison**: Compare multiple models
- **Performance Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visual Comparisons**: Bar charts for all metrics
- **Best Model Selection**: Automatically identify top performer

### 🎨 Visualizations
- **Confusion Matrix Heatmaps**
- **ROC Curves**
- **Precision-Recall Curves**
- **Feature Importance Plots**
- **Learning Curves**
- **Training History Graphs**

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+**: Programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization

### Machine Learning
- **Random Forest**: Ensemble decision trees
- **Gradient Boosting**: Boosted ensemble method
- **XGBoost**: Extreme gradient boosting
- **Extra Trees**: Extremely randomized trees
- **SMOTE**: Synthetic minority over-sampling

### Storage & Persistence
- **JSON**: Metadata storage
- **Joblib**: Model serialization
- **File System**: Organized directory structure

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/OLA---Ensemble-Learning.git
cd OLA---Ensemble-Learning
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import streamlit; import sklearn; import xgboost; print('All dependencies installed successfully!')"
```

---

## 💻 Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

### Quick Start Guide

#### 1. Upload Data
- Navigate to **📊 Data Management**
- Click **Upload Dataset** tab
- Upload your CSV file (sample: `ola_driver.csv`)
- Provide a dataset name (optional)

#### 2. Explore Data
- Go to **📈 Data Visualization**
- Select your dataset
- Explore distributions, correlations, and statistics

#### 3. Train Model
- Navigate to **🤖 Model Training**
- Select dataset
- Choose model type (Random Forest, Gradient Boosting, etc.)
- Configure hyperparameters
- Click **Train Model**
- Save the trained model

#### 4. Make Predictions
- Go to **🔮 Single Prediction** or **📁 Batch Prediction**
- Select a trained model
- Input driver information or upload CSV
- Get predictions with confidence scores

#### 5. Compare Models
- Navigate to **📊 Model Comparison**
- Select multiple models
- View side-by-side performance comparison

---

## 📁 Project Structure

```
OLA---Ensemble-Learning/
│
├── main.py                      # Main Streamlit application
├── config.py                    # Configuration and constants
├── data_preprocessing.py        # Data preprocessing functions
├── model_training.py            # Model training and evaluation
├── model_manager.py             # Model versioning and storage
├── dataset_manager.py           # Dataset management
├── utils.py                     # Utility functions and visualizations
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory
│   ├── raw/                     # Original datasets
│   ├── processed/               # Processed datasets
│   └── uploads/                 # User-uploaded datasets
│
├── models/                      # Trained models
│   ├── model_metadata.json      # Model metadata
│   └── *.joblib                 # Serialized models
│
├── artifacts/                   # Preprocessors (imputers, scalers)
│   └── *.joblib
│
└── reports/                     # Generated reports
    └── *.csv
```

---

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
```python
- Load CSV data
- Handle missing values (KNN Imputation)
- Feature engineering (tenure, rating trends, income trends)
- Categorical encoding (One-Hot)
- Numerical scaling (StandardScaler)
- Train-test split (80-20, stratified)
```

### 2. Class Imbalance Handling
```python
- Detect class imbalance
- Apply SMOTE (Synthetic Minority Over-sampling Technique)
- Balance training data
```

### 3. Model Training
```python
- Train multiple ensemble models
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Cross-validation (5-fold)
- Model evaluation on test set
```

### 4. Model Evaluation
```python
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Feature Importance
- Learning Curves
```

### 5. Model Versioning
```python
- Save model with timestamp
- Store metadata (hyperparameters, metrics, dataset info)
- Track all models for comparison
```

### 6. Prediction
```python
- Load trained model and preprocessors
- Preprocess input data
- Generate predictions
- Calculate confidence scores
```

---

## 📊 Dataset

### Expected Format

CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Driver_ID | int | Unique driver identifier |
| Age | int | Driver age |
| Gender | int | Gender (0=Male, 1=Female) |
| City | str | City code (C1, C2, ...) |
| Education_Level | int | Education level (0-3) |
| Income | float | Monthly income |
| Dateofjoining | date | Joining date (DD/MM/YY) |
| LastWorkingDate | date | Last working date (if attrited) |
| Joining Designation | int | Designation at joining |
| Grade | int | Current grade |
| Quarterly Rating | int | Performance rating (1-5) |
| Total Business Value | float | Total business generated |
| MMM-YY | date | Month-year |

### Sample Data

A sample dataset `ola_driver.csv` is included in the repository.

---

## 📸 Screenshots

### Home Page
![Home Page](https://via.placeholder.com/800x500/1f77b4/ffffff?text=Home+Page)

### Data Visualization
![Data Visualization](https://via.placeholder.com/800x500/2ca02c/ffffff?text=Data+Visualization)

### Model Training
![Model Training](https://via.placeholder.com/800x500/ff7f0e/ffffff?text=Model+Training)

### Model Comparison
![Model Comparison](https://via.placeholder.com/800x500/d62728/ffffff?text=Model+Comparison)

### Predictions
![Predictions](https://via.placeholder.com/800x500/9467bd/ffffff?text=Predictions)

---

## 🎓 Academic Context

### BTech Project Details

- **Course**: Bachelor of Technology (B.Tech)
- **Specialization**: Computer Science / Data Science / AI & ML
- **Project Type**: Final Year Major Project
- **Domain**: Machine Learning, Predictive Analytics
- **Application Area**: Human Resource Management

### Key Learning Outcomes

1. **Machine Learning**: Ensemble methods, hyperparameter tuning, cross-validation
2. **Data Engineering**: ETL pipelines, data quality, preprocessing
3. **Software Engineering**: Modular design, version control, deployment
4. **Web Development**: Interactive dashboards, full-stack application
5. **Project Management**: Documentation, testing, presentation

### Presentation Tips

1. **Start with the Problem**: Explain driver attrition impact
2. **Demo the System**: Show live predictions
3. **Highlight Features**: Dataset management, model comparison, visualizations
4. **Show Technical Depth**: Explain SMOTE, ensemble methods, hyperparameter tuning
5. **Business Value**: Discuss ROI and retention strategies

---

## 🚀 Deployment

### Local Deployment

Already covered in [Usage](#usage) section.

### Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Sign in with GitHub
4. Select your repository
5. Set `main.py` as the main file
6. Deploy!

### Heroku Deployment

1. Create `Procfile`:
```
web: streamlit run main.py --server.port $PORT
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy to Heroku

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OLA**: For the inspiration and domain
- **scikit-learn**: For excellent ML library
- **Streamlit**: For the amazing web framework
- **SMOTE**: Chawla et al. for the technique
- **College Faculty**: For guidance and support

---

## 📧 Contact

**Project Maintainer**: [Your Name]

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Project Link**: [https://github.com/yourusername/OLA---Ensemble-Learning](https://github.com/yourusername/OLA---Ensemble-Learning)

---

## 📚 References

1. Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." JAIR, 2002.
2. Breiman, L. "Random Forests." Machine Learning, 2001.
3. Friedman, J. H. "Greedy Function Approximation: A Gradient Boosting Machine." 1999.
4. Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD, 2016.
5. scikit-learn Documentation: https://scikit-learn.org/
6. Streamlit Documentation: https://docs.streamlit.io/

---

## ⭐ Star History

If you find this project helpful, please give it a star! ⭐

---

<div align="center">

**Made with ❤️ for BTech Final Year Project**

**© 2025 OLA Driver Attrition Prediction System**

</div>
