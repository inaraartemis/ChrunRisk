# 📊 Customer Churn Prediction Platform

🚀 **Live Demo:** [https://churn-prediction-api-5kwg.onrender.com/](https://churn-prediction-api-5kwg.onrender.com/)

## Problem Statement
Customer churn is a critical metric for telecommunications companies. Predicting which customers are likely to leave allows businesses to take proactive measures (e.g., discounts, personalized offers) to retain them. This platform provides a machine learning-based solution to predict churn risk levels and understand the key drivers behind customer decisions.

## Architecture
The platform is built with a modular architecture:
- **Data Layer**: Telco Customer Churn dataset (IBM).
- **Model Component**: Scikit-Learn pipeline with One-Hot Encoding, Scaling, and Random Forest Classifier.
- **Backend (FastAPI)**: RESTful API for single and batch predictions.
- **Frontend (Custom UI)**: Modern HTML/CSS/JS dashboard with Glassmorphism, animations, and Chart.js visualizations.

## Model Performance
| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression (Baseline) | 80.55% | 65.72% | 55.88% | 0.8420 |
| Random Forest (Final) | 78.21% | 61.28% | 48.66% | 0.8183 |

*Note: While Logistic Regression showed slightly higher metrics, Random Forest was chosen for its robust feature importance insights.*

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python model/train.py
```

### 3. Start the Backend API
```bash
python backend/main.py
```

### 4. Launch the Frontend UI
```bash
python frontend/app.py
```
<img width="1918" height="922" alt="image" src="https://github.com/user-attachments/assets/b29d99e2-910f-4345-bde5-b3856792538b" />

## Folder Structure
- `data/`: Dataset storage.
- `model/`: Training scripts and saved model files.
- `backend/`: FastAPI application.
- `frontend/`: Gradio interface.
- `requirements.txt`: Project dependencies.
- `README.md`: Documentation.

## Features
- **Individual Prediction**: Input customer details to get a risk gauge and probability.
- **Batch Processing**: Upload CSV files for bulk churn analysis.
- **Insights**: Visualize feature importance to understand churn drivers.
- **Risk Categorization**:
  - 🟢 **Low Risk**: < 30%
  - 🟡 **Medium Risk**: 30% - 60%
  - 🔴 **High Risk**: > 60%
