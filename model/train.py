import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Create directory if not exists
os.makedirs("ChurnPredict/model", exist_ok=True)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Drop CustomerID as it's not a feature
    df = df.drop('customerID', axis=1)
    
    # Handle TotalCharges: Convert to numeric, handle empty strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0) # Or mean/median, but 0 is safe for new customers
    
    # Encode Target: Churn
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Identify feature types
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return X, y, categorical_features, numerical_features

def main():
    filepath = "ChurnPredict/data/telco_churn.csv"
    X, y, cat_features, num_features = load_and_preprocess_data(filepath)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Logic Regression (Baseline)
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Random Forest (Final)
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    models = {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_pipeline
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }
        
    print("\nModel Comparison:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    # Save the best model (Random Forest is usually better for this dataset)
    # We will pick Random Forest as the final model
    best_model = rf_pipeline
    with open("ChurnPredict/model/churn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    # Save feature names for UI importance plot
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    onehot_features = list(cat_encoder.get_feature_names_out(cat_features))
    feature_names = num_features + onehot_features
    
    with open("ChurnPredict/model/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("\nBest model and feature names saved to ChurnPredict/model/")

if __name__ == "__main__":
    main()
