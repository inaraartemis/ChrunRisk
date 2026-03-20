import gradio as gr
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os

PORT = os.environ.get("PORT", 8015)
API_URL = f"http://127.0.0.1:{PORT}"

# Load feature importance from model
MODEL_PATH = "model/churn_model.pkl"
FEATURE_NAMES_PATH = "model/feature_names.pkl"

def get_feature_importance():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
        return None
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
        
    importances = model.named_steps['classifier'].feature_importances_
    # Pick top 10
    indices = np.argsort(importances)[-10:]
    
    fig = px.bar(
        x=importances[indices], 
        y=[feature_names[i] for i in indices],
        orientation='h',
        title="Top 10 Feature Importances",
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    return fig

def predict_single(gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges):
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        res_data = response.json()
        
        prediction = res_data["prediction"]
        prob = res_data["probability"]
        risk = res_data["risk_category"]
        
        # Risk Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Churn Probability: {risk} Risk"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "salmon"}
                ],
            }
        ))
        
        return f"Prediction: {prediction}", f"Probability: {prob:.2%}", f"Risk Category: {risk}", fig
    except Exception as e:
        return "Error", str(e), "N/A", None

def predict_batch(file):
    try:
        files = {'file': open(file.name, 'rb')}
        response = requests.post(f"{API_URL}/predict-batch", files=files)
        res_data = response.json()
        
        results = res_data["batch_results"]
        df_results = pd.DataFrame(results)
        
        # Visualization for batch
        # Count of risk categories
        risk_counts = df_results['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Category', 'Count']
        fig = px.pie(risk_counts, values='Count', names='Risk Category', title="Batch Churn Risk Distribution", color_discrete_map={'Low':'green', 'Medium':'yellow', 'High':'red'})
        
        return df_results, fig
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}]), None

# Define Gradio Theme
theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

with gr.Blocks(theme=theme, title="Telco Churn Prediction Platform") as demo:
    gr.Markdown("# 📊 Customer Churn Prediction Platform")
    gr.Markdown("Predict customer churn using machine learning and get actionable insights.")
    
    with gr.Tabs():
        with gr.TabItem("Single Customer Prediction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Demographic Info")
                    gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                    senior = gr.Radio(["Yes", "No"], label="Senior Citizen", value="No")
                    partner = gr.Radio(["Yes", "No"], label="Partner", value="No")
                    dependents = gr.Radio(["Yes", "No"], label="Dependents", value="No")
                    
                    gr.Markdown("### Service Details")
                    tenure = gr.Slider(0, 100, step=1, label="Tenure (Months)", value=12)
                    phone = gr.Radio(["Yes", "No"], label="Phone Service", value="Yes")
                    multiple = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
                    internet = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                    
                with gr.Column():
                    gr.Markdown("### Features & Add-ons")
                    security = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                    backup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                    protection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                    support = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                    tv = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No")
                    movies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No")
                    
                    gr.Markdown("### Billing")
                    contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                    paperless = gr.Radio(["Yes", "No"], label="Paperless Billing", value="Yes")
                    payment = gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check")
                    monthly = gr.Number(label="Monthly Charges ($)", value=70.0)
                    total = gr.Number(label="Total Charges ($)", value=840.0)

            predict_btn = gr.Button("Predict Churn", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    out_pred = gr.Textbox(label="Prediction Result")
                    out_prob = gr.Textbox(label="Churn Probability")
                    out_risk = gr.Textbox(label="Risk Level")
                with gr.Column():
                    out_plot = gr.Plot()

            predict_btn.click(
                predict_single,
                inputs=[gender, senior, partner, dependents, tenure, phone, multiple, internet, security, backup, protection, support, tv, movies, contract, paperless, payment, monthly, total],
                outputs=[out_pred, out_prob, out_risk, out_plot]
            )

        with gr.TabItem("Batch Upload (CSV)"):
            gr.Markdown("### Upload a CSV file for bulk churn prediction")
            csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            batch_btn = gr.Button("Run Batch Prediction", variant="primary")
            
            with gr.Row():
                out_batch_df = gr.Dataframe(label="Results")
                out_batch_plot = gr.Plot()
                
            batch_btn.click(predict_batch, inputs=csv_input, outputs=[out_batch_df, out_batch_plot])

        with gr.TabItem("Model Insights"):
            gr.Markdown("### Feature Importance Analysis")
            gr.Markdown("Understand which factors contribute most to customer churn.")
            importance_plot = gr.Plot(value=get_feature_importance())
            refresh_btn = gr.Button("Refresh Insights")
            refresh_btn.click(get_feature_importance, outputs=importance_plot)

if __name__ == "__main__":
    demo.launch()
