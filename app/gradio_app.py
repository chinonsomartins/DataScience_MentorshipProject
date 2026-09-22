"""Gradio UI for the churn model.

Run (in a SEPARATE terminal from the API, both with the venv active):
    python app/gradio_app.py

Requires api/main.py to already be running (uvicorn api.main:app --port 8000) -
this UI is a client, it does not load the model itself. See the module
docstring reasoning: one inference path (predict.py, behind the API),
not two copies that can drift out of sync.
"""
from __future__ import annotations

import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"

# Mirrors the Literal choices in api/main.py's CustomerRecord exactly.
# If you add/change a category there, update it here too - there's no
# automatic sync between the two, since they're separate processes.
YES_NO = ["Yes", "No"]
YES_NO_INTERNET = ["Yes", "No", "No internet service"]


def predict_churn(
    gender, senior_citizen, partner, dependents, tenure, phone_service,
    multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method, monthly_charges, total_charges,
):
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
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
        "TotalCharges": float(total_charges),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
    except requests.exceptions.ConnectionError:
        return "⚠️ Can't reach the API. Is `uvicorn api.main:app --port 8000` running?"

    if response.status_code == 422:
        return f"⚠️ Validation error — check inputs:\n{response.json()['detail']}"
    if response.status_code != 200:
        return f"⚠️ API error ({response.status_code}): {response.text}"

    result = response.json()
    label = "🔴 Likely to churn" if result["churn_predicted"] else "🟢 Likely to stay"
    return f"{label}\n\nChurn probability: {result['churn_probability']:.1%}"


with gr.Blocks(title="Customer Churn Predictor") as demo:
    gr.Markdown("# Customer Churn Predictor")
    gr.Markdown("Fill in customer details and get a churn risk prediction from the XGBoost model.")

    with gr.Row():
        with gr.Column():
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Female")
            senior_citizen = gr.Radio(YES_NO, label="Senior Citizen", value="No")
            partner = gr.Radio(YES_NO, label="Has Partner", value="No")
            dependents = gr.Radio(YES_NO, label="Has Dependents", value="No")
            tenure = gr.Slider(0, 100, value=12, step=1, label="Tenure (months)")
            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"
            )
            payment_method = gr.Dropdown(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                label="Payment Method",
                value="Electronic check",
            )
            paperless_billing = gr.Radio(YES_NO, label="Paperless Billing", value="Yes")
            monthly_charges = gr.Number(label="Monthly Charges ($)", value=70.0)
            total_charges = gr.Number(label="Total Charges ($)", value=840.0)

        with gr.Column():
            phone_service = gr.Radio(YES_NO, label="Phone Service", value="Yes")
            multiple_lines = gr.Radio(
                ["Yes", "No", "No phone service"], label="Multiple Lines", value="No"
            )
            internet_service = gr.Dropdown(
                ["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"
            )
            online_security = gr.Radio(YES_NO_INTERNET, label="Online Security", value="No")
            online_backup = gr.Radio(YES_NO_INTERNET, label="Online Backup", value="No")
            device_protection = gr.Radio(YES_NO_INTERNET, label="Device Protection", value="No")
            tech_support = gr.Radio(YES_NO_INTERNET, label="Tech Support", value="No")
            streaming_tv = gr.Radio(YES_NO_INTERNET, label="Streaming TV", value="No")
            streaming_movies = gr.Radio(YES_NO_INTERNET, label="Streaming Movies", value="No")

    predict_btn = gr.Button("Predict Churn Risk", variant="primary")
    output = gr.Textbox(label="Result", lines=3)

    predict_btn.click(
        fn=predict_churn,
        inputs=[
            gender, senior_citizen, partner, dependents, tenure, phone_service,
            multiple_lines, internet_service, online_security, online_backup,
            device_protection, tech_support, streaming_tv, streaming_movies,
            contract, paperless_billing, payment_method, monthly_charges, total_charges,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()