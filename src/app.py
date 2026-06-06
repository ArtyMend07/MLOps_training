import streamlit as st
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Sistema de Predicao de Churn (Telco)")

# Formularios
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Dados Pessoais")
    gender = st.selectbox("Genero", ["Male", "Female"])
    senior_citizen = st.selectbox("Idoso (SeniorCitizen)", [0, 1])
    partner = st.selectbox("Possui Parceiro(a)?", ["Yes", "No"])
    dependents = st.selectbox("Possui Dependentes?", ["Yes", "No"])

with col2:
    st.subheader("Contrato e Faturamento")
    tenure = st.slider("Meses de Contrato (tenure)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Faturamento Digital", ["Yes", "No"])
    payment = st.selectbox("Metodo de Pagamento", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Cobranca Mensal ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Cobranca Total Acumulada ($)", min_value=0.0, max_value=10000.0, value=600.0)

with col3:
    st.subheader("Servicos")
    phone = st.selectbox("Servico Telefonico", ["Yes", "No"])
    multiple = st.selectbox("Multiplas Linhas", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Provedor de Internet", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Seguranca Online", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Backup Online", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Protecao de Dispositivo", ["Yes", "No", "No internet service"])
    support = st.selectbox("Suporte Tecnico", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Filmes", ["Yes", "No", "No internet service"])

if st.button("Prever Risco de Churn", type="primary"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    with st.spinner("Processando inferencia..."):
        try:
            resposta = requests.post(API_URL, json=payload)
            resposta.raise_for_status()
            dados = resposta.json()
            
            st.divider()
            st.subheader("Resultado da Predicao")
            
            churn = dados.get("predicao_churn", 0)
            prob = dados.get("probabilidade", 0.0)
            
            if churn == 1:
                st.error("ALERTA: Alto Risco de Evasao (Churn Detectado)")
            else:
                st.success("SEGURO: Cliente propenso a Retencao")
                
            if prob is not None:
                st.write(f"Probabilidade Calculada pelo Modelo: {prob:.2%}")
                
                # Grafico
                fig, ax = plt.subplots(figsize=(6, 1))
                ax.barh(["Risco de Churn"], [prob], color="red" if churn == 1 else "green")
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
                st.pyplot(fig)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Falha de conexao com a API: {e}")
            st.info("O servidor FastAPI esta rodando? Utilize: uvicorn src.api:app --reload")
