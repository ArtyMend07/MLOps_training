import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MODEL_DIR

modelo_preditivo = {}

class ClientData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    caminho_modelo = os.path.join(MODEL_DIR, "best_model.pkl")
    caminho_scaler = os.path.join(MODEL_DIR, "scaler.pkl")
    caminho_colunas = os.path.join(MODEL_DIR, "columns.pkl")
    
    if not (os.path.exists(caminho_modelo) and os.path.exists(caminho_scaler) and os.path.exists(caminho_colunas)):
        raise RuntimeError("Artefatos do modelo nao encontrados. Execute o pipeline de treinamento primeiro.")

    modelo_preditivo["modelo"] = joblib.load(caminho_modelo)
    modelo_preditivo["scaler"] = joblib.load(caminho_scaler)
    modelo_preditivo["colunas_treino"] = joblib.load(caminho_colunas)
    
    yield
    modelo_preditivo.clear()

app = FastAPI(title="Telco Customer Churn API", lifespan=lifespan)

@app.post("/predict")
def predict_churn(cliente: ClientData):
    try:
        dados_df = pd.DataFrame([cliente.model_dump()])
        
        # Mapeamento binario
        colunas_binarias = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in colunas_binarias:
            dados_df[col] = dados_df[col].map({'Yes': 1, 'No': 0})
            
        dados_df['gender'] = dados_df['gender'].map({'Male': 1, 'Female': 0})
        
        # Mapeamento dummy (One-Hot Encoding)
        dados_df = pd.get_dummies(dados_df, columns=[
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ])
        
        # Alinhamento das colunas com o treinamento (adiciona as que faltam com 0, remove extras)
        colunas_treino = modelo_preditivo["colunas_treino"]
        for col in colunas_treino:
            if col not in dados_df.columns:
                dados_df[col] = 0
        dados_df = dados_df[colunas_treino]
        
        # Escalonamento
        scaler = modelo_preditivo["scaler"]
        dados_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            dados_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )
        
        # Inferencia
        modelo = modelo_preditivo["modelo"]
        predicao = modelo.predict(dados_df)[0]
        probabilidade = modelo.predict_proba(dados_df)[0][1] if hasattr(modelo, "predict_proba") else None
        
        return {
            "predicao_churn": int(predicao),
            "probabilidade": probabilidade
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
