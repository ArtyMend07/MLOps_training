from src.config import DATA_PATH, PROCESSED_DATA_PATH
from src.data_preprocessing import load_data, preprocess_data, log_processed_data
from src.visualization import *
from src.model import split_data, train_all_models, plot_roc_curves

def main():
    print("Iniciando o pipeline MLOps para Customer Churn...")
    
    df = load_data(DATA_PATH)
    
    X, y = preprocess_data(df)
    
    df_processado = X.copy()
    df_processado['Churn'] = y
    log_processed_data(df_processado, PROCESSED_DATA_PATH)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Executando o treinamento e o rastreamento via MLflow...")
    resultados = train_all_models(X_train, X_test, y_train, y_test)
    
    print("Modelos treinados com sucesso. Melhores métricas do teste:")
    for modelo, acuracia in resultados.items():
        print(f" - {modelo}: {acuracia:.4f}")

if __name__ == "__main__":
    main()
