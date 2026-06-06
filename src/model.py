import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn  # or mlflow.xgboost if needed
import joblib

from src.config import MODEL_DIR

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Print classification report
    print(f"\nEvaluation for {model_name}")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

def train_all_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    mlflow.set_experiment("Telco_Customer_Churn")

    from sklearn.model_selection import cross_validate
    import numpy as np

    best_model = None
    best_score = 0.0
    results = {}

    for nome_modelo, modelo in models.items():
        with mlflow.start_run(run_name=nome_modelo):
            cv_resultados = cross_validate(modelo, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            media_acuracia_cv = np.mean(cv_resultados['test_score'])
            desvio_acuracia_cv = np.std(cv_resultados['test_score'])
            
            mlflow.log_metric("mean_test_score", media_acuracia_cv)
            mlflow.log_metric("std_test_score", desvio_acuracia_cv)
            
            params = modelo.get_params()
            for chave, valor in params.items():
                mlflow.log_param(chave, str(valor)[:250])
            
            modelo.fit(X_train, y_train)
            
            y_pred = modelo.predict(X_test)
            acuracia_teste = accuracy_score(y_test, y_pred)
            f1_teste = f1_score(y_test, y_pred)
            
            mlflow.log_metric("test_accuracy", acuracia_teste)
            mlflow.log_metric("test_f1", f1_teste)
            
            if hasattr(modelo, "predict_proba"):
                y_proba = modelo.predict_proba(X_test)[:, 1]
                auc_teste = roc_auc_score(y_test, y_proba)
                mlflow.log_metric("test_roc_auc", auc_teste)
            
            mlflow.sklearn.log_model(modelo, "modelo")
            
            results[nome_modelo] = acuracia_teste
            
            if f1_teste > best_score:
                best_score = f1_teste
                best_model = modelo

    caminho_melhor_modelo = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, caminho_melhor_modelo)
    
    return results


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curve Comparison of Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
