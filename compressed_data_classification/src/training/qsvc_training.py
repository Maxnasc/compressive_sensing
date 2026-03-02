"""
Module: training/qsvc_training.py

Quadratic Support Vector Machine (QSVC) for electrical disturbance classification.

This module implements Quadratic SVM training (polynomial kernel degree 2) for 
multiclass classification of electrical disturbances. The training pipeline includes:
- Compressive sensing transformation (optional)
- XPQRS feature extraction
- Hyperparameter tuning via GridSearchCV with 10-fold CV
- Model evaluation and visualization
- Result persistence

The polynomial (quadratic) kernel is effective for capturing non-linear relationships
in electrical signal features.

Author: Maxnasc7
License: MIT
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import numpy as np
import seaborn as sns
import os
from pathlib import Path

# Módulos do Sklearn para Classificação
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.svm import SVC # <-- Alteração: Importando SVC

# Imports de funções auxiliares
from compressed_data_classification.src.pipelines.CS_transformer import CompressiveSensingTransformer
from compressed_data_classification.src.pipelines.FE_transformer import XPQRSFeatureExtractor
from codecarbon import OfflineEmissionsTracker
from sklearn.pipeline import Pipeline

# --- A função principal foi renomeada e os parâmetros internos ajustados ---

def qsvc_training(X_train, y_train, X_test, y_test, X, technique, label_encoder):
    """
    Train and evaluate a Quadratic SVM (SVC with polynomial kernel, degree=2).
    
    This function builds a complete ML pipeline that optionally includes compressive sensing
    transformation and XPQRS feature extraction, then performs hyperparameter tuning
    using GridSearchCV with 10-fold cross-validation on an SVM with quadratic kernel.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : np.ndarray
        Encoded training labels
    X_test : pd.DataFrame or np.ndarray
        Testing features
    y_test : np.ndarray
        Encoded testing labels
    X : pd.DataFrame or np.ndarray
        Complete feature matrix (for reference)
    technique : str
        Compressive sensing technique to apply:
        - 'original_data': No compression
        - 'random_mesurements': Random CS measurements
        - 'reconstructed_2_dot_5': Reconstructed from 2.5 kHz sampling
        - others: Apply CS transformer
    label_encoder : LabelEncoder
        Fitted label encoder for inverse transforming predictions
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'melhores_parametros': Best hyperparameters found
        - 'acuracia': Test set accuracy
        - 'roc_auc_score': ROC-AUC score
        - 'mse': Mean squared error
        - 'relatorio_classificacao': Classification report
        - 'best_mean_score': Best CV score
        - 'std_best_score_k_fold': Standard deviation of CV scores
    
    Notes
    -----
    - SVM kernel: Polynomial degree 2 (quadratic)
    - Cross-validation: 10-fold
    - Saves trained model to: compressed_data_classification/src/models/best_qsvc_results/model_{technique}.pkl
    - Saves results to: compressed_data_classification/src/models/best_qsvc_results/results_{technique}.json
    - Saves confusion matrix plot to: compressed_data_classification/src/models/best_qsvc_results/plots/confusion_matrix_{technique}.png
    """

    # --- Configurações de Paths ---
    # É uma boa prática garantir que os diretórios existam
    base_path = Path("compressed_data_classification/src/models/best_qsvc_results")
    Path(base_path, "emissions").mkdir(parents=True, exist_ok=True)
    Path(base_path, "plots").mkdir(parents=True, exist_ok=True)
    
    # Ajustando paths para o SVC (RBF)
    model_path = f"{base_path}/model_{technique}.pkl"
    results_path = f"{base_path}/results_{technique}.json"
    report_result_path = f"{base_path}/report_result_{technique}.json"
    scatter_plot_path = f"{base_path}/plots/confusion_matrix_{technique}.png"


    # Iniciando o tracker de emissões <- CODECARBON
    # tracker = OfflineEmissionsTracker(
    #     country_iso_code="BRA",
    #     output_file=f"compressed_data_classification/models/best_models/emissions/emissions_SVC_RBF_{technique}.csv",
    #     log_level='critical'
    # )
         
    # Criando o pipeline
    pipeline_steps = [
        ('feature_extraction', XPQRSFeatureExtractor()),
        ('qsvc', SVC())
    ]
    
    # ✅ Aplicar CS_transformer APENAS se não for original_data ou random_mesurements
    if technique not in ['original_data', 'random_mesurements']:
        pipeline_steps.insert(0, ('cs_transformer', CompressiveSensingTransformer(technique=technique, verbose=True)))
    
    pipeline = Pipeline(pipeline_steps)
    
    param_grid = {
        # C (ou K no artigo): Fator de penalidade. 
        # O padrão do MATLAB/Artigo é 1.
        'qsvc__C': [1000], 

        # Habilita probabilidades para evitar erro no predict_proba
        'qsvc__probability': [True],
        
        # Kernel fixo em polinomial de grau 2 conforme o artigo
        'qsvc__kernel': ['poly'],
        'qsvc__degree': [2],
        
        # Gamma: 'scale' é a abordagem moderna recomendada (1 / (n_features * X.var()))
        'qsvc__gamma': ['scale'],
        
        # Coef0: O parâmetro 'r' na fórmula (gamma*<x,x'> + r)^d. 
        # Em kernels quadráticos do MATLAB, o padrão é 1.
        'qsvc__coef0': [1],
        
        # O MATLAB usa One-vs-One (OvO) por padrão para SVM multiclasse
        'qsvc__decision_function_shape': ['ovo'],
        
        'qsvc__verbose': [True]
    }
    
    # Melhores parâmetros para cada técnica
    # Carregando o json com as melhores métricas
    # try:
    #     with open(results_path) as arq:
    #         result_json_content = json.load(arq)
        
    #     best_param_grid = {
    #         "qsvc__C": [result_json_content['melhores_parametros']['qsvc__C']],
    #         "qsvc__gamma": [result_json_content['melhores_parametros']['qsvc__gamma']],
    #         "qsvc__kernel": [result_json_content['melhores_parametros']['qsvc__kernel']],
    #         "qsvc__random_state": [42],
    #         "qsvc__probability": [True], # Necessário para predict_proba
    #     }
    # except:
    # total_combinations = len(ParameterGrid(param_grid))
    # print(f"Total de Combinações do Grid Search: {total_combinations}")

    # GridSearch com validação cruzada
    # Usando 'accuracy' como métrica principal para classificação multiclasse
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy", # <-- Alteração: Usando métrica de classificação / roc_auc_ovr
        n_jobs=1,
        verbose=1,
    )

    # Iniciando as medições de carbono
    # tracker.start()

    # Executar busca
    # Atenção: SVC requer que y_train seja um vetor de classes (0, 1, ..., 16)
    grid_search.fit(X_train, y_train)

    # Finalizando as medições de carbono
    # emissions: float = tracker.stop()

    # Resultados
    best_model = grid_search.best_estimator_
    best_index = grid_search.best_index_
    # best_model = joblib.load(model_path)
    print("\nMelhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)

    # Salvar o modelo ajustado
    joblib.dump(best_model, model_path)

    # Avaliação no conjunto de teste
    y_pred = best_model.predict(X_test)
    
    # Métricas de Classificação
    y_pred_proba = best_model.predict_proba(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    report = classification_report(y_test, y_pred, zero_division=0)
    std_test_score = grid_search.cv_results_['std_test_score'][best_index]
    mean_test_score = grid_search.best_score_

    print(f"\nDesempenho no conjunto de teste:")
    print(f"Acurácia (Accuracy): {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)

    # Preparação dos dados para o JSON de resultados
    doc = {
        "melhores_parametros": grid_search.best_params_,
        "acuracia": round(accuracy, 4),
        "roc_auc_score": round(roc_score, 4),
        "mse": round(mse, 4),
        "relatorio_classificacao": report, # Pode ser útil salvar o relatório completo
        # "n_combinations": total_combinations,
        "best_mean_score": round(mean_test_score, 4),
        "std_best_score_k_fold": round(std_test_score, 4)
        # "mean_emission": round((emissions / total_combinations), 4) if total_combinations > 0 and total_combinations != None else 0,
    }

    # Garantir que o diretório de plots exista
    Path(scatter_plot_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(doc, f, indent=4)

    with open(report_result_path, "w") as f:
        json.dump(report, f, indent=4)

    # --- Plot: Matriz de Confusão para 17 Classes ---
    # (Substitui os plots de regressão)
    
    y_test_str = label_encoder.inverse_transform(y_test)
    y_pred_str = label_encoder.inverse_transform(y_pred)
    
    cm = confusion_matrix(y_test_str, y_pred_str, labels=label_encoder.classes_)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        cbar=False,
        linewidths=.5,
        linecolor='black',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Matriz de Confusão (SVC RBF) - Classificação 17 Classes")
    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.savefig(scatter_plot_path)
    # plt.show()

    return doc

# --- Exemplo de Uso (simulação de dados de entrada) ---
if __name__ == '__main__':
    print("Simulando dados de entrada para SVC (17 classes)...")
    
    # Criando dados de simulação: 1000 amostras, 10 características, 17 classes
    N_SAMPLES = 1000
    N_FEATURES = 10
    N_CLASSES = 17
    
    # X (Características)
    X = pd.DataFrame(np.random.rand(N_SAMPLES, N_FEATURES), columns=[f'feature_{i}' for i in range(N_FEATURES)])
    
    # y (Classes) - Garantindo que as 17 classes estejam presentes
    y = np.random.randint(0, N_CLASSES, N_SAMPLES)
    y = pd.Series(y)

    # Divisão Simples (simulando X_train, X_test, y_train, y_test)
    split_index = int(0.8 * N_SAMPLES)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Dimensões do Treinamento: X={X_train.shape}, y={y_train.shape}")
    print(f"Dimensões do Teste: X={X_test.shape}, y={y_test.shape}")
    print(f"Número de classes únicas em y_train: {y_train.nunique()}")

    # Execução da função
    try:
        results = qsvc_training(X_train, y_train, X_test, y_test, X)
        print("\nTreinamento concluído. Resultados salvos.")
    except Exception as e:
        print(f"\nOcorreu um erro durante a execução: {e}")
        print("Verifique se as bibliotecas necessárias estão instaladas (incluindo codecarbon) e se as classes de y são inteiros (labels).")