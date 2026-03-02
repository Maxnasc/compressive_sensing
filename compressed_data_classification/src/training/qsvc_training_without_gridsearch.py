"""
Module: training/qsvc_training_without_gridsearch.py

Quadratic SVM training without grid search hyperparameter optimization.

This module implements a simplified Quadratic SVM training pipeline using predefined
hyperparameters instead of GridSearchCV. It provides faster training when hyperparameters
have been previously tuned.

Features:
- Uses fixed hyperparameters (SVM C and gamma)
- Cross-validation evaluation
- Model evaluation and visualization
- Efficient training without grid search overhead
- Result persistence

This variant is useful for production inference or final model training after
hyperparameter tuning has been completed.

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
from sklearn.preprocessing import StandardScaler

# Imports de funções auxiliares
from compressed_data_classification.src.pipelines.CS_transformer import CompressiveSensingTransformer
from compressed_data_classification.src.pipelines.FE_transformer import XPQRSFeatureExtractor
from codecarbon import OfflineEmissionsTracker
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# --- A função principal foi renomeada e os parâmetros internos ajustados ---

def qsvc_training_without_gridsearch(X_train, y_train, X_test, y_test, X, technique, label_encoder):
    """
    Train and evaluate a Quadratic SVM without GridSearchCV hyperparameter tuning.
    
    This function builds and trains a Quadratic SVM pipeline with fixed hyperparameters.
    It's suitable for final model training after hyperparameters have been optimized.
    
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
        - 'melhores_parametros': Hyperparameters used (not from optimization)
        - 'acuracia': Test set accuracy
        - 'roc_auc_score': ROC-AUC score
        - 'mse': Mean squared error
        - 'relatorio_classificacao': Classification report
        - 'best_mean_score': Mean cross-validation score
        - 'std_best_score_k_fold': Standard deviation of CV scores
    
    Notes
    -----
    - SVM kernel: Polynomial degree 2 (quadratic)
    - Cross-validation: 10-fold (for quick validation without full grid search)
    - Uses predefined C and gamma parameters instead of tuning
    - Saves trained model to: compressed_data_classification/src/models/best_qsvc_results/model_{technique}.pkl
    - Saves results to: compressed_data_classification/src/models/best_qsvc_results/results_{technique}.json
    - Saves confusion matrix plot to: compressed_data_classification/src/models/best_qsvc_results/plots/confusion_matrix_{technique}.png
    """
    
    # X_train = np.array(X_train, copy=True)
    # y_train = np.array(y_train, copy=True)

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
         
    from sklearn.ensemble import BaggingClassifier

    # 1. Configurar o estimador base com os parâmetros que você definiu
    svc_base = SVC(
        C=1000, 
        kernel='poly', 
        degree=2, 
        gamma='scale', 
        coef0=1, 
        decision_function_shape='ovo',
        max_iter=10000,  # <-- Força a parada após 10k iterações
        tol=1e-3,        # Tolerância (pode aumentar para 1e-2 para ser mais rápido)
        probability=True
    )

    # 2. Criar o Bagging para usar todos os núcleos do processador
    # Isso resolverá o baixo consumo de CPU que você notou
    qsvc_bagging = BaggingClassifier(
        estimator=svc_base, 
        n_estimators=10, 
        n_jobs=1, 
        verbose=1
    )

    # 3. Montar o Pipeline (sem as dobras do GridSearch)
    pipeline_steps = [
        ('feature_extraction', XPQRSFeatureExtractor()),
        ('scaler', StandardScaler()),
        ('qsvc', qsvc_bagging)
    ]

    # Adicionar o CS_transformer se necessário
    if technique not in ['original_data', 'random_mesurements']:
        pipeline_steps.insert(0, ('cs_transformer', CompressiveSensingTransformer(technique=technique, verbose=True)))

    pipeline = Pipeline(pipeline_steps)

    # 4. Treinar diretamente
    # O consumo de CPU subirá agora porque o Bagging processa n_estimators em paralelo
    pipeline.fit(X_train, y_train)
    
    # Avalia o pipeline em 10 partes sem o peso do GridSearch
    # cv_results = cross_validate(pipeline, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
    # print(f"Acurácia Média: {cv_results['test_score'].mean():.4f}")

    # Finalizando as medições de carbono
    # emissions: float = tracker.stop()

    # Resultados
    # Salvar o modelo ajustado
    joblib.dump(pipeline, model_path)

    # Avaliação no conjunto de teste
    y_pred = pipeline.predict(X_test)
    
    # Métricas de Classificação
    y_pred_proba = pipeline.predict_proba(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    report = classification_report(y_test, y_pred, zero_division=0)
    # std_test_score = grid_search.cv_results_['std_test_score'][best_index]
    # mean_test_score = grid_search.best_score_

    print(f"\nDesempenho no conjunto de teste:")
    print(f"Acurácia (Accuracy): {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)

    # Preparação dos dados para o JSON de resultados
    doc = {
        # "melhores_parametros": grid_search.best_params_,
        "acuracia": round(accuracy, 4),
        "roc_auc_score": round(roc_score, 4),
        "mse": round(mse, 4),
        "relatorio_classificacao": report, # Pode ser útil salvar o relatório completo
        # "Acurácia Média": cv_results['test_score'],
        # "n_combinations": total_combinations,
        # "best_mean_score": round(mean_test_score, 4),
        # "std_best_score_k_fold": round(std_test_score, 4)
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
        results = qsvc_training_without_gridsearch(X_train, y_train, X_test, y_test, X)
        print("\nTreinamento concluído. Resultados salvos.")
    except Exception as e:
        print(f"\nOcorreu um erro durante a execução: {e}")
        print("Verifique se as bibliotecas necessárias estão instaladas (incluindo codecarbon) e se as classes de y são inteiros (labels).")