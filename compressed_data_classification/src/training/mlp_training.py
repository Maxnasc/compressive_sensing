"""
Module: training/mlp_training.py

Multi-Layer Perceptron (MLP) neural network training for electrical disturbance classification.

This module implements a complete training pipeline for MLP classifiers that can process:
1. Raw electrical signals
2. Compressively sensed signals
3. Extracted feature vectors

The training pipeline includes:
- Compressive sensing transformation (optional)
- XPQRS feature extraction (15 features from signal derivatives)
- Hyperparameter tuning via GridSearchCV
- Model evaluation with confusion matrix visualization
- Model persistence and result logging

Author: Maxnasc7
License: MIT
Reference: XPQRS feature extraction based on signal processing techniques
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from codecarbon import OfflineEmissionsTracker
import os
from pathlib import Path
from compressed_data_classification.src.pipelines.CS_transformer import CompressiveSensingTransformer
from compressed_data_classification.src.pipelines.FE_transformer import XPQRSFeatureExtractor


def mlp_training(X_train, y_train, X_test, y_test, X, technique, label_encoder):
    """
    Train and evaluate an MLP classifier with compressive sensing and feature extraction.
    
    This function builds a complete ML pipeline that optionally includes compressive sensing
    transformation and XPQRS feature extraction, then performs hyperparameter tuning
    using GridSearchCV with 10-fold cross-validation.
    
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
    - Saves trained model to: compressed_data_classification/src/models/best_models_result/mlp/model_{technique}.pkl
    - Saves results to: compressed_data_classification/src/models/best_models_result/mlp/results_{technique}.json
    - Saves confusion matrix plot to: compressed_data_classification/src/models/best_models_result/mlp/plots/confusion_matrix_{technique}.png
    """

    # Iniciando o tracker de emissões <- CODECARBON
    # tracker = OfflineEmissionsTracker(
    #     country_iso_code="BRA",
    #     output_file=f"compressed_data_classification/models/best_models/emissions/emissions_MLP_{technique}.csv", log_level='critical'
    # )
    
    # --- Configurações de Paths ---
    # É uma boa prática garantir que os diretórios existam
    base_path = Path("compressed_data_classification/src/models/best_models_result/mlp")
    Path(base_path, "emissions").mkdir(parents=True, exist_ok=True)
    Path(base_path, "plots").mkdir(parents=True, exist_ok=True)
    
    # Ajustando paths para o SVC (RBF)
    model_path = f"compressed_data_classification/src/models/best_models_result/mlp/model_{technique}.pkl"
    results_path = f"compressed_data_classification/src/models/best_models_result/mlp/results_{technique}.json"
    report_result_path = f"compressed_data_classification/src/models/best_models_result/mlp/report_result_{technique}.json"
    cm_plot_path = f"compressed_data_classification/src/models/best_models_result/mlp/plots/confusion_matrix_{technique}.png"
         
    # Criando o pipeline
    pipeline_steps = [
        ('feature_extraction', XPQRSFeatureExtractor()),
        ('mlp', MLPClassifier(random_state=42))
    ]
    
    # ✅ Aplicar CS_transformer APENAS se não for original_data ou random_mesurements
    if technique not in ['original_data', 'random_mesurements']:
        pipeline_steps.insert(0, ('cs_transformer', CompressiveSensingTransformer(technique=technique, verbose=True, n_jobs=-1)))
    
    pipeline = Pipeline(pipeline_steps)

    # param_grid = {
    #     "mlp__hidden_layer_sizes": [(20, 10), (10, 10), (15, 10)],
    #     "mlp__activation": ["tanh", "relu", "identity", "logistic"],
    #     "mlp__alpha": [0.01, 0.001, 0.0001],
    #     "mlp__early_stopping": [True],
    #     "mlp__learning_rate": ["constant"],
    #     "mlp__learning_rate_init": [0.001],
    #     "mlp__max_iter": [5000, 10000],
    #     "mlp__solver": ["adam", "sgd"],
    # }
    
    # param_grid = {
    #     "mlp__hidden_layer_sizes": [(20, 10), (10, 10), (15, 10)],
    #     "mlp__activation": ["relu", "logistic"],
    #     "mlp__alpha": [0.01, 0.001],
    #     "mlp__early_stopping": [True],
    #     "mlp__learning_rate": ["constant"],
    #     "mlp__learning_rate_init": [0.001],
    #     "mlp__max_iter": [5000, 10000],
    #     "mlp__solver": ["adam", "sgd"],
    # }
    
    param_grid = {
        "mlp__hidden_layer_sizes": [(20, 10)],
        "mlp__activation": ["relu"],
        "mlp__alpha": [0.01],
        "mlp__early_stopping": [True],
        "mlp__learning_rate": ["constant"],
        "mlp__learning_rate_init": [0.001],
        "mlp__max_iter": [5000],
        "mlp__solver": ["sgd"],
    }
    
    # Melhores parâmetros para cada técnica
    # Carregando o json com as melhores métricas
    # try:
    #     with open("compressed_data_classification/src/models/best_models_result_without_sliding_window/mlp/results_reconstructed_2_dot_5.json") as arq:
    #         result_json_content = json.load(arq)
        
    #     best_param_grid = {
    #         "mlp__hidden_layer_sizes": [(result_json_content['melhores_parametros']['mlp__hidden_layer_sizes'][0], result_json_content['melhores_parametros']['mlp__hidden_layer_sizes'][1])],
    #         "mlp__activation": [result_json_content['melhores_parametros']['mlp__activation']],
    #         "mlp__alpha": [result_json_content['melhores_parametros']['mlp__alpha']],
    #         "mlp__early_stopping": [result_json_content['melhores_parametros']['mlp__early_stopping']],
    #         "mlp__learning_rate": [result_json_content['melhores_parametros']['mlp__learning_rate']],
    #         "mlp__learning_rate_init": [result_json_content['melhores_parametros']['mlp__learning_rate_init']],
    #         "mlp__max_iter": [result_json_content['melhores_parametros']['mlp__max_iter']],
    #         "mlp__solver": [result_json_content['melhores_parametros']['mlp__solver']],
    #     }
    # except:
    #     total_combinations = len(ParameterGrid(param_grid))
    #     print(f"Total de Combinações do Grid Search: {total_combinations}")

    # GridSearch com validação cruzada
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="accuracy", # <-- Alteração: Usando métrica de classificação
        n_jobs=-1,
        verbose=1,
    )

    # Iniciando as medições de carbono
    # tracker.start()

    # Executar busca
    grid_search.fit(X_train, y_train)

    # Finalizando as medições de carbono
    # emissions: float = tracker.stop()
    
    # Carregando o modelo

    # Resultados
    best_model = grid_search.best_estimator_
    best_index = grid_search.best_index_
    # best_model = joblib.load(f"compressed_data_classification/models/best_models/mlp/mlp_model_{technique}.pkl")
    print("\nMelhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)

    # Salvar o modelo ajustado
    joblib.dump(best_model, model_path)
    # joblib.dump(best_model, f"compressed_data_classification/models/mlp/mlp_model_{technique}.pkl")

    # Avaliação no conjunto de teste
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    mse = mean_squared_error(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    std_test_score = grid_search.cv_results_['std_test_score'][best_index]
    mean_test_score = grid_search.best_score_

    print(f"\nDesempenho no conjunto de teste:")
    print(f"MSE: {mse:.4f}")
    print(f"ACURACIA: {accuracy:.4f}")
    print(f"CLASSIF_REPORT: {report}")

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

    with open(results_path, "w") as f:
        json.dump(doc, f)

    with open(report_result_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # --- Plot: Matriz de Confusão para 17 Classes ---
    # (Substitui os plots de regressão)
    # Carrega o encoder usado no treinamento
    # label_encoder = joblib.load("compressed_data_classification/models/label_encoder.pkl")

    # Converte back para os nomes originais
    y_test_str = label_encoder.inverse_transform(y_test)
    y_pred_str = label_encoder.inverse_transform(y_pred)

    # Cria matriz de confusão com os labels originais
    cm = confusion_matrix(y_test_str, y_pred_str, labels=label_encoder.classes_)

    plt.figure(figsize=(14, 12))
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
    plt.title("Matriz de Confusão (MLP) - Labels Originais")
    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.savefig(cm_plot_path)

    return doc
