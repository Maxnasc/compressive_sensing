import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from codecarbon import OfflineEmissionsTracker
import os
from pathlib import Path
from compressed_data_classification.CS_transformer import CompressiveSensingTransformer


def mlp_training(X_train, y_train, X_test, y_test, X, technique):

    # Iniciando o tracker de emissões <- CODECARBON
    tracker = OfflineEmissionsTracker(
        country_iso_code="BRA",
        output_file=f"compressed_data_classification/models/emissions/emissions_MLP_{technique}.csv", log_level='critical'
    )
    
    # Instanciando o compressive sensing transformer
    cs_transformer = CompressiveSensingTransformer(technique=technique)

    # Modelo base
    mlp = MLPClassifier(random_state=42)
    
    # Criando o pipeline
    pipeline = Pipeline([
        ('cs_trasnformer', cs_transformer),
        ('mlp', mlp)
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(20, 10), (10, 10), (15, 10)],
        "mlp__activation": ["tanh", "relu", "identity", "logistic"],
        "mlp__alpha": [0.01, 0.001, 0.0001],
        "mlp__early_stopping": [True],
        "mlp__learning_rate": ["constant"],
        "mlp__learning_rate_init": [0.001],
        "mlp__max_iter": [5000, 10000],
        "mlp__solver": ["lbfgs"],
    }

    total_combinations = len(ParameterGrid(param_grid))

    # GridSearch com validação cruzada
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    # Iniciando as medições de carbono
    tracker.start()

    # Executar busca
    grid_search.fit(X_train, y_train)

    # Finalizando as medições de carbono
    emissions: float = tracker.stop()

    # Resultados
    best_model = grid_search.best_estimator_
    print("\nMelhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)

    # Salvar o modelo ajustado
    joblib.dump(best_model, f"compressed_data_classification/models/mlp_model_{technique}.pkl")

    # Avaliação no conjunto de teste
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nDesempenho no conjunto de teste:")
    print(f"MSE: {mse:.4f}")
    print(f"ACURACIA: {accuracy:.4f}")
    print(f"CLASSIF_REPORT: {report:.4f}")

    doc = {
        "melhores_parametros": grid_search.best_params_,
        "acuracia": round(accuracy, 4),
        "relatorio_classificacao": report, # Pode ser útil salvar o relatório completo
        "n_combinations": total_combinations,
        "mean_emission": round((emissions / total_combinations), 4) if total_combinations > 0 and total_combinations != None else 0,
    }

    with open(f"compressed_data_classification/models/mlp/mlp_results_optical_{technique}.json", "w") as f:
        json.dump(doc, f)
        
    # --- Plot: Matriz de Confusão para 17 Classes ---
    # (Substitui os plots de regressão)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        cbar=False,
        linewidths=.5,
        linecolor='black'
    )
    plt.title("Matriz de Confusão (SVC RBF) - Classificação 17 Classes")
    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.savefig(f"compressed_data_classification/models/mlp/confusion_matrix_{technique}.png")
    # plt.show()

    return doc
