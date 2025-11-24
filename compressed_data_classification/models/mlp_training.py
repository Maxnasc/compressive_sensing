import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import numpy as np
from codecarbon import OfflineEmissionsTracker
import os
from pathlib import Path


def mlp_training(X_train, y_train, X_test, y_test, X):

    # Iniciando o tracker de emissões <- CODECARBON
    tracker = OfflineEmissionsTracker(
        country_iso_code="BRA",
        output_file="compressed_data_classification/models/emissions/emissions_MLP.csv", log_level='critical'
    )

    param_grid = {
        "hidden_layer_sizes": [(20, 10), (10, 10), (15, 10)],
        "activation": ["tanh", "relu", "identity", "logistic"],
        "alpha": [0.01, 0.001, 0.0001],
        "early_stopping": [True],
        "learning_rate": ["constant"],
        "learning_rate_init": [0.001],
        "max_iter": [5000, 10000],
        "solver": ["lbfgs"],
    }

    total_combinations = len(ParameterGrid(param_grid))

    # Modelo base
    mlp = MLPClassifier(random_state=42)

    # GridSearch com validação cruzada
    grid_search = GridSearchCV(
        estimator=mlp,
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
    joblib.dump(best_model, "compressed_data_classification/models/mlp_model.pkl")

    # Avaliação no conjunto de teste
    y_pred = best_model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nDesempenho no conjunto de teste:")
    # print(f"MSE: {mse:.4f}")
    print(f"ACURACIA: {accuracy:.4f}")
    print(f"CLASSIF_REPORT: {report:.4f}")

    doc = {
        "melhores_parametros": grid_search.best_params_,
        "acuracia": round(accuracy, 4),
        "relatorio_classificacao": report, # Pode ser útil salvar o relatório completo
        "n_combinations": total_combinations,
        "mean_emission": round((emissions / total_combinations), 4) if total_combinations > 0 else 0,
    }

    with open("sensor_potencial_hidrico_ai/model/mlp/mlp_results_optical.json", "w") as f:
        json.dump(doc, f)

    # Plot de resulado de predição
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color="royalblue", s=60)
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--"
    )  # linha de referência y = x

    plt.title("Comparação entre Valores Reais e Preditos")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predito (Random Forest)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "sensor_potencial_hidrico_ai/model/mlp/plots/Scatter_real_vs_predito_optical.png"
    )
    # plt.show()

    df_resultado = pd.DataFrame({"Real": y_test, "Predito": y_pred})

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_resultado["Real"].values, label="Valor Real", marker="o")
    plt.plot(df_resultado["Predito"].values, label="Valor Predito", marker="x")
    plt.title("Comparação das Curvas: Valor Real vs Predito")
    plt.xlabel("Índice / Amostra")
    plt.ylabel("Potencial Hídrico")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "sensor_potencial_hidrico_ai/model/mlp/plots/Comparacao_real_predito_conjunto_de_teste_optical.png"
    )
    # plt.show()

    return doc
