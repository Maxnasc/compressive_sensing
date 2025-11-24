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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC # <-- Alteração: Importando SVC
from codecarbon import OfflineEmissionsTracker

# --- A função principal foi renomeada e os parâmetros internos ajustados ---

def svc_rbf_training(X_train, y_train, X_test, y_test, X):
    """
    Treina e avalia um classificador SVM com Kernel RBF (SVC) 
    usando Grid Search para um problema de classificação multiclasse.
    """

    # --- Configurações de Paths ---
    # É uma boa prática garantir que os diretórios existam
    base_path = Path("compressed_data_classification/models/")
    Path(base_path, "emissions").mkdir(parents=True, exist_ok=True)
    
    # Ajustando paths para o SVC (RBF)
    model_path = "compressed_data_classification/models/svc_rbf_model.pkl"
    results_path = "sensor_potencial_hidrico_ai/model/svc_rbf/svc_rbf_results_optical.json"
    scatter_plot_path = "sensor_potencial_hidrico_ai/model/svc_rbf/plots/Confusion_Matrix_optical.png"


    # Iniciando o tracker de emissões <- CODECARBON
    tracker = OfflineEmissionsTracker(
        country_iso_code="BRA",
        output_file=Path(base_path, "emissions/emissions_SVC_RBF.csv").as_posix(),
        log_level='critical'
    )
    
    # --- Hiperparâmetros para o SVC com Kernel RBF ---
    # C: Parâmetro de Regularização (inverso da força de regularização)
    # gamma: Coeficiente do kernel RBF (influencia a "alcance" de uma única amostra de treinamento)
    param_grid = {
        "C": [0.1, 1, 10], # Regularização
        "gamma": [0.001, 0.01, 0.1], # Kernel RBF
        "kernel": ["rbf"], # Focando no Kernel RBF
        "random_state": [42]
    }

    total_combinations = len(ParameterGrid(param_grid))
    print(f"Total de Combinações do Grid Search: {total_combinations}")


    # Modelo base: Support Vector Classifier
    svc = SVC(random_state=42)

    # GridSearch com validação cruzada
    # Usando 'accuracy' como métrica principal para classificação multiclasse
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy", # <-- Alteração: Usando métrica de classificação
        n_jobs=-1,
        verbose=1,
    )

    # Iniciando as medições de carbono
    tracker.start()

    # Executar busca
    # Atenção: SVC requer que y_train seja um vetor de classes (0, 1, ..., 16)
    grid_search.fit(X_train, y_train)

    # Finalizando as medições de carbono
    emissions: float = tracker.stop()

    # Resultados
    best_model = grid_search.best_estimator_
    print("\nMelhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)

    # Salvar o modelo ajustado
    joblib.dump(best_model, model_path)

    # Avaliação no conjunto de teste
    y_pred = best_model.predict(X_test)
    
    # Métricas de Classificação
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nDesempenho no conjunto de teste:")
    print(f"Acurácia (Accuracy): {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)

    # Preparação dos dados para o JSON de resultados
    doc = {
        "melhores_parametros": grid_search.best_params_,
        "acuracia": round(accuracy, 4),
        "relatorio_classificacao": report, # Pode ser útil salvar o relatório completo
        "n_combinations": total_combinations,
        "mean_emission": round((emissions / total_combinations), 4) if total_combinations > 0 and total_combinations != None else 0,
    }

    # Garantir que o diretório de plots exista
    Path(scatter_plot_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(doc, f, indent=4)

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
        results = svc_rbf_training(X_train, y_train, X_test, y_test, X)
        print("\nTreinamento concluído. Resultados salvos.")
    except Exception as e:
        print(f"\nOcorreu um erro durante a execução: {e}")
        print("Verifique se as bibliotecas necessárias estão instaladas (incluindo codecarbon) e se as classes de y são inteiros (labels).")