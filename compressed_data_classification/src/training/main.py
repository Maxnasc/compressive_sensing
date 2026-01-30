import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import sys
import os
from rich.console import Console
from rich.status import Status

# Pega o caminho absoluto até a pasta 'compressed_data_classification'
# Baseado na sua imagem, subindo 3 níveis a partir do main.py
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

from compressed_data_classification.src.training.mlp_training import mlp_training
from compressed_data_classification.src.training.svm_with_rbf_training import svm_with_rbf_training
from compressed_data_classification.src.training.qsvc_training import qsvc_training

import time
from compressed_data_classification.utils.utils import send_telegram_msg

def import_and_split_dataset(data_path):
    # Carregue os dados
    df = pd.read_csv(data_path)

    # df = df.drop(columns=['Unnamed: 0', 'F3_norm'])

    # Variável alvo

    y = df['target']
    X = df.drop(columns=['target'])

    # Encoder das labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("\nMapeamento de Classes:", class_mapping)

    # Divisão em treino e teste para avaliação final depois do ajuste
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder, X

# Iniciando o código
start_global = time.time()

# Técnica de compressão de dados energy/topk/pca/pure_alpha/random_mesurements
# techniques = ['energy','topk','pca','pure_alpha','random_mesurements', 'original_data']
# techniques = ['original_data', 'reconstructed_2_dot_5', 'reconstructed_random', 'random_mesurements']
# techniques = ['reconstructed_2_dot_5', 'reconstructed_random', 'random_mesurements']
techniques = ['original_data']

# Console para exibir status
console = Console()


# Treinar usando os diferentes métodos de treinamento e salvar cada um
for technique in techniques:
    
    if technique == 'original_data':
        X_train, X_test, y_train, y_test, label_encoder, X = import_and_split_dataset('compressed_data_classification/data/raw/data.csv')
    elif technique == 'reconstructed_2_dot_5':
        X_train, X_test, y_train, y_test, label_encoder, X = import_and_split_dataset('compressed_data_classification/data/processed/data_sampled_2_dot_5_khz.csv')
    else:
        X_train, X_test, y_train, y_test, label_encoder, X = import_and_split_dataset('compressed_data_classification/data/processed/data_sampled_with_phi.csv')
    
    print(f"\n{'='*60}")
    print(f"Técnica: {technique}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"{'='*60}\n")
    
    try:
        # Modelos
        # with Status(f"[bold green]Treinando MLP para {technique}...[/]", spinner="dots"):
        #     mlp_info = mlp_training(X_train, y_train, X_test, y_test, X, technique, label_encoder)
        # print()
        # with Status(f"[bold green]Treinando SVM com RBF para {technique}...[/]", spinner="dots"):
        #     svm_rbf = svm_with_rbf_training(X_train, y_train, X_test, y_test, X, technique, label_encoder)
        # print()
        with Status(f"[bold green]Treinando SVM quadrático para {technique}...[/]", spinner="dots"):
            qsvc = qsvc_training(X_train, y_train, X_test, y_test, X, technique, label_encoder)
        print()
    except  Exception as e:
        send_telegram_msg(f"Falha na execução do código -> ERRO: {e}")
        # send_telegram_msg(f"Falha na execução do código: {technique}")
        break

    # Montando tabela de comparação entre os modelos
    # df = pd.DataFrame([mlp_info, svm_rbf, qsvc])
    df = pd.DataFrame([qsvc])
    # df = df.drop('melhores_parametros')

    # print(df)
    df.to_excel(f'compressed_data_classification\src\models/best_models_result/resultados_modelos_{technique}.xlsx')


end_global = time.time()
duration = end_global-start_global

minutes, seconds = divmod(duration, 60)

send_telegram_msg(f"🚀 Treinamento dos modelos usando compressive sensing OMP terminado!\n⏱️ Tempo total: {int(minutes)}m {seconds:.2f}s")

# Mostrar todos os gráficos
# plt.show() 