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
# import seaborn as sns
# from ridge_regressor.ridge_regressor_training import ridge_training
# from randon_forest.random_forest_training import random_forest_training
# from xg_boost.xgboost_training import xgboost_training

# Linha corrigida: Adiciona o diretório raiz 'COMPRESSIVE_SENSING' ao sys.path
# '...' sobe dois níveis, chegando no diretório raiz do projeto.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 
# Note o '..', '..' acima

from mlp_training import mlp_training
from svm_with_rbf_training import svc_rbf_training
# from adaboost.adaboost_training import adaboost_training
# from catboost.catboost_training import catboost_training

# Carregue os dados
df = pd.read_csv('compressed_data_classification/data.csv')

# df = df.drop(columns=['Unnamed: 0', 'F3_norm'])

# Variável alvo
y = df['target']
X = df.drop(columns=['target', 'Unnamed: 0'])

# Encoder das labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nMapeamento de Classes:", class_mapping)

# Divisão em treino e teste para avaliação final depois do ajuste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

# Técnica de compressão de dados energy/topk/pca/pure_alpha/random_mesurements
# techniques = ['energy','topk','pca','pure_alpha','random_mesurements', 'original_data']
techniques = ['random_mesurements', 'original_data']

# Console para exibir status
console = Console()

# Treinar usando os diferentes métodos de treinamento e salvar cada um
for technique in techniques:
    # Modelos
    # print()
    # print("RIDGE REGRESSOR")
    # rr_info = ridge_training(X_train, y_train, X_test, y_test, X)
    # print()
    # print("RANDOM FOREST")
    # rf_info = random_forest_training(X_train, y_train, X_test, y_test, X)
    # print()
    # print("XGBOOST")
    # xgb_info = xgboost_training(X_train, y_train, X_test, y_test, X)
    with Status(f"[bold green]Treinando MLP para {technique}...[/]", spinner="dots"):
        mlp_info = mlp_training(X_train, y_train, X_test, y_test, X, technique, label_encoder)
    print()
    with Status(f"[bold green]Treinando SVM com RBF para {technique}...[/]", spinner="dots"):
        svm_rbf = svc_rbf_training(X_train, y_train, X_test, y_test, X, technique, label_encoder)
    print()
    # print("ADABOOST")
    # ada_info = adaboost_training(X_train, y_train, X_test, y_test, X)

    # Montando tabela de comparação entre os modelos
    # df = pd.DataFrame([rr_info, rf_info, xgb_info, mlp_info])
    # df = pd.DataFrame([rr_info, rf_info, mlp_info, ada_info])
    df = pd.DataFrame([mlp_info, svm_rbf])
    # df = df.drop('melhores_parametros')

    # print(df)
    # df.to_excel(f'sensor_potencial_hidrico_ai/model/resultados_modelos_{technique}.xlsx')

# Mostrar todos os gráficos
# plt.show()