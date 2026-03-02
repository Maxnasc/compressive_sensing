"""
Module: analysis/analyse_signals.py

Signal visualization and analysis for electrical disturbances.

This script loads electrical disturbance data and creates visualizations showing
representative signals from each disturbance class. It helps understand the
characteristics of different electrical disturbances.

The module:
- Loads signal data from CSV
- Groups signals by disturbance class
- Plots one representative signal per class
- Saves visualization to file

Author: Maxnasc7
License: MIT
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Importado para cálculos de layout

# --- Configurações Iniciais ---
plots_path = "compressed_data_classification/plots"
data_file = "compressed_data_classification/data.csv"

# 1. Leitura e Pré-processamento Único de Dados
signals = pd.read_csv(data_file)

# Remove as colunas que não são dados do sinal uma única vez (melhora a eficiência)
signal_data = signals.drop(columns=["Unnamed: 0"]) # Mantendo 'target' por enquanto para seleção

# Identifica as classes e calcula o número de gráficos necessários
classes = signal_data["target"].unique()
num_classes = len(classes)

# Define o layout da grade (ex: 3 colunas, número de linhas calculado)
ncols = 3 # Alterado par a versão reduzida com 3 imagens
# nrows = int(np.ceil(num_classes / ncols))
nrows = 1

# --- Criação e Plotagem dos Subgráficos ---

# 2. Cria a figura e a grade de eixos
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 4))
# fig.suptitle(
#     "Exemplo de Distúrbios por Classe (Uma Amostra por Classe)", fontsize=16
# )

# Acha uma amostra para cada classe (a primeira ocorrência)
# Itera sobre a matriz de Eixos de forma linear (axs.flatten())
for ax, class_name in zip(axs.flatten(), classes):
    
    # Seleciona a primeira linha que pertence a esta classe
    # .loc[] retorna um booleano, .iloc[0] pega a primeira linha True
    sample_series = signal_data.loc[signal_data["target"] == class_name].iloc[0]

    # Remove 'target' desta Series específica antes de plotar
    signal_values = sample_series.drop("target")

    # Plota a Series no Eixo atual (ax)
    ax.plot(signal_values.values, linewidth=1)
    
    # Configurações do Eixo
    ax.set_title(class_name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Oculta as marcas e etiquetas do Eixo X (limpeza visual)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    # OU usando o comando que você já usava: ax.get_xaxis().set_visible(False)

# 3. Limpa Eixos Não Utilizados (se houverem)
num_plots_made = len(classes)
num_total_axes = nrows * ncols

# for i in range(num_plots_made, num_total_axes):
#     fig.delaxes(axs.flatten()[i])

for i in range(3, num_total_axes):
    fig.delaxes(axs.flatten()[i])

# 4. Ajuste e Exibição
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta layout, deixando espaço para suptitle

# Salvando a figura (descomente para usar)
plt.savefig(f'{plots_path}/exemplo_disturbios_3.png', bbox_inches='tight')

plt.show()