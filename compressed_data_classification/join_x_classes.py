import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

folder = 'compressed_data_classification/data_by_class'

default_path = os.path.join(folder, '*.csv')

csv_files = glob.glob(default_path) # Retorna uma lista com todos os csv

sufix = '.csv'
y = [os.path.basename(file_path).removesuffix(sufix) for file_path in csv_files]
columns=[f's{i+1}' for i in range(100)]
data = pd.DataFrame()

for file_path in csv_files:
    # Ler o arquivo csv
    csv_data = pd.read_csv(file_path, header=None)
    
    if csv_data.shape[1] != len(columns):
        print(f"AVISO: {os.path.basename(file_path)} tem {csv_data.shape[1]} colunas, esperado {len(columns)}.")
        continue # Pular arquivos com número incorreto de colunas
    
    csv_data.columns = columns
    
    class_name = os.path.basename(file_path).removesuffix(sufix)
    csv_data['target'] = class_name
    
    data = pd.concat([data, csv_data], axis=0, ignore_index=True)
    
data.to_csv('compressed_data_classification/data.csv')
print('Dados completos salvos em: compressed_data_classification/data.csv')
print(data.head())

# Printando um sinal de amostra
first_line = data.iloc[0]
data_to_plot = first_line.drop('target', errors='ignore')

plt.figure(figsize=(12,5))
plt.plot(data_to_plot)
plt.title(f"Plot da Primeira Linha (Classe: {first_line['target']})")
plt.xlabel("Índice da Feature (s1, s2, ...)")
plt.ylabel("Valor")
plt.show()