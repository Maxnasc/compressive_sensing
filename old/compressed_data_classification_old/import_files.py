import kagglehub
import scipy.io as sio
import os
import numpy as np

# Download latest version
path = kagglehub.dataset_download("sumairaziz/seed-power-quality-disturbance-dataset")
print("Path to dataset files:", path)

# Localizar o arquivo .mat
file_name = '5Kfs_1Cycle_50f_1000Sam_1A.mat'
file_path = os.path.join(path, file_name)
print("Caminho completo do arquivo .mat:", file_path)

mat_contents = sio.loadmat(file_path)

try:
    # Tenta usar o nome comum para o array dentro do arquivo .mat
    data_key = 'data' # Este é um palpite comum, mas pode variar.
    # Se o nome da chave for o nome do arquivo, seria: '5Kfs_1Cycle_50f_1000Sam_1A'
    
    # Vamos assumir que o array principal se chama 'DATASET_SEED' (ou verificar as chaves)
    # Para datasets de QEE, a chave é frequentemente o nome do arquivo ou 'data'
    
    # Verificação das chaves:
    print("Chaves encontradas no arquivo .mat:", mat_contents.keys()) 
    
    # Baseado em datasets similares, o nome da chave pode ser 'x' ou o nome do arquivo.
    # Vamos tentar uma chave comum ou a única grande matriz.
    
    # Assumindo a chave correta após inspecionar o .mat:
    # Se a chave for o nome do arquivo:
    # all_signals = mat_contents['5Kfs_1Cycle_50f_1000Sam_1A'] 
    
    # Para o SEED, a chave é tipicamente o nome do arquivo. 
    # **Atenção:** Se der erro, verifique as chaves de `mat_contents`
    all_signals = mat_contents['5Kfs_1Cycle_50f_1000Sam_1A']
    
    # all_signals terá dimensão (17 x 1000 x 100) ou (1000 x 100 x 17) dependendo de como foi salvo.
    print(f"\nShape da matriz de dados carregada: {all_signals.shape}")

except KeyError as e:
    print(f"Erro ao encontrar a chave principal dos dados: {e}. Verifique as chaves em 'mat_contents.keys()'")
    
a=1