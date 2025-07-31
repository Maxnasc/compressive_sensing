import pandas as pd
import numpy as np
import os
from sklearn.decomposition import DictionaryLearning
from joblib import dump

def get_windows(path: str, window_size=50000):
    data = pd.read_csv(path, skiprows=14)
    data.columns = ['TIME', 'CH1', 'CH2', 'CH3', 'CH4']
    signal = data['CH1'].values
    
    # Normalização dos dados
    signal = (signal - np.mean(signal))/ np.std(signal)
    
    # Dividir em janelas
    windows = [signal[i:i+window_size] for i in range(0, len(signal)-window_size, window_size)]
    return windows

def get_x_data():
    
    all_windows = []
    # Caminhar pela pasta principal
    for root, _, files in os.walk('data'):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                windows = get_windows(path=file_path)
                all_windows.append(windows)
    X = np.vstack(all_windows)
    
    return X

def train_dictionary(X, n_components=100):
    dict_learner = DictionaryLearning(n_components=n_components, alpha=1.0, max_iter=1000, verbose=True)
    D = dict_learner.fit(X).components_
    
    # Salvar o dicionário D
    dump(D, 'compressive_sensing/dictionary/trained_dictionary.npy')
    print('Dicionário salvo!')
    

if __name__ == "__main__":
    
    '''
    1- Dados CSV (CH1, CH3)
        ↓
    2 - Dividir em janelas
        ↓
    3 - Empilhar e normalizar
        ↓
    4 - Treinar dicionário (DictionaryLearning)
        ↓
    5 - Codificar novos sinais (Lasso ou OMP)
        ↓
    6 - Reconstrução do sinal
        ↓
    → Visualização ou Classificação de faltas
    '''
    
    # 1,2,3 Ler os arquivos csv e montar a matriz X para o treinamento do dicionário
    X = get_x_data()
    
    # 4 Treinamento do dicionário
    train_dictionary(X=X)