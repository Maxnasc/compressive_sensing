import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from joblib import load

# Passo 1: importar o sinal elétrico
sinal_original = np.load('compressive_sensing/test_signal_250M.npy')

# Passo 2: Criar um dicionário D com átomos senoidais e cosenoidais (base de Fourier)
def gerar_dicionario_fourier(n_componentes, tamanho_sinal):
    D = []
    for k in range(1, n_componentes + 1):
        D.append(np.sin(2 * np.pi * k * np.arange(tamanho_sinal) / tamanho_sinal))
        D.append(np.cos(2 * np.pi * k * np.arange(tamanho_sinal) / tamanho_sinal))
    D = np.array(D)
    # Normalizar os átomos
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    return D

n_componentes = 50
D = load('compressive_sensing/dictionary/trained_dictionary.pkl')

# Passo 3: Amostragem esparsa (ex: 30% das amostras)
taxa_amostragem = 0.3
np.random.seed(42)
N = 50000
K = 1024
mascara = np.zeros(N, dtype=bool)
mascara[np.random.choice(N, K, replace=False)] = True
indices_amostrados = np.where(mascara)[0]
y = sinal_original[mascara]  # amostras observadas

# Reduzir o dicionário para as posições observadas
D_reduzido = D[:, mascara].T  # shape (num_amostras, num_atom)

# Passo 4: Reconstrução com Lasso
lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(D_reduzido, y)
coef = lasso.coef_

# Reconstrução do sinal completo
sinal_reconstruido = D.T @ coef  # shape (1024,)

# Passo 5: Visualização
plt.figure(figsize=(12, 6))
plt.plot(sinal_original, label='Sinal Original', linewidth=2)
plt.plot(sinal_reconstruido, '--', label='Reconstruído (CS)', linewidth=2)
plt.scatter(indices_amostrados, y, color='red', label='Amostras Esparsas', zorder=5)
plt.title('Reconstrução de Sinal Elétrico via Compressive Sensing')
plt.xlabel('Tempo (amostras)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
