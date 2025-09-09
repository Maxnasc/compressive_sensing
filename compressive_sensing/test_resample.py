import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from joblib import load

# Passo 1: importar o sinal elétrico
sinal_original = np.load('compressive_sensing/test_signal_250M.npy')

# Passo 2: Importar o dicionário treinado
D = load('compressive_sensing/dictionary/trained_dictionary.pkl')  # shape (n_atoms, N=50000)

# Passo 3: Amostragem esparsa
taxa_amostragem = 0.3
np.random.seed(42)
N = sinal_original.shape[0]
K = 100
mascara = np.zeros(N, dtype=bool)
mascara[np.random.choice(N, K, replace=False)] = True
indices_amostrados = np.where(mascara)[0]
y = sinal_original[mascara]  # amostras observadas

# Reduzir o dicionário para as posições observadas
D_reduzido = D[:, mascara]  # shape (n_atoms, num_amostras)

# Passo 4: Reconstrução com Lasso
lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(D_reduzido.T, y)  # Transpor para shape (num_amostras, n_atoms)
coef = lasso.coef_

# Reconstrução do sinal completo
sinal_reconstruido = D.T @ coef  # shape (50000,)

# Passo 5: Visualização
plt.figure(figsize=(12, 6))
plt.plot(sinal_original, label='Sinal Original', linewidth=2)
plt.plot(sinal_reconstruido, '--', label='Reconstruído (CS)', linewidth=2)
# plt.scatter(indices_amostrados, y, color='red', label='Amostras Esparsas', zorder=5)
plt.title('Reconstrução de Sinal Elétrico via Compressive Sensing')
plt.xlabel('Tempo (amostras)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
