import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# === 1. CARREGAR SINAL ===
x = np.load('compressive_sensing/test_signal_250M.npy')  # Sinal original (ex: 50000 amostras)
x = x.astype(np.float64)
n = len(x)

# === 2. DEFINIR PARÂMETROS DE CS ===
K = 1000                     # Número de amostras esparsas
alpha_lasso = 0.001          # Regularização L1
random_state = 42

# === 3. GERAR DICIONÁRIO DE FOURIER ===
def fourier_dictionary(n):
    """Gera matriz D (Fourier) de dimensão n x n"""
    t = np.arange(n)
    D = np.zeros((n, n), dtype=complex)
    for k in range(n):
        D[:, k] = np.exp(2j * np.pi * k * t / n)
    # Separar real e imaginário (base real)
    D_real = np.hstack([D.real, D.imag])
    return D_real  # shape (n, 2n)

D = fourier_dictionary(n)  # D shape: (n, 2n)
D = D / np.linalg.norm(D, axis=0)  # Normalização por coluna

# === 4. GERAR AMOSTRAS ESPARSAS ===
np.random.seed(random_state)
indices = np.sort(np.random.choice(n, size=K, replace=False))
y = x[indices]              # medições
Phi = D[indices, :]         # sensing matrix (K, 2n)

# === 5. RESOLVER LASSO ===
lasso = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=10000)
lasso.fit(Phi, y)
alpha = lasso.coef_         # shape: (2n,)

# === 6. RECONSTRUIR O SINAL ===
x_hat = D @ alpha           # shape: (n,)

# === 7. PLOTAR RESULTADO ===
plt.figure(figsize=(16, 5))
plt.plot(x, label='Sinal Original')
plt.plot(x_hat.real, label='Reconstruído (CS - Fourier)', linestyle='--')
plt.scatter(indices, y, color='red', s=5, label='Amostras Esparsas')
plt.legend()
plt.xlabel('Tempo (amostras)')
plt.ylabel('Amplitude')
plt.title('Reconstrução via Compressive Sensing com Dicionário de Fourier')
plt.grid(True)
plt.tight_layout()
plt.show()
