import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# 1. Parâmetros
n = 256  # tamanho do sinal original
k = 10   # número de coeficientes não-nulos (esparsidade)
m = 100  # número de medições (m << n)

# 2. Geração do sinal esparso no domínio de Fourier
np.random.seed(0)
x_freq = np.zeros(n, dtype=complex)
indices = np.random.choice(n, k, replace=False)
x_freq[indices] = np.random.randn(k) + 1j*np.random.randn(k)

# Sinal no tempo (transformada inversa)
x = np.fft.ifft(x_freq).real

# 3. Medições com matriz aleatória (m x n)
Phi = np.random.randn(m, n) / np.sqrt(m)
y = Phi @ x  # medidas obtidas

# 4. Reconstrução via L1 minimização (Basis Pursuit)
x_rec = cp.Variable(n)
objective = cp.Minimize(cp.norm1(x_rec))
constraints = [Phi @ x_rec == y]
problem = cp.Problem(objective, constraints)
problem.solve()

# 5. Visualização
plt.figure(figsize=(12,5))
plt.plot(x, label='Sinal Original')
plt.plot(x_rec.value, '--', label='Sinal Reconstruído')
plt.legend()
plt.title('Compressive Sensing - Reconstrução de Sinal')
plt.grid()
plt.show()
