import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# 1. Criar sinal elétrico sintético (soma de senóides)
fs = 500  # Hz
t = np.linspace(0, 1, fs, endpoint=False)
sinal = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# 2. Construir dicionário de senóides (senos e cossenos com várias frequências)
freqs = np.arange(1, 201)  # de 1Hz até 200Hz
D = np.array([np.sin(2*np.pi*f*t) for f in freqs] +
             [np.cos(2*np.pi*f*t) for f in freqs]).T  # shape: (fs, 2*len(freqs))

# 3. Codificação esparsa via Lasso (L1 regularização)
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(D, sinal)
coef = lasso.coef_

# 4. Reconstrução do sinal
sinal_rec = D @ coef

# 5. Visualização
plt.figure(figsize=(12,5))
plt.plot(t, sinal, label='Sinal Original')
plt.plot(t, sinal_rec, '--', label='Reconstrução Esparsa')
plt.legend()
plt.title('Codificação Esparsa de Sinal Elétrico')
plt.grid()
plt.show()

# 6. Espectro Esparso
plt.figure(figsize=(10,4))
plt.stem(np.abs(coef), use_line_collection=True)
plt.title('Coeficientes Esparsos (Frequências Detectadas)')
plt.xlabel('Índice do átomo')
plt.ylabel('Peso')
plt.grid()
plt.show()
