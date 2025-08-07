import numpy as np

# Passo 1: Gerar sinal elétrico puro (ex: 60 Hz, 1 segundo, 1024 amostras)
fs = int(250e6)  # taxa de amostragem
t = np.linspace(0, 1, fs, endpoint=False)
frequencia = 60  # Hz
sinal_original = np.sin(2 * np.pi * frequencia * t)

np.save('compressive_sensing/test_signal_250M.npy')