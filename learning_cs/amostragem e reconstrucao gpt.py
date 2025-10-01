# Reconstrução CS com dicionário concatenado (Identidade + Wavelet)
# Pipeline para reconstrução de sinais esparsos com preservação de spikes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
import pywt
from numpy.linalg import norm
import os


# ---------------------------------------------------------
# Constrói a matriz de base wavelet (cada coluna = átomo)
# ---------------------------------------------------------
def build_wavelet_matrix(N, wavelet='db4', level=None):
    if level is None:
        level = pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len)
    zero_signal = np.zeros(N)
    coeffs_template = pywt.wavedec(zero_signal, wavelet, level=level)
    idx_map = []
    for i, arr in enumerate(coeffs_template):
        for j in range(arr.size):
            idx_map.append((i, j))
    Psi = np.zeros((N, len(idx_map)))
    for k, (iarr, jpos) in enumerate(idx_map):
        coeffs = [np.copy(a) for a in coeffs_template]
        coeffs[iarr][jpos] = 1.0
        atom = pywt.waverec(coeffs, wavelet)
        atom = atom[:N]
        Psi[:, k] = atom
    return Psi


# ---------------------------------------------------------
# Reconstrução usando OMP com dicionário concatenado
# (Identidade + Wavelet)
# ---------------------------------------------------------
def reconstruct_with_concat_omp(x_full, mask, wavelet='db4',
                                level=None, n_nonzero_coefs=50):
    N = x_full.shape[0]
    meas_idx = np.where(mask)[0]
    M = meas_idx.size

    # Matriz de amostragem
    Phi = np.zeros((M, N))
    for i, idx in enumerate(meas_idx):
        Phi[i, idx] = 1.0

    # Dicionário concatenado
    I = np.eye(N)
    Psi_wave = build_wavelet_matrix(N, wavelet=wavelet, level=level)
    Psi = np.concatenate([I, Psi_wave], axis=1)

    # Sistema de medição
    A = Phi.dot(Psi)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_norm = A / col_norms

    y = x_full[meas_idx]

    omp = OrthogonalMatchingPursuit(
        n_nonzero_coefs=n_nonzero_coefs
    )
    omp.fit(A_norm, y)
    alpha = omp.coef_ / col_norms
    x_rec = Psi.dot(alpha)
    return x_rec, alpha


# ---------------------------------------------------------
# Detecção de spikes baseada em mediana (MAD)
# ---------------------------------------------------------
def detect_spikes(x, k=5.0):
    med = np.median(x)
    sigma = np.median(np.abs(x - med)) / 0.6745 + 1e-12
    threshold = med + k * sigma
    spike_idx = np.where(np.abs(x - med) > k * sigma)[0]
    return spike_idx, threshold


# ---------------------------------------------------------
# Carrega ou gera um sinal exemplo
# ---------------------------------------------------------
def load_or_make_example(path_dir='.'):
    files = os.listdir(path_dir)
    x = None
    mask = None

    # tenta carregar arquivos comuns
    for name in ['x_original.npy', 'x.npy', 'signal.npy', 'sinal.npy', 'FaltamonofasicaAT45km20ohm200khz.mat']:
        p = os.path.join(path_dir, name)
        if os.path.exists(p):
            x = np.load(p)
            break

    for name in ['mask.npy', 'meas_mask.npy', 'mask_meas.npy']:
        p = os.path.join(path_dir, name)
        if os.path.exists(p):
            mask = np.load(p).astype(bool)
            break

    # tenta npz
    npz_candidates = [f for f in files if f.endswith('.npz')]
    if x is None and npz_candidates:
        data = np.load(os.path.join(path_dir, npz_candidates[0]))
        for k in ['x', 'x_original', 'signal', 'sinal']:
            if k in data:
                x = data[k]
                break

    # gera exemplo sintético se não encontrar nada
    if x is None:
        N = 1024
        t = np.arange(N)
        base = 0.5 * np.sin(2 * np.pi * 5 * t / N) + \
               0.2 * np.sin(2 * np.pi * 20 * t / N)
        x = base.copy()
        rng = np.random.RandomState(42)
        spike_positions = rng.choice(np.arange(100, N - 100),
                                     size=8, replace=False)
        for p in spike_positions:
            x[p:p + 3] += rng.uniform(2, 5)

    if mask is None:
        N = x.shape[0]
        rng = np.random.RandomState(1)
        M = int(0.4 * N)  # 40% de amostras
        idx = rng.choice(N, size=M, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True

    return x, mask


# ---------------------------------------------------------
# Exemplo de execução
# ---------------------------------------------------------
if __name__ == "__main__":
    x_original, mask = load_or_make_example('.')
    N = x_original.size
    print("Signal length N =", N)

    wavelet = 'db4'
    expected_spikes = 8
    n_nonzero = expected_spikes + 30

    x_rec, alpha = reconstruct_with_concat_omp(
        x_original, mask,
        wavelet=wavelet, level=None,
        n_nonzero_coefs=n_nonzero
    )

    meas_idx = np.where(mask)[0]

    plt.figure()
    plt.plot(x_original)
    plt.scatter(meas_idx, x_original[meas_idx], s=10)
    plt.title("Original signal and measurement locations")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")

    plt.figure()
    plt.plot(x_original, label="original")
    plt.plot(x_rec, label="reconstructed")
    plt.title("Original vs Reconstructed (OMP with concatenated dictionary)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.figure()
    plt.plot(x_original - x_rec)
    plt.title("Reconstruction error (original - recon)")
    plt.xlabel("Sample index")
    plt.ylabel("Error")

    spikes_rec, thresh = detect_spikes(x_rec, k=5.0)
    print("Detected spikes in reconstructed signal (indices):", spikes_rec[:20])
    print("Threshold used:", thresh)

    plt.show()
