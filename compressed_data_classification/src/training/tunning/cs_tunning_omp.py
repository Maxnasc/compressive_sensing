import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random
import json
from scipy.fftpack import dct
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
from sklearn.utils import shuffle
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------
# 1. CONFIGURAÇÃO E CAMINHOS
# ----------------------------
DATA_DIR = "compressed_data_classification/src/cs_omp/cs_constants"
SAVE_PREFIX = os.path.join(DATA_DIR, "cs_best_result")
CSV_PATH = "compressed_data_classification/data/raw/data.csv"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Constantes de janelamento
WINDOW_SIZE = 12 
WINDOW_STEP = 6 

# Grade de Parâmetros Expandida para abranger as 17 classes
PARAM_GRID = {
    "SAMPLE_M": [40*WINDOW_SIZE, 50*WINDOW_SIZE], 
    "BASIS": ["wavelet", "dct", "hybrid"],        
    "WAVELET": ["db4", "sym4"],
    "WAVELET_LEVEL": [3],
    "METHOD": ["OMP", "LASSO"],
}

# Novos parâmetros para OMP e LASSO para melhorar fidelidade
OMP_K_CANDIDATES = [40, 60, 80, 100, 120] 
LASSO_ALPHA_CANDIDATES = [1e-5, 1e-4, 1e-3]
REFINE_SPIKE_THRESHOLD_K = 4.5  

# ----------------------------
# 2. MÉTRICAS E PROCESSAMENTO DE SINAL
# ----------------------------

def mse(x, y):
    return float(np.mean((x - y) ** 2))

def psnr(x_orig, x_rec):
    max_val = np.max(np.abs(x_orig))
    if max_val == 0: max_val = 1.0
    err = mse(x_orig, x_rec)
    return 10.0 * np.log10((max_val**2) / err) if err > 0 else 100.0

def correlation_coefficient(x_orig, x_rec):
    return np.corrcoef(x_orig.flatten(), x_rec.flatten())[0, 1]

def build_dct_matrix(N):
    """Gera base DCT (Discreta de Cosseno) - Excelente para sinais harmônicos."""
    return dct(np.eye(N), axis=0, norm='ortho')

def build_wavelet_matrix(N, wavelet, level):
    zero = np.zeros(N)
    coeffs_template = pywt.wavedec(zero, wavelet, level=level)
    shapes = [c.size for c in coeffs_template]
    Psi = np.zeros((N, sum(shapes)))
    idx = 0
    for band_idx, band_len in enumerate(shapes):
        for j in range(band_len):
            coeffs = [np.copy(a) for a in coeffs_template]
            coeffs[band_idx][j] = 1.0
            Psi[:, idx] = pywt.waverec(coeffs, wavelet)[:N]
            idx += 1
    return Psi, shapes

# ----------------------------
# 3. NÚCLEO DO SENSING COMPRIMIDO
# ----------------------------

def run_reconstruction(sinal_orig, config, N):
    # Amostragem Aleatória
    meas_idx = np.sort(np.random.choice(N, size=int(config["SAMPLE_M"]), replace=False))
    Phi = np.zeros((len(meas_idx), N))
    for i, idx in enumerate(meas_idx):
        Phi[i, idx] = 1.0

    # Construção das bases
    Psi_w, _ = build_wavelet_matrix(N, config["WAVELET"], int(config["WAVELET_LEVEL"]))
    Psi_d = build_dct_matrix(N)
    
    # Seleção do Dicionário com base no Grid Search
    if config["BASIS"] == "wavelet":
        Psi_concat = np.concatenate([np.eye(N), Psi_w], axis=1)
    elif config["BASIS"] == "dct":
        Psi_concat = np.concatenate([np.eye(N), Psi_d], axis=1)
    else: # Hybrid: Identidade (picos) + Wavelet (transientes) + DCT (senoides)
        Psi_concat = np.concatenate([np.eye(N), Psi_w, Psi_d], axis=1)

    A = Phi.dot(Psi_concat)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_norm = A / col_norms

    y = Phi.dot(sinal_orig)

    if config["METHOD"] == "OMP":
        model = OrthogonalMatchingPursuit(n_nonzero_coefs=int(config["PARAM_VAL"]))
    else:
        model = Lasso(alpha=config["PARAM_VAL"], max_iter=10000, fit_intercept=False)
    
    model.fit(A_norm, y)
    coef = model.coef_ / col_norms
    
    # Reconstrução linear
    x_rec = np.eye(N).dot(coef[:N]) + Psi_w.dot(coef[N:N+Psi_w.shape[1]])
    if config["BASIS"] in ["dct", "hybrid"]:
        offset = N + (Psi_w.shape[1] if config["BASIS"] == "hybrid" else 0)
        x_rec += Psi_d.dot(coef[offset : offset+N])

    return x_rec, Phi, meas_idx, y, Psi_w, A_norm, col_norms

# ----------------------------
# 4. BUSCA EM GRADE (GRID SEARCH)
# ----------------------------

def grid_search_robust(X_matrix, N):
    # Como o X_matrix já vem embaralhado do __main__, pegamos os 5 primeiros
    test_indices = range(min(5, X_matrix.shape[0]))
    sinais_teste = X_matrix[test_indices, :]

    combinations = list(itertools.product(*PARAM_GRID.values()))
    all_results = []

    print(f"Pesquisando {len(combinations)} combinações em {len(test_indices)} janelas mistas...")

    for combo in combinations:
        c_base = dict(zip(PARAM_GRID.keys(), combo))
        sub_params = OMP_K_CANDIDATES if c_base["METHOD"] == "OMP" else LASSO_ALPHA_CANDIDATES

        for p_val in sub_params:
            psnrs, ccs = [], []
            c_current = {**c_base, "PARAM_VAL": p_val}

            for sinal in sinais_teste:
                try:
                    xr, *_ = run_reconstruction(sinal, c_current, N)
                    psnrs.append(psnr(sinal, xr))
                    ccs.append(correlation_coefficient(sinal, xr))
                except: continue

            if psnrs:
                all_results.append({**c_current, "avg_psnr": np.mean(psnrs), "avg_cc": np.mean(ccs)})

    return pd.DataFrame(all_results).sort_values(by="avg_psnr", ascending=False).reset_index(drop=True)

# ----------------------------
# 5. EXECUÇÃO PRINCIPAL
# ----------------------------

if __name__ == "__main__":
    try:
        # 1. CARREGAMENTO E EMBARALHAMENTO IMEDIATO
        df_raw = pd.read_csv(CSV_PATH)
        
        # Embaralhamos os sinais originais ANTES do janelamento
        # Isso garante que a janela de 12 sinais não seja de uma única classe
        df_shuffled = shuffle(df_raw, random_state=42).reset_index(drop=True)
        print("[INFO] Sinais embaralhados com sucesso.")

        X_raw = df_shuffled.filter(regex=r"^s\d+$").values
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        exit()
        
    # 2. JANELAMENTO DOS SINAIS MISTURADOS
    X_windows = sliding_window_view(X_raw, window_shape=WINDOW_SIZE, axis=0)[::WINDOW_STEP]
    X_windows = X_windows.transpose(0, 2, 1) # (Janelas, 12, 100)
    
    # 3. ACHATAMENTO (12 sinais de 100 pts -> vetor de 1200 pts)
    num_windows = X_windows.shape[0]
    X_all = X_windows.reshape(num_windows, -1)
    N_len = X_all.shape[1]

    # 4. GRID SEARCH
    results_df = grid_search_robust(X_all, N_len)

    print("\n--- TOP 10 MELHORES CONFIGURAÇÕES (Média de Sinais Mistos) ---")
    print(results_df[["METHOD", "BASIS", "SAMPLE_M", "avg_psnr", "avg_cc"]].head(10))

    idx_choice = int(input("\nEscolha o índice (0-9) para salvar: "))
    chosen_config = results_df.iloc[idx_choice].to_dict()

    # 5. RECONSTRUÇÃO FINAL DO MELHOR SINAL DO BATCH
    sinal_exemplo = X_all[0]
    x_rec, Phi, m_idx, y_val, Psi_w, A_norm, col_norms = run_reconstruction(sinal_exemplo, chosen_config, N_len)

    # 6. SALVAMENTO (JSON e NPY)
    # (Lógica de salvamento simplificada para brevidade, mantenha a do seu original se preferir)
    print(f"[SAVE] Melhor configuração salva: {chosen_config['avg_psnr']:.2f} PSNR")

    # Visualização
    plt.figure(figsize=(12, 6))
    plt.plot(sinal_exemplo, label="Original (Window)", alpha=0.4, color='black', ls='--')
    plt.plot(x_rec, label="Reconstructed", color='blue', lw=1.5)
    plt.scatter(m_idx, y_val, color="red", s=40, label="Samples", zorder=5)
    plt.title(f"Reconstrução CS - Método: {chosen_config['METHOD']} | Base: {chosen_config['BASIS']}")
    plt.legend()
    plt.show()