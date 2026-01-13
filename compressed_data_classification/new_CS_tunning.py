import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
from scipy.interpolate import interp1d

# ----------------------------
# CONFIGURAÇÃO E GRADE DE PARÂMETROS
# ----------------------------
DATA_DIR = "compressed_data_classification"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

PARAM_GRID = {
    'SAMPLE_M': [30, 40, 50, 60],            # M amostras
    'WAVELET': ['db4', 'db8', 'sym4'],      # Famílias Wavelet
    'WAVELET_LEVEL': [2, 3, 4],             # Níveis de decomposição
    'METHOD': ['OMP', 'LASSO'],             # Algoritmos
}

# Hiperparâmetros específicos
OMP_K_CANDIDATES = [10, 20, 30, 40]
LASSO_ALPHA_CANDIDATES = [1e-4, 1e-3, 1e-2]

# ----------------------------
# FUNÇÕES DE UTILITÁRIO E MÉTRICAS
# ----------------------------

def mse(x, y):
    return float(np.mean((x - y) ** 2))

def psnr(x_original, x_reconstruido):
    # Para sinais normalizados ou de energia unitária, MAX_I costuma ser 1.0 ou np.max(x_original)
    max_val = np.max(np.abs(x_original))
    if max_val == 0: max_val = 1.0
    err = mse(x_original, x_reconstruido)
    if err == 0: return float('inf')
    return 10.0 * np.log10((max_val**2) / err)

def correlation_coefficient(x_orig, x_rec):
    if x_orig.ndim > 1: x_orig = x_orig.flatten()
    if x_rec.ndim > 1: x_rec = x_rec.flatten()
    return np.corrcoef(x_orig, x_rec)[0, 1]

def load_csv_signal(path="compressed_data_classification/data.csv"):
    try:
        df = pd.read_csv(path)
        X = df.filter(regex=r"^s\d+$").values 
        y_labels = df["target"].values
        N_signal = X.shape[1]
        return X, y_labels, N_signal
    except FileNotFoundError:
        print(f"[Erro] Arquivo {path} não encontrado.")
        return None, None, None

# ----------------------------
# NÚCLEO DO COMPRESSIVE SENSING
# ----------------------------

def build_wavelet_matrix_and_shapes(N, wavelet, level):
    zero = np.zeros(N)
    coeffs_template = pywt.wavedec(zero, wavelet, level=level)
    shapes = [c.size for c in coeffs_template]
    K = sum(shapes)
    Psi = np.zeros((N, K))
    idx = 0
    for band_idx, band_len in enumerate(shapes):
        for j in range(band_len):
            coeffs = [np.copy(a) for a in coeffs_template]
            coeffs[band_idx][j] = 1.0
            atom = pywt.waverec(coeffs, wavelet)[:N]
            Psi[:, idx] = atom
            idx += 1
    return Psi, shapes

def construct_Phi_from_mask_or_meas_idx(meas_idx, N):
    M = meas_idx.size
    Phi = np.zeros((M, N))
    for i, idx in enumerate(meas_idx):
        Phi[i, idx] = 1.0
    return Phi, meas_idx

def build_A_and_normalize(Phi, Psi_concat):
    A = Phi.dot(Psi_concat)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_norm = A / col_norms
    return A, A_norm, col_norms

def solve_omp(A_norm, y, k):
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    model.fit(A_norm, y)
    return model.coef_.copy()

def solve_lasso(A_norm, y, alpha):
    model = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False)
    model.fit(A_norm, y)
    return model.coef_.copy()

def alpha_to_reconstruction(alpha_full, N, K, Psi_wave, shapes, wavelet_name):
    alpha_I = alpha_full[:N]
    alpha_wave = alpha_full[N:]
    x_rec = alpha_I + Psi_wave.dot(alpha_wave)
    return x_rec

# ----------------------------
# GRID SEARCH E PLOT
# ----------------------------

def run_extensive_grid_search(X_matrix, N_len):
    # Seleciona 5 sinais aleatórios para o teste de robustez
    random_indices = random.sample(range(X_matrix.shape[0]), 5)
    sinais_teste = X_matrix[random_indices, :]
    
    keys = PARAM_GRID.keys()
    base_combinations = list(itertools.product(*PARAM_GRID.values()))
    
    results_list = []
    print(f"Iniciando busca extensiva: {len(base_combinations)} combinações básicas...")

    for combo in base_combinations:
        config_base = dict(zip(keys, combo))
        sub_params = OMP_K_CANDIDATES if config_base['METHOD'] == 'OMP' else LASSO_ALPHA_CANDIDATES
        
        for val in sub_params:
            config = config_base.copy()
            if config['METHOD'] == 'OMP': config['K'] = val
            else: config['alpha'] = val
            
            m_psnr, m_cc = [], []
            
            for i in range(5):
                sinal_orig = sinais_teste[i, :]
                meas_idx = np.sort(np.random.choice(N_len, size=config['SAMPLE_M'], replace=False))
                Phi, _ = construct_Phi_from_mask_or_meas_idx(meas_idx, N_len)
                Psi_w, shapes = build_wavelet_matrix_and_shapes(N_len, config['WAVELET'], config['WAVELET_LEVEL'])
                
                I = np.eye(N_len)
                Psi_concat = np.concatenate([I, Psi_w], axis=1)
                _, A_norm, col_norms = build_A_and_normalize(Phi, Psi_concat)
                
                y = Phi.dot(sinal_orig)
                
                try:
                    if config['METHOD'] == 'OMP':
                        coef = solve_omp(A_norm, y, config['K']) / col_norms
                    else:
                        coef = solve_lasso(A_norm, y, config['alpha']) / col_norms
                    
                    x_rec = alpha_to_reconstruction(coef, N_len, Psi_w.shape[1], Psi_w, shapes, config['WAVELET'])
                    m_psnr.append(psnr(sinal_orig, x_rec))
                    m_cc.append(correlation_coefficient(sinal_orig, x_rec))
                except:
                    continue
            
            if m_psnr:
                avg_psnr, avg_cc = np.mean(m_psnr), np.mean(m_cc)
                res = {**config, 'avg_psnr': avg_psnr, 'avg_cc': avg_cc}
                results_list.append(res)
                if avg_cc >= 0.97 and avg_psnr >= 25:
                    print(f"✅ Config Satisfatória: {config['METHOD']} | M={config['SAMPLE_M']} | CC={avg_cc:.4f}")

    df = pd.DataFrame(results_list)
    return df.sort_values(by='avg_cc', ascending=False)

def plot_best_result(x_orig, x_rec, meas_idx, y_samples, title):
    
    plt.figure(figsize=(12, 5))
    plt.plot(x_orig, label='Original', color='gray', alpha=0.5, linestyle='--')
    plt.plot(x_rec, label='Reconstruído', color='blue', linewidth=1.5)
    plt.scatter(meas_idx, y_samples, color='red', s=25, label='Amostras Aleatórias', zorder=5)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ----------------------------
# EXECUÇÃO PRINCIPAL
# ----------------------------

if __name__ == "__main__":
    X, labels, N = load_csv_signal()
    
    if X is not None:
        df_res = run_extensive_grid_search(X, N)
        
        print("\n--- TOP 5 CONFIGURAÇÕES ---")
        print(df_res[['METHOD', 'SAMPLE_M', 'WAVELET', 'avg_psnr', 'avg_cc']].head(5))
        
        # Testar a melhor configuração no primeiro sinal para visualização
        best = df_res.iloc[0]
        sinal_plot = X[0, :]
        
        # Re-executa apenas para o plot
        meas_idx = np.sort(np.random.choice(N, size=int(best['SAMPLE_M']), replace=False))
        Phi, _ = construct_Phi_from_mask_or_meas_idx(meas_idx, N)
        Psi_w, shapes = build_wavelet_matrix_and_shapes(N, best['WAVELET'], int(best['WAVELET_LEVEL']))
        I = np.eye(N)
        Psi_concat = np.concatenate([I, Psi_w], axis=1)
        _, A_norm, col_norms = build_A_and_normalize(Phi, Psi_concat)
        y_plot = Phi.dot(sinal_plot)
        
        if best['METHOD'] == 'OMP':
            c = solve_omp(A_norm, y_plot, int(best['K'])) / col_norms
        else:
            c = solve_lasso(A_norm, y_plot, best['alpha']) / col_norms
            
        x_rec_plot = alpha_to_reconstruction(c, N, Psi_w.shape[1], Psi_w, shapes, best['WAVELET'])
        
        titulo = f"Melhor Reconstrução: {best['METHOD']} (CC={best['avg_cc']:.4f}, PSNR={best['avg_psnr']:.2f}dB)"
        plot_best_result(sinal_plot, x_rec_plot, meas_idx, y_plot, titulo)