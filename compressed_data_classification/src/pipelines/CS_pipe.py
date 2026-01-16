# pipeline_cs_full.py
# Pipeline completo: carga -> amostragem -> dicionário concatenado -> OMP/Lasso -> refinamento de picos -> métricas/plots
# Requisitos: numpy, matplotlib, scikit-learn, pywt, mplcursors (opcional para hover)
# Execute no Jupyter (VSCode): antes rode %matplotlib widget para interatividade

import os
from pathlib import Path
import pickle
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
import json
import mplcursors
import scipy.io as sio
from scipy.interpolate import interp1d
import pandas as pd
import kagglehub
from sklearn.model_selection import cross_val_score

# ----------------------------
# CONFIG (ajuste estes parâmetros)
# ----------------------------
DATA_DIR = "compressed_data_classification/src/cs/cs_constants"  # onde procurar sinal e salvar resultados

# Importando os dados de configuração do Lasso
with open('compressed_data_classification/src/cs/results/metrics/metricsbest_cs_tune_metrics.json', 'r') as f:
    cs_params = json.load(f)
    
cs_conf_params = cs_params.get('config_parameters')
SAMPLE_M = cs_conf_params.get('SAMPLE_M')  # número de amostras M se não houver Phi/mask
WAVELET = cs_conf_params.get('WAVELET')  # wavelet para parte wavelet
WAVELET_LEVEL = cs_conf_params.get('WAVELET_LEVEL')  # nível de decomposição (tente 1..4)
METHODS = ["LASSO"]  # quais métodos testar
OMP_candidates = [
    10,
    20,
    30,
    50,
    60,
]  # valores de n_nonzero_coefs para testar (ajuste conforme k90)
LASSO_alphas = cs_conf_params.get('PARAM_VAL')  # alphas para Lasso
REFINE_SPIKE_THRESHOLD_K = 4.5  # k para MAD threshold de picos
SAVE_PREFIX = os.path.join(DATA_DIR, "cs_result")  # prefixo para salvar outputs

RANDOM_SEED = 42
# ----------------------------

np.random.seed(RANDOM_SEED)

def plot_alpha(alpha_full, N):
    plt.figure(figsize=(14,4))
    plt.stem(alpha_full)
    plt.title("Coeficientes α (Fourier + Wavelet)")
    plt.xlabel("Índice do coeficiente")
    plt.ylabel("Amplitude")
    plt.axvline(N, color='red', linestyle='--', linewidth=2)
    plt.text(N+5, max(alpha_full)*0.8, "Início da base Wavelet", color='red')
    plt.grid(True)
    # plt.show()
    
def plot_sparsity_pattern(alpha_full):
    nnz = np.nonzero(alpha_full)[0]

    plt.figure(figsize=(14,2))
    plt.scatter(nnz, np.zeros_like(nnz), marker='|', s=200)
    plt.title("Padrão de Esparsidade de α")
    plt.xlabel("Índice do coeficiente")
    plt.yticks([])
    plt.grid(True)
    # plt.show()
    
def plot_alpha_split(alpha_full, N):
    alpha_I = alpha_full[:N]
    alpha_wave = alpha_full[N:]

    fig, ax = plt.subplots(2, 1, figsize=(14,6))

    ax[0].stem(alpha_I)
    ax[0].set_title("Coeficientes α — Parte 1 (Fourier / Identidade)")
    ax[0].grid(True)

    ax[1].stem(alpha_wave)
    ax[1].set_title("Coeficientes α — Parte 2 (Wavelet)")
    ax[1].grid(True)

    plt.tight_layout()
    # plt.show()

def plot_alpha_hist(alpha_full):
    plt.figure(figsize=(7,4))
    plt.hist(alpha_full, bins=80)
    plt.title("Distribuição das Magnitudes de α")
    plt.xlabel("Valor do coeficiente")
    plt.ylabel("Frequência")
    plt.grid(True)
    # plt.show()

# ----------------------------
# UTIL: carregar sinal / Phi / mask / y
# ----------------------------

def load_dot_mat_signal():
    mat_contents = sio.loadmat("data\ATPdraw\1MHz_samples.mat")
    print(mat_contents.keys())

    s1 = mat_contents["s1"]
    s2 = mat_contents["s2"]
    s3 = mat_contents["s3"]
    tempo = mat_contents["tempo"].flatten()
    return s1[:, 0]
    print(len(tempo))

def save_cs_structures(path, Phi, A_norm, col_norms, N, shapes):
    data = {"Phi": Phi, "A_norm": A_norm, "col_norms": col_norms, "N": N, "shapes": shapes}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[SAVE] Estruturas CS salvas em: {path}")

def load_csv_signal():
    # Carregue todos os arquivos CSV e combine-os em uma matriz X (17000 x 100)
    # E crie o vetor de rótulos y_labels (17000 x 1)
    
    # ----------------------------------------------------
    # ESTA PARTE É UM EXEMPLO GENÉRICO DO SEED DATASET!
    # A implementação real dependerá da sua estrutura de arquivos.
    # Vou reescrever para o formato comum do SEED (17 arquivos CSV):
    # ----------------------------------------------------
    
    # Substitua 'caminho/para/dados' pelo caminho real onde estão os CSVs, 
    # que é o 'path' retornado pelo kagglehub.
    # Exemplo:
    # data_path = Path(kagglehub.dataset_download("sumairaziz/seed-power-quality-disturbance-dataset")) 
    
    # Exemplo de leitura de um único arquivo (assumindo que o seu 'data.csv' 
    # já é a concatenação com uma coluna 'target'):
    
    try:
        # Tenta carregar o DataFrame que causou o problema
        df = pd.read_csv("compressed_data_classification/data.csv")
    except FileNotFoundError:
        # Se não encontrar, você precisará carregar os 17 arquivos e concatenar
        print("[Alerta] 'data.csv' não encontrado. Garanta que você carregou o dataset.")
        return None, None, None # Retorna algo nulo para interromper

    # A dimensão X será (17000, 100)
    X = df.filter(regex=r'^s\d+$').values  # Seleciona apenas as colunas 's1' a 's100'
    
    # O rótulo é a coluna 'target'
    y_labels = df['target'].values
    
    # N é o comprimento de um ÚNICO sinal (100 amostras)
    N_signal = X.shape[1] 

    return X, y_labels, N_signal


def try_load_signal(data_dir=DATA_DIR):
    candidates = ["x_original.npy", "x.npy", "signal.npy", "sinal.npy"]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            x = np.load(p)
            print(f"[load] encontrado {p}")
            return x
    # tentar .npz
    for f in os.listdir(data_dir):
        if f.endswith(".npz"):
            data = np.load(os.path.join(data_dir, f))
            for key in ("x", "x_original", "signal", "sinal"):
                if key in data:
                    print(f"[load] encontrado {key} dentro de {f}")
                    return data[key]
    raise FileNotFoundError(
        "Não encontrei arquivo de sinal em /mnt/data. Coloque x_original.npy ou similar."
    )


def try_load_Phi_or_mask(data_dir=DATA_DIR):
    # procura Phi.npy, mask.npy, meas_idx.npy
    phi_path = os.path.join(data_dir, "Phi.npy")
    if os.path.exists(phi_path):
        Phi = np.load(phi_path)
        print(f"[load] Phi carregado de {phi_path}, shape = {Phi.shape}")
        return Phi
    mask_path = os.path.join(data_dir, "mask.npy")
    if os.path.exists(mask_path):
        mask = np.load(mask_path).astype(bool)
        print(f"[load] mask carregado de {mask_path}, sum(mask)={mask.sum()}")
        return mask
    meas_idx_path = os.path.join(data_dir, "meas_idx.npy")
    if os.path.exists(meas_idx_path):
        meas_idx = np.load(meas_idx_path).astype(int)
        print(f"[load] meas_idx carregado de {meas_idx_path}, len={len(meas_idx)}")
        return meas_idx
    print("[load] não foi encontrado Phi/mask/meas_idx no diretório")
    return None


# ----------------------------
# CONSTRUIR DICIONÁRIO WAVELET (matriz) E FORMATO DE BANDAS
# ----------------------------
def build_wavelet_matrix_and_shapes(N, wavelet=WAVELET, level=WAVELET_LEVEL):
    # Retorna: Psi_wave (N x K), shapes (lista de comprimentos por banda)
    if level is None:
        level = pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len)
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
            atom = pywt.waverec(coeffs, wavelet)
            atom = atom[:N]
            Psi[:, idx] = atom
            idx += 1
    return Psi, shapes


# ----------------------------
# Construir Phi a partir de mask / meas_idx / matriz
# ----------------------------
def construct_Phi_from_mask_or_meas_idx(mask_or_idx, N):
    if isinstance(mask_or_idx, np.ndarray) and mask_or_idx.dtype == bool:
        meas_idx = np.where(mask_or_idx)[0]
    elif isinstance(mask_or_idx, np.ndarray) and np.issubdtype(
        mask_or_idx.dtype, np.integer
    ):
        meas_idx = mask_or_idx
    else:
        raise ValueError(
            "mask_or_idx deve ser vetor booleano (mask) ou array de índices (meas_idx)"
        )
    M = meas_idx.size
    Phi = np.zeros((M, N))
    for i, idx in enumerate(meas_idx):
        Phi[i, idx] = 1.0
    return Phi, meas_idx


# ----------------------------
# Normalizar colunas de A e reescalar coeficientes
# ----------------------------
def build_A_and_normalize(Phi, Psi_concat):
    A = Phi.dot(Psi_concat)  # M x (N+K)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_norm = A / col_norms
    return A, A_norm, col_norms


# ----------------------------
# Reconstrução (OMP ou Lasso) e desnormalização
# ----------------------------
def solve_omp(A_norm, y, S_calculado):
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=S_calculado)
    model.fit(A_norm, y)
    coef_norm = model.coef_.copy()
    return coef_norm


def solve_lasso(A_norm, y, alpha=1e-3):
    model = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False)
    model.fit(A_norm, y)
    coef_norm = model.coef_.copy()
    return coef_norm


# ----------------------------
# Converter coef_flat (alpha_full) para reconstrução e waverec
# ----------------------------
def alpha_to_reconstruction(alpha_full, N, K, Psi_wave, shapes, wavelet_name=WAVELET):
    # alpha_full length N+K (primeiro N = identidade, depois K = wavelet flat)
    alpha_I = alpha_full[:N]
    alpha_wave = alpha_full[N:]
    x_direct = alpha_I + Psi_wave.dot(alpha_wave)
    # reconstrução via waverec
    s_coeffs = []
    cur = 0
    for l in shapes:
        s_coeffs.append(alpha_wave[cur : cur + l].copy())
        cur += l
    x_waverec = pywt.waverec(s_coeffs, wavelet_name)[:N]
    return x_direct, x_waverec, alpha_I, alpha_wave


# ----------------------------
# Detectar picos via MAD e refinar amplitudes por LS limitada aos índices detectados
# ----------------------------
def detect_spikes_mad(x, k=REFINE_SPIKE_THRESHOLD_K):
    med = np.median(x)
    sigma = np.median(np.abs(x - med)) / 0.6745 + 1e-12
    idx = np.where(np.abs(x - med) > k * sigma)[0]
    return idx, med, sigma


def refine_spikes_by_LS(Phi, meas_idx, y, spike_indices):
    # Phi: M x N ; meas_idx: array indices (global indices of samples) in signal
    # spike_indices: indices in the time-domain (global positions)
    # constrói Phi_spikes (M x n_spikes) e resolve LS
    if spike_indices.size == 0:
        return np.array([])
    M = Phi.shape[0]
    # map each spike index to row positions inside meas_idx
    Phi_sp = np.zeros((M, spike_indices.size))
    for j, sidx in enumerate(spike_indices):
        # filas onde meas_idx == sidx
        pos = np.where(meas_idx == sidx)[0]
        if pos.size > 0:
            Phi_sp[pos, j] = 1.0
    if np.all(Phi_sp == 0):
        return np.zeros(spike_indices.size)
    a, *_ = np.linalg.lstsq(Phi_sp, y, rcond=None)
    return a


# ----------------------------
# Métricas
# ----------------------------
def mse(x, y):
    return float(np.mean((x - y) ** 2))


def psnr(x_original, x_reconstruido):
    """Calcula a Razão Sinal-Ruído de Pico (PSNR) em dB."""
    # A potência máxima (MAX^2) para sinais normalizados entre -1 e 1 é 1.0 (1^2)
    # Se MAX^2 = (np.max(x_original)**2), seria mais robusto se o sinal não fosse normalizado
    # Assumindo sinais normalizados: MAX_I = 1.0
    MAX_I = 1.0

    # Calcular o Erro Quadrático Médio (MSE)
    err = mse(x_original, x_reconstruido)

    # Previne divisão por zero (caso os sinais sejam idênticos)
    if err == 0:
        return float("inf")

    # PSNR = 10 * log10 (MAX_I^2 / MSE)
    return 10.0 * np.log10((MAX_I**2) / err)


def correlation_coefficient(x_original, x_reconstruido):
    """Calcula o Coeficiente de Correlação (CC) entre os sinais."""
    # O np.corrcoef retorna uma matriz 2x2. Queremos o valor off-diagonal (0, 1) ou (1, 0)
    if x_original.ndim > 1:
        x_original = x_original.flatten()
    if x_reconstruido.ndim > 1:
        x_reconstruido = x_reconstruido.flatten()

    return np.corrcoef(x_original, x_reconstruido)[0, 1]


# ----------------------------
# Interpolação dos resultados
# ----------------------------
def intepolate(signal):
    valid_indices = np.where(signal != 0)[0]
    valid_values = signal[valid_indices]
    f_interp = interp1d(
        valid_indices, valid_values, kind="linear", fill_value="extrapolate"
    )
    indices_to_fill = np.arange(len(signal))
    interpolated_signal = f_interp(indices_to_fill)
    return interpolated_signal


# ----------------------------
# MAIN: integra tudo
# ----------------------------
def pipeline_full():
    # load signal
    # sinal = load_dot_mat_signal()
    # sinal = load_csv_signal()
    sinal, y_labels, N = load_csv_signal()
    # N = sinal.size
    print(f"[info] sinal carregado, N = {N}")

    # carregar Phi/mask/meas_idx se existir
    phi_or_mask = try_load_Phi_or_mask(DATA_DIR)

    if phi_or_mask is None:
        # criar mask aleatória com SAMPLE_M amostras (sem repetição)
        meas_idx = np.random.choice(N, size=SAMPLE_M, replace=False)
        meas_idx = np.sort(meas_idx)
        Phi, meas_idx = construct_Phi_from_mask_or_meas_idx(meas_idx, N)
        print(f"[info] Phi construído aleatoriamente com M = {meas_idx.size}")
    elif isinstance(phi_or_mask, np.ndarray) and phi_or_mask.ndim == 2:
        Phi = phi_or_mask
        # derive meas_idx as rows with single 1 per row
        # assume each row has single 1 at sample position
        rows = Phi.shape[0]
        meas_idx = []
        for i in range(rows):
            pos = np.where(Phi[i] != 0)[0]
            if pos.size > 0:
                meas_idx.append(pos[0])
            else:
                meas_idx.append(-1)
        meas_idx = np.array(meas_idx)
        print(f"[info] Phi carregado, derived meas_idx length = {meas_idx.size}")
    else:
        # mask_or_idx
        if phi_or_mask.dtype == bool:
            Phi, meas_idx = construct_Phi_from_mask_or_meas_idx(phi_or_mask, N)
        else:
            Phi, meas_idx = construct_Phi_from_mask_or_meas_idx(
                np.array(phi_or_mask, dtype=int), N
            )
        print(f"[info] Phi construído a partir de mask/meas_idx, M = {meas_idx.size}")

    # construir Psi_wave e shapes
    Psi_wave, shapes = build_wavelet_matrix_and_shapes(
        N, wavelet=WAVELET, level=WAVELET_LEVEL
    )
    K = Psi_wave.shape[1]
    print(f"[info] Psi_wave shape = {Psi_wave.shape}, total K = {K}")

    # concatenated dictionary
    I = np.eye(N)
    Psi_concat = np.concatenate([I, Psi_wave], axis=1)  # N x (N+K)

    # build A and normalize
    A, A_norm, col_norms = build_A_and_normalize(Phi, Psi_concat)
    
    save_cs_structures('compressed_data_classification/cs_constants/cs_constants.pkl', Phi, A_norm, col_norms, N, shapes)
    
    X_matrix = sinal
    Phi_T = Phi.T # Calculando a transposta de phi por causa do erro de dimensionalidade
    y_cs_matrix = X_matrix.dot(Phi_T)
    
    def get_alpha_for_signal(signal, Phi, Psi_concat, A_norm, col_norms, alpha_lasso=1e-4):
        # calcula y
        y = Phi.dot(signal)
        # solução Lasso (reaproveita sua função)
        model = Lasso(alpha=alpha_lasso, max_iter=10000, fit_intercept=False)
        model.fit(A_norm, y)
        coef_norm = model.coef_.copy()
        coef = coef_norm / col_norms
        return coef

    # exemplo com 100 sinais (ou menos)


    num = min(200, X_matrix.shape[0])
    alphas = np.zeros((num, A_norm.shape[1]))
    nnz_counts = []
    for i in range(num):
        coef = get_alpha_for_signal(X_matrix[i], Phi, None, A_norm, col_norms, alpha_lasso=1e-4)
        alphas[i,:] = coef
        nnz_counts.append(np.count_nonzero(np.abs(coef) > 1e-8))

    print("alpha shape:", alphas.shape)
    print("nnz median:", np.median(nnz_counts), "mean:", np.mean(nnz_counts))
    
    # vars: alphas (num_signals x D), y_labels (num_signals), shapes (lista dos comprimentos wavelet), N

    # 1) Energia por banda (wavelet)
    def energy_per_band(alpha_wave, shapes):
        # alpha_wave is the tail of alpha_full (after N)
        energies = []
        cur = 0
        for l in shapes:
            block = alpha_wave[cur:cur+l]
            energies.append(np.sum(block**2))
            cur += l
        return np.array(energies)

    # construir matriz de features energy
    alpha_wave_part = alphas[:, N:]  # N..end
    E = np.vstack([energy_per_band(alpha_wave_part[i], shapes) for i in range(alpha_wave_part.shape[0])])
    print("Energy features shape:", E.shape)

    # 2) Top-K global indices
    K = 40  # ajuste
    mean_abs = np.mean(np.abs(alphas), axis=0)
    topk_idx = np.argsort(mean_abs)[-K:]
    X_topk = alphas[:, topk_idx]
    print("Top-K features shape:", X_topk.shape)

    # 3) PCA on alphas
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(alphas)
    pca = PCA(n_components=40)  # ajuste
    A_pca = pca.fit_transform(A_scaled)
    print("PCA features shape:", A_pca.shape)

    # Quick classifier test (RandomForest)
    def quick_eval(X, y):
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        return scores.mean(), scores.std()

    for name, X in [("y (M measurements)", y_cs_matrix[:num,:]),
                    ("energy bands", E),
                    ("topk alpha", X_topk),
                    ("pca alpha", A_pca)]:
        mean, std = quick_eval(X, y_labels[:num])
        print(f"{name}: acc={mean:.3f} ± {std:.3f}")

# Execução
if __name__ == "__main__":
    out = pipeline_full()
    print("Pipeline finalizado.")
    plt.show()
