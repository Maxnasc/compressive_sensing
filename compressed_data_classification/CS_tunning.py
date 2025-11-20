# pipeline_cs_full.py
# Pipeline completo: carga -> amostragem -> dicionário concatenado -> OMP/Lasso -> refinamento de picos -> métricas/plots
# Requisitos: numpy, matplotlib, scikit-learn, pywt, mplcursors (opcional para hover)
# Execute no Jupyter (VSCode): antes rode %matplotlib widget para interatividade

import os
from pathlib import Path
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
import json
import mplcursors
import scipy.io as sio
from scipy.interpolate import interp1d
import pandas as pd
import kagglehub

# ----------------------------
# CONFIG (ajuste estes parâmetros)
# ----------------------------
DATA_DIR = "compressed_data_classification/cs_metrics"  # onde procurar sinal e salvar resultados
SAMPLE_M = 40  # número de amostras M se não houver Phi/mask
RANDOM_SEED = 42
WAVELET = "db8"  # wavelet para parte wavelet
WAVELET_LEVEL = 4  # nível de decomposição (tente 1..4)
METHODS = ["OMP", "LASSO"]  # quais métodos testar
OMP_candidates = [
    10,
    20,
    30,
    50,
    60,
]  # valores de n_nonzero_coefs para testar (ajuste conforme k90)
LASSO_alphas = [1e-5, 1e-4, 1e-3, 1e-2, ]  # alphas para Lasso
REFINE_SPIKE_THRESHOLD_K = 4.5  # k para MAD threshold de picos
SAVE_PREFIX = os.path.join(DATA_DIR, "cs_result")  # prefixo para salvar outputs
# ----------------------------

np.random.seed(RANDOM_SEED)

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
    
    X_matrix = sinal
    Phi_T = Phi.T # Calculando a transposta de phi por causa do erro de dimensionalidade
    y_cs_matrix = X_matrix.dot(Phi_T)
    
    ##########################################################################
    # SALVANDO AS AMOSTRAS COMPRIMIDAS PARA TREINAMENTO DO modelo de linguagem
    ##########################################################################
    
    data_compressed = pd.DataFrame(y_cs_matrix)
    data_compressed.columns = [f'm_{i+1}' for i in range(y_cs_matrix.shape[1])]
    data_compressed['target'] = y_labels
    output_file_name = 'compressed_data_classification/seed_data_compressed_M40_db8.csv'
    data_compressed.to_csv(output_file_name, index=False)
    print(f"\n[SAVE] Dataset comprimido salvo em: {output_file_name}")
    np.save("compressed_data_classification/Phi_final_M40.npy", Phi)
    
    ##########################################################################
    
    # Para fins de medição das métricas é preciso aplica-las somente à um sinal
    sinal_to_reconstruct = X_matrix[0,:] # Primeira linha do df
    y = y_cs_matrix[0,:] # <- Sinal amostrado aqui
    # y = sinal  # <- Considerando o sinal original como sendo o amostrado
    
    print(
        f"[info] A shape = {A.shape}, A_norm shape = {A_norm.shape}, y shape = {y.shape}"
    )

    results = {}

    # Sweep OMP candidates
    if "OMP" in METHODS:
        results["OMP"] = []
        for S in OMP_candidates:
            if S <= 0 or S > A_norm.shape[1]:
                continue
            try:
                coef_norm = solve_omp(A_norm, y, S)
                coef = coef_norm / col_norms
                x_direct, x_waverec, alpha_I, alpha_wave = alpha_to_reconstruction(
                    coef, N, K, Psi_wave, shapes, wavelet_name=WAVELET
                )
                m = mse(sinal_to_reconstruct, x_direct)
                p = psnr(sinal_to_reconstruct, x_direct)
                cc = correlation_coefficient(sinal_to_reconstruct, x_direct)
                results["OMP"].append(
                    {
                        "S": S,
                        "mse": m,
                        "psnr": p,
                        "cc": cc,
                        "coef": coef,
                        "x_direct": x_direct,
                        "x_waverec": x_waverec,
                    }
                )
                print(f"[OMP] S={S} MSE={m:.4e} PSNR={p:.4e} CC={cc:.4e}")
            except Exception as e:
                print(f"[OMP] erro S={S} -> {e}")

    # Sweep Lasso alphas
    if "LASSO" in METHODS:
        results["LASSO"] = []
        for a in LASSO_alphas:
            try:
                coef_norm = solve_lasso(A_norm, y, alpha=a)
                coef = coef_norm / col_norms
                x_direct, x_waverec, alpha_I, alpha_wave = alpha_to_reconstruction(
                    coef, N, K, Psi_wave, shapes, wavelet_name=WAVELET
                )
                m = mse(sinal_to_reconstruct, x_direct)
                p = psnr(sinal_to_reconstruct, x_direct)
                cc = correlation_coefficient(sinal_to_reconstruct, x_direct)
                results["LASSO"].append(
                    {
                        "alpha": a,
                        "mse": m,
                        "psnr": p,
                        "cc": cc,
                        "coef": coef,
                        "x_direct": x_direct,
                        "x_waverec": x_waverec,
                    }
                )
                print(f"[LASSO] alpha={a} MSE={m:.4e} PSNR={p:.4e} CC={cc:.4e}")
            except Exception as e:
                print(f"[LASSO] erro alpha={a} -> {e}")

    # Escolher o melhor resultado de cada família para posterior refinamento
    best = {}
    if results.get("OMP"):
        best["OMP"] = min(results["OMP"], key=lambda r: r["mse"])
        print(f"[best OMP] S={best['OMP']['S']} MSE={best['OMP']['mse']:.4e}")
    if results.get("LASSO"):
        best["LASSO"] = min(results["LASSO"], key=lambda r: r["mse"])
        print(
            f"[best LASSO] alpha={best['LASSO']['alpha']} MSE={best['LASSO']['mse']:.4e}"
        )

    # Refinamento de picos no melhor método (escolhe por MSE)
    chosen_method = None
    chosen_res = None
    cand_methods = [(k, (v if isinstance(v, dict) else v)) for k, v in best.items()]
    if best:
        # decide por menor MSE entre OMP e LASSO
        mm = []
        for k in best:
            val = best[k]
            mm.append((k, val["mse"]))
        chosen_method = min(mm, key=lambda t: t[1])[0]
        chosen_res = best[chosen_method]
        print(
            f"[choose] escolhendo {chosen_method} para refinamento (MSE {chosen_res['mse']:.4e})"
        )
    else:
        print("[warn] nenhum resultado gerado (talvez parâmetros inválidos)")
        return results

    # detect spikes em reconstrução direta e refinar amplitudes
    x_direct = chosen_res["x_direct"]
    spike_idx, med, sigma = detect_spikes_mad(x_direct)
    print(f"[spikes] detectados {spike_idx.size} picos (k={REFINE_SPIKE_THRESHOLD_K})")

    if spike_idx.size > 0:
        a_ref = refine_spikes_by_LS(Phi, meas_idx, y, spike_idx)
        # aplicar ajustes no x_direct
        x_refined = x_direct.copy()
        # colocar amplitudes refinadas nos índices dos picos
        for j, sidx in enumerate(spike_idx):
            x_refined[sidx] = a_ref[j]
        chosen_res["x_refined"] = x_refined
        print("[refine] amplitudes refinadas aplicadas")
    else:
        chosen_res["x_refined"] = x_direct.copy()
        print("[refine] sem picos para refinar")

    # Gerando sinais interpolados
    chosen_res["x_direct_interpolated"] = intepolate((chosen_res["x_direct"]))
    chosen_res["x_refined_interpolated"] = intepolate((chosen_res["x_refined"]))

    # -----------------------------------------------------
    # CÁLCULO DE MÉTRICAS PARA SINAL REFINADO (adicione aqui)
    # -----------------------------------------------------
    x_refined = chosen_res["x_refined"]
    chosen_res["mse_refined"] = mse(sinal_to_reconstruct, x_refined)
    chosen_res["psnr_refined"] = psnr(sinal_to_reconstruct, x_refined)
    chosen_res["cc_refined"] = correlation_coefficient(sinal_to_reconstruct, x_refined)

    print(
        f"\n[Métricas Refinadas] MSE: {chosen_res['mse_refined']:.4e}, PSNR: {chosen_res['psnr_refined']:.2f} dB, CC: {chosen_res['cc_refined']:.4f}"
    )
    # -----------------------------------------------------

    # Salvar resultados principais (atualizar 'best' e 'to_save')
    out_json = SAVE_PREFIX + "_summary.json"
    to_save = {
        "N": N,
        "M": len(meas_idx),
        "wavelet": WAVELET,
        "wavelet_level": WAVELET_LEVEL,
        "best": {
            k: {
                "mse": float(best[k]["mse"]),
                "psnr": float(best[k]["psnr"]),  # <--- NOVA
                "cc": float(best[k]["cc"]),  # <--- NOVA
                "param": best[k].get("S", best[k].get("alpha")),
            }
            for k in best
        },
        "refined_metrics": {  # <--- NOVO BLOCO
            "mse": float(chosen_res["mse_refined"]),
            "psnr": float(chosen_res["psnr_refined"]),
            "cc": float(chosen_res["cc_refined"]),
        },
    }

    with open(out_json, "w") as f:
        json.dump(to_save, f, indent=2)
    np.save(SAVE_PREFIX + "_x_original.npy", sinal)
    np.save(SAVE_PREFIX + "_meas_idx.npy", meas_idx)
    np.save(SAVE_PREFIX + "_Phi.npy", Phi)
    np.save(SAVE_PREFIX + "_y.npy", y)
    np.save(SAVE_PREFIX + "_x_rec_best.npy", chosen_res["x_direct"])
    np.save(SAVE_PREFIX + "_x_rec_refined.npy", chosen_res["x_refined"])
    print(f"[save] resultados salvos com prefixo {SAVE_PREFIX}_*")

    # Plots removidos

    return {
        "results": results,
        "best": best,
        "chosen_method": chosen_method,
        "chosen_res": chosen_res,
    }


# Execução
if __name__ == "__main__":
    out = pipeline_full()
    print("Pipeline finalizado.")
