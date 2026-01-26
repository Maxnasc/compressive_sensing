import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random
import json
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------
# 1. CONFIGURAÇÃO E CAMINHOS
# ----------------------------
DATA_DIR = "compressed_data_classification/src/cs/cs_constants"
SAVE_PREFIX = os.path.join(DATA_DIR, "cs_best_result")
# CSV_PATH = os.path.join(DATA_DIR, "data.csv")
CSV_PATH = "compressed_data_classification/data/raw/data.csv"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Grade de Parâmetros Extensiva
PARAM_GRID = {
    "SAMPLE_M": [30*12, 40*12, 50*12, 60*12],
    "WAVELET": ["db4", "db8", "sym4"],
    "WAVELET_LEVEL": [2, 3, 4],
    "METHOD": ["OMP", "LASSO"],
}
OMP_K_CANDIDATES = [10, 20, 30, 40]
LASSO_ALPHA_CANDIDATES = [1e-4, 1e-3, 1e-2]
REFINE_SPIKE_THRESHOLD_K = 4.5  # Sensibilidade para o refinamento MAD

# Constantes de janelamento
WINDOW_SIZE = 12 # Janela de 12 sinais
WINDOW_STEP = 6 # Padrão de sobreposição (metade)

# ----------------------------
# 2. MÉTRICAS E PROCESSAMENTO DE SINAL
# ----------------------------


def mse(x, y):
    return float(np.mean((x - y) ** 2))


def psnr(x_orig, x_rec):
    max_val = np.max(np.abs(x_orig))
    if max_val == 0:
        max_val = 1.0
    err = mse(x_orig, x_rec)
    return 10.0 * np.log10((max_val**2) / err) if err > 0 else 100.0


def correlation_coefficient(x_orig, x_rec):
    return np.corrcoef(x_orig.flatten(), x_rec.flatten())[0, 1]


def detect_spikes_mad(x, k=REFINE_SPIKE_THRESHOLD_K):
    med = np.median(x)
    sigma = np.median(np.abs(x - med)) / 0.6745 + 1e-12
    return np.where(np.abs(x - med) > k * sigma)[0]


def refine_spikes_by_LS(Phi, meas_idx, y, spike_indices):
    if spike_indices.size == 0:
        return np.array([])
    M = Phi.shape[0]
    Phi_sp = np.zeros((M, spike_indices.size))
    for j, sidx in enumerate(spike_indices):
        pos = np.where(meas_idx == sidx)[0]
        if pos.size > 0:
            Phi_sp[pos, j] = 1.0
    if np.all(Phi_sp == 0):
        return np.zeros(spike_indices.size)
    a, *_ = np.linalg.lstsq(Phi_sp, y, rcond=None)
    return a


# ----------------------------
# 3. NÚCLEO DO SENSING COMPRIMIDO
# ----------------------------


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


def run_reconstruction(sinal_orig, config, N):
    # Amostragem Aleatória
    meas_idx = np.sort(np.random.choice(N, size=int(config["SAMPLE_M"]), replace=False))
    Phi = np.zeros((len(meas_idx), N))
    for i, idx in enumerate(meas_idx):
        Phi[i, idx] = 1.0

    # Dicionário Híbrido (Identidade + Wavelet)
    Psi_w, shapes = build_wavelet_matrix(
        N, config["WAVELET"], int(config["WAVELET_LEVEL"])
    )
    Psi_concat = np.concatenate([np.eye(N), Psi_w], axis=1)

    A = Phi.dot(Psi_concat)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_norm = A / col_norms

    y = Phi.dot(sinal_orig)

    # Algoritmo de Reconstrução
    if config["METHOD"] == "OMP":
        model = OrthogonalMatchingPursuit(n_nonzero_coefs=int(config["PARAM_VAL"]))
        model.fit(A_norm, y)
    else:
        model = Lasso(alpha=config["PARAM_VAL"], max_iter=5000, fit_intercept=False)
        model.fit(A_norm, y)

    coef = model.coef_ / col_norms
    x_rec = (np.eye(N).dot(coef[:N])) + (Psi_w.dot(coef[N:]))

    return x_rec, Phi, meas_idx, y, Psi_w, A_norm, col_norms


# ----------------------------
# 4. BUSCA EM GRADE (GRID SEARCH)
# ----------------------------


def grid_search_robust(X_matrix, N):
    # Testamos 5 sinais aleatórios para definir a robustez da configuração
    test_indices = random.sample(range(X_matrix.shape[0]), 5)
    sinais_teste = X_matrix[test_indices, :]

    combinations = list(itertools.product(*PARAM_GRID.values()))
    all_results = []

    print(
        f"Pesquisando {len(combinations) * 3} variações em {len(test_indices)} sinais..."
    )

    for combo in combinations:
        c_base = dict(zip(PARAM_GRID.keys(), combo))
        sub_params = (
            OMP_K_CANDIDATES if c_base["METHOD"] == "OMP" else LASSO_ALPHA_CANDIDATES
        )

        for p_val in sub_params:
            psnrs, ccs = [], []
            local_best_cc = -1
            local_best_idx = -1

            c_current = {**c_base, "PARAM_VAL": p_val}

            for idx_sig, sinal in zip(test_indices, sinais_teste):
                try:
                    xr, _, _, _, _, _, _ = run_reconstruction(sinal, c_current, N)
                    cur_psnr = psnr(sinal, xr)
                    cur_cc = correlation_coefficient(sinal, xr)

                    psnrs.append(cur_psnr)
                    ccs.append(cur_cc)

                    # Rastreia qual sinal individual se saiu melhor nesta config
                    if cur_cc > local_best_cc:
                        local_best_cc = cur_cc
                        local_best_idx = idx_sig
                except:
                    continue

            if psnrs:
                all_results.append(
                    {
                        **c_current,
                        "avg_psnr": np.mean(psnrs),
                        "avg_cc": np.mean(ccs),
                        "best_signal_idx": local_best_idx,
                        "best_signal_cc": local_best_cc,
                    }
                )

    return (
        pd.DataFrame(all_results)
        .sort_values(by="avg_psnr", ascending=False)
        .reset_index(drop=True)
    )


# ----------------------------
# 5. SALVAMENTO E REFINAMENTO
# ----------------------------


def finalize_and_save(X_matrix, config, N):
    # Recupera o sinal que obteve o melhor resultado individual
    target_idx = int(config["best_signal_idx"])
    sinal_original = X_matrix[target_idx]

    print(
        f"\n[Finalizando] Usando sinal índice {target_idx} "
        f"(Melhor CC individual: {config['best_signal_cc']:.4f})"
    )

    # Reconstrução direta
    x_dir, Phi, m_idx, y, Psi_w, A_norm, col_norms = run_reconstruction(sinal_original, config, N)

    # Refinamento de picos (MAD + LS)
    spike_idx = detect_spikes_mad(x_dir)
    x_refined = x_dir.copy()

    if spike_idx.size > 0:
        a_ref = refine_spikes_by_LS(Phi, m_idx, y, spike_idx)
        for j, sidx in enumerate(spike_idx):
            x_refined[sidx] = a_ref[j]

    # Métricas finais
    metrics = {
        "mse_refined": float(mse(sinal_original, x_refined)),
        "psnr_refined": float(psnr(sinal_original, x_refined)),
        "cc_refined": float(correlation_coefficient(sinal_original, x_refined)),
    }

    # Estrutura JSON final
    config_parameters = {k: v for k, v in config.items()}
    config_parameters['N'] = N
    config_parameters['WINDOW_SIZE'] = WINDOW_SIZE
    config_parameters['WINDOW_STEP'] = WINDOW_STEP
    to_save = {
        "config_parameters": config_parameters,
        "performance_on_selected_signal": metrics,
    }

    out_json = "compressed_data_classification/src/cs/results/metrics/best_cs_tune_metrics.json"

    # -------- SALVAMENTO INTEGRADO --------
    with open(out_json, "w") as f:
        json.dump(to_save, f, indent=2)

    np.save(SAVE_PREFIX + "_x_original.npy", sinal_original)
    np.save(SAVE_PREFIX + "_meas_idx.npy", m_idx)
    np.save(SAVE_PREFIX + "_Phi.npy", Phi)
    np.save(SAVE_PREFIX + "_Psi_w.npy", Psi_w)
    np.save(SAVE_PREFIX + "_A_norm.npy", A_norm)
    np.save(SAVE_PREFIX + "_col_norms.npy", col_norms)
    np.save(SAVE_PREFIX + "_y.npy", y)
    np.save(SAVE_PREFIX + "_x_rec_best.npy", x_dir)
    np.save(SAVE_PREFIX + "_x_rec_refined.npy", x_refined)

    print(f"[save] resultados salvos com prefixo {SAVE_PREFIX}_*")

    return sinal_original, x_refined, m_idx, y, metrics


# ----------------------------
# 6. EXECUÇÃO PRINCIPAL
# ----------------------------

if __name__ == "__main__":
    # Carregamento
    try:
        df_data = pd.read_csv(CSV_PATH)
        X_raw = df_data.filter(regex=r"^s\d+$").values
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        exit()
        
    # Fazendo o janelamento dos sinais com 12 sinais por janela para tunning
    
    
    X_windows = sliding_window_view(X_raw, window_shape=window_size, axis=0)[::step]
    X_windows = X_windows.transpose(0,2,1) # Ajusta para (Janelas, 12, 100)
    
    # ACHATAMENTO: Transformamos cada janela 12x100 em um vetor de 1200
    num_windows = X_windows.shape[0]
    X_all = X_windows.reshape(num_windows, -1)
    
    N_len = X_all.shape[1]

    # Estágio 1: Grid Search
    results_df = grid_search_robust(X_all, N_len)

    print("\n--- TOP 10 MELHORES CONFIGURAÇÕES (Média de 5 sinais) ---")
    print(results_df[["METHOD", "SAMPLE_M", "WAVELET", "avg_psnr", "avg_cc"]].head(10))

    # Estágio 2: Escolha do Usuário
    idx_choice = int(
        input(
            "\nEscolha o índice (0-9) para salvar o modelo e o melhor sinal correspondente: "
        )
    )
    chosen_config = results_df.iloc[idx_choice].to_dict()

    # Estágio 3: Refinamento e Salvamento
    s_orig, s_ref, m_idx, y_val, final_m = finalize_and_save(
        X_all, chosen_config, N_len
    )

    # Define o tamanho da fonte padrão para todos os elementos
    plt.rcParams.update({"font.size": 24})

    # Estágio 4: Visualização
    plt.figure(figsize=(14, 7))
    plt.plot(s_orig, label="Original Signal", color="black", alpha=0.3, ls="--")
    plt.plot(s_ref, label="Signal Reconstructed", color="blue", lw=1.5)
    plt.scatter(m_idx, y_val, color="red", s=40, label="Samples", zorder=5)

    plt.title(f"Sampled and reconstructed signal")
    plt.xlabel("Samples (n)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(
        "compressed_data_classification/src/cs/results/plots/sampled_and_reconstructed_signal.png"
    )
    plt.show()
