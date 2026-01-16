import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pickle
# ------------------------------------------------------------
# Funções utilitárias
# ------------------------------------------------------------

def compute_alpha(signal, Phi, A_norm, col_norms, alpha_lasso=1e-4):
    """Resolve o LASSO e retorna o vetor alfa (coeficientes esparsos)."""
    model = Lasso(alpha=alpha_lasso, max_iter=10000, fit_intercept=False)
    model.fit(A_norm, Phi.dot(signal))
    coef_norm = model.coef_.copy()
    coef = coef_norm / col_norms
    return coef


def energy_per_band(alpha_wave, shapes):
    """Extrai energia por banda wavelet a partir da parte wavelet de alfa."""
    energies = []
    cur = 0
    for l in shapes:
        block = alpha_wave[cur:cur+l]
        energies.append(np.sum(block**2))
        cur += l
    return np.array(energies)


def extract_topk_alpha(alphas, K=40):
    """Seleciona os K índices globais mais importantes (maior média das magnitudes)."""
    mean_abs = np.mean(np.abs(alphas), axis=0)
    topk_idx = np.argsort(mean_abs)[-K:]
    return alphas[:, topk_idx], topk_idx

# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------

def extract_features_dataset(
    X_matrix,
    y_labels,
    Phi,
    A_norm,
    col_norms,
    N,
    shapes,
    alpha_lasso=1e-4,
    K_topk=40,
    pca_components=40,
):
    """
    Extrai alfa, energy bands, top-K alpha e PCA(alpha) para todo o dataset.
    Retorna um dicionário com todos os conjuntos de features.
    """

    num = X_matrix.shape[0]
    D = A_norm.shape[1]

    # 1) Calcular alphas
    alphas = np.zeros((num, D))
    for i in range(num):
        alphas[i] = compute_alpha(X_matrix[i], Phi, A_norm, col_norms, alpha_lasso)

    # 2) Energy per band
    alpha_wave = alphas[:, N:]
    E = np.vstack([energy_per_band(alpha_wave[i], shapes) for i in range(num)])

    # 3) Top-K alpha
    X_topk, topk_idx = extract_topk_alpha(alphas, K=K_topk)

    # 4) PCA(alpha)
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(alphas)
    pca = PCA(n_components=pca_components)
    A_pca = pca.fit_transform(A_scaled)

    return {
        "alpha": alphas,
        "energy": E,
        "topk_alpha": X_topk,
        "pca_alpha": A_pca,
        "topk_idx": topk_idx,
    }

# ------------------------------------------------------------
# Função para salvar tudo em CSV
# ------------------------------------------------------------

# ------------------------------------------------------------
# Salvar e carregar estruturas CS (Phi, A_norm, col_norms, N, shapes)
# ------------------------------------------------------------

def save_cs_structures(path, Phi, A_norm, col_norms, N, shapes):
    data = {"Phi": Phi, "A_norm": A_norm, "col_norms": col_norms, "N": N, "shapes": shapes}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[SAVE] Estruturas CS salvas em: {path}")

def load_cs_structures(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[LOAD] Estruturas CS carregadas de: {path}")
    return data["Phi"], data["A_norm"], data["col_norms"], data["N"], data["shapes"]

# ------------------------------------------------------------(output_dir, feature_dict, y_labels):
    """Salva alfa, energy, top-K alfa e PCA em arquivos CSV no diretório especificado."""
    alpha = feature_dict["alpha"]
    energy = feature_dict["energy"]
    topk = feature_dict["topk_alpha"]
    pca_a = feature_dict["pca_alpha"]

    df_alpha = pd.DataFrame(alpha)
    df_alpha["label"] = y_labels
    df_alpha.to_csv(f"{output_dir}/alpha_full.csv", index=False)

    df_energy = pd.DataFrame(energy)
    df_energy["label"] = y_labels
    df_energy.to_csv(f"{output_dir}/energy_bands.csv", index=False)

    df_topk = pd.DataFrame(topk)
    df_topk["label"] = y_labels
    df_topk.to_csv(f"{output_dir}/topk_alpha.csv", index=False)

    df_pca = pd.DataFrame(pca_a)
    df_pca["label"] = y_labels
    df_pca.to_csv(f"{output_dir}/pca_alpha.csv", index=False)

    print("[SAVE] Todos os conjuntos de features foram salvos em:", output_dir)
