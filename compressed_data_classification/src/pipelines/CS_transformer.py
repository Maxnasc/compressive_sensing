# CS_transformer_fixed.py
import os
import pickle
import json
import time
from typing import Optional, Tuple, List, Union
from sklearn.base import clone

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso


class CompressiveSensingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn para extrair features a partir de compressed sensing:
    - salva/carrega estruturas CS (Phi, Psi_concat opcional, A_norm, col_norms, N, shapes)
    - extrai alpha (via Lasso), energia por banda (wavelet), top-K alpha, PCA(alpha), ou alpha completo
    - suporta Phi em 3 formatos: dense matrix (M x N), boolean mask (N,), meas_idx (k,)
    """

    def __init__(
        self,
        technique: str = "energy",
        cs_structures_path: str = "compressed_data_classification/src/cs/cs_constants",
        cs_metrics_path: str = "compressed_data_classification/src/cs/results/metrics/best_cs_tune_metrics.json",
        lasso_alpha: float = 1e-4,
        K_topk: int = 40,
        pca_components: int = 40,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.cs_structures_path = cs_structures_path
        self.cs_metrics_path = cs_metrics_path
        self.lasso_alpha = lasso_alpha
        self.K_topk = K_topk
        self.pca_components = pca_components
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Controle de qual abordagem vai sar usada na trasnformação
        self.technique = technique

        # placeholders (populados por load_cs_structures ou manualmente)
        self.Phi = None
        self.Psi_concat = None  # opcional (pode ser salvo)
        self.Psi_wave = None
        self.A_norm = None
        self.col_norms = None
        self.N = None
        self.shapes = None
        self.window_size = None
        self.window_step = None
        
        # modelo de otimização de alpha
        self.base_model = None

        # runtime caches
        self._topk_idx = None

    # -------------------------
    # I/O das estruturas CS
    # -------------------------
    def save_cs_structures(
        self,
        path: Optional[str] = None,
        Phi: Optional[np.ndarray] = None,
        A_norm: Optional[np.ndarray] = None,
        col_norms: Optional[np.ndarray] = None,
        N: Optional[int] = None,
        shapes: Optional[List[int]] = None,
        Psi_concat: Optional[np.ndarray] = None,
    ):
        """Salva as estruturas num único .pkl."""
        if path is None:
            path = self.cs_structures_path
        data = {
            "Phi": Phi if Phi is not None else self.Phi,
            "A_norm": A_norm if A_norm is not None else self.A_norm,
            "col_norms": col_norms if col_norms is not None else self.col_norms,
            "N": N if N is not None else self.N,
            "shapes": shapes if shapes is not None else self.shapes,
            "Psi_concat": Psi_concat if Psi_concat is not None else self.Psi_concat,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        if self.verbose:
            print(f"[SAVE] Estruturas CS salvas em: {path}")

    def load_cs_structures(self, path: Optional[str] = None) -> None:
        """Carrega estruturas salvas anteriormente."""
        if path is None:
            path = self.cs_structures_path
        with open(self.cs_metrics_path, "r") as f:
            metrics = json.load(f)

        # with open(path, "rb") as f:
        #     data = pickle.load(f)
        self.Phi = np.load(f"{path}/cs_best_result_Phi.npy")
        self.Psi_wave = np.load(f"{path}/cs_best_result_Psi_w.npy")
        self.A_norm = np.load(f"{path}/cs_best_result_A_norm.npy")
        self.col_norms = np.load(f"{path}/cs_best_result_col_norms.npy")
        self.N = metrics.get("config_parameters").get("N")
        self.lasso_alpha = metrics.get("config_parameters").get("PARAM_VAL")
        self.window_size = metrics.get("config_parameters").get("WINDOW_SIZE")
        self.window_step = metrics.get("config_parameters").get("WINDOW_STEP")
        # self.shapes = data.get("shapes")
        # self.Psi_concat = data.get("Psi_concat", None)
        
        # Instanciando o Lasso para otimização
        self.base_model = Lasso(alpha=self.lasso_alpha, max_iter=2000, fit_intercept=False)
        
        if self.verbose:
            print(f"[LOAD] Estruturas CS carregadas de: {path}")

    # -------------------------
    # utilitário: checar Phi tipo e aplicar
    # -------------------------
    @staticmethod
    def _is_mask(arr) -> bool:
        return isinstance(arr, np.ndarray) and arr.dtype == bool and arr.ndim == 1

    @staticmethod
    def _is_index_array(arr) -> bool:
        return (
            isinstance(arr, np.ndarray)
            and np.issubdtype(arr.dtype, np.integer)
            and arr.ndim == 1
        )

    def _y_from_Phi(self, signal: np.ndarray) -> np.ndarray:
        """
        Produz y = Phi * signal considerando três formatos de Phi:
        - matrix dens a (M x N)
        - mask booleana (N,)
        - meas_idx (k,)
        """
        if self.Phi is None:
            raise ValueError(
                "Phi não definido. Carregue cs_structures ou defina Phi manualmente."
            )
        if isinstance(self.Phi, np.ndarray) and self.Phi.ndim == 2:
            # dense matrix
            return self.Phi.dot(signal)
        elif self._is_mask(self.Phi):
            return signal[self.Phi]
        elif self._is_index_array(self.Phi):
            return signal[self.Phi]
        else:
            raise TypeError(
                "Formato de Phi não suportado. Use matrix (M x N), mask (N,), ou meas_idx (k,)"
            )

    # -------------------------
    # Resolver alpha (Lasso) por uma amostra
    # -------------------------
    def _compute_alpha_single(
        self, signal: np.ndarray, alpha_lasso: Optional[float] = None
    ) -> np.ndarray:
        """Resolve Lasso para uma única amostra e retorna coeficientes desnormalizados (alpha)."""
        if alpha_lasso is None:
            alpha_lasso = self.lasso_alpha
        if self.A_norm is None or self.col_norms is None:
            raise ValueError(
                "A_norm e col_norms devem estar definidos antes de calcular alpha."
            )
        y = self._y_from_Phi(signal)
        model = Lasso(alpha=alpha_lasso, max_iter=10000, fit_intercept=False)
        model.fit(self.A_norm, y)
        coef_norm = model.coef_.copy()
        coef = coef_norm / (self.col_norms + 1e-16)  # evitar div por zero
        return coef

    def compute_alphas(
        self,
        X: np.ndarray,
        alpha_lasso: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ) -> np.ndarray:
        """Compute alphas for all rows in X. Parallelizable with joblib."""
        if n_jobs is None:
            n_jobs = self.n_jobs
        num = X.shape[0]
        # parallel
        if n_jobs == 1:
            alphas = np.vstack(
                [self._compute_alpha_single(X.iloc[i], alpha_lasso) for i in range(num)]
            )
        else:
            alphas = np.vstack(
                Parallel(n_jobs=n_jobs)(
                    delayed(self._compute_alpha_single)(X[i], alpha_lasso)
                    for i in range(num)
                )
            )
        return alphas

    # -------------------------
    # feature extractors
    # -------------------------
    @staticmethod
    def energy_per_band(alpha_wave: np.ndarray, shapes: List[int]) -> np.ndarray:
        energies = []
        cur = 0
        for l in shapes:
            block = alpha_wave[cur : cur + l]
            energies.append(np.sum(block**2))
            cur += l
        return np.array(energies)

    @staticmethod
    def extract_topk_idx_global(alphas: np.ndarray, K: int) -> np.ndarray:
        mean_abs = np.mean(np.abs(alphas), axis=0)
        topk_idx = np.argsort(mean_abs)[-K:]
        return np.sort(topk_idx)  # retorna ordenado (bom para slicing)

    def reconstruct_from_y(
        self,
        y: np.ndarray,
        method: str = "LASSO",
    ) -> np.ndarray:
        """
        Reconstrói o sinal x a partir do sinal subamostrado y.
        Replica exatamente o pipeline do cs_tunning.py
        """
        if self.A_norm is None or self.col_norms is None:
            raise ValueError("Estruturas CS não carregadas.")
        
        t0 = time.perf_counter()

        # # Resolver alpha
        # if method == "OMP":
        #     model = OrthogonalMatchingPursuit(n_nonzero_coefs=int(self.lasso_alpha))
        #     model.fit(self.A_norm, y)
        # else:
        
        # clonando o modelo para evitar os erros de thread
        # model = clone(self.base_model)
            
        self.base_model.fit(self.A_norm, y)
        t1 = time.perf_counter()

        coef = self.base_model.coef_ / self.col_norms
        t2 = time.perf_counter()

        # Reconstrução no domínio do tempo
        alpha_I = coef[: self.N]
        
        alpha_wave = coef[self.N :]
        x_rec = alpha_I + self.Psi_wave.dot(alpha_wave)
        t3 = time.perf_counter()
        
        # Relatório de Tempos dentro da função de reconstrução
        print(f"\n--- Profiling reconstruct_from_y ---")
        print(f"fit do modelo: {t1-t0:.4f}s")
        print(f"coeficientes:        {t2-t1:.4f}s")
        print(f"Multiplicação matricial:  {t3-t2:.4f}s")
        print(f"Total:  {t3-t0:.4f}s")
        
        return x_rec

    # -------------------------
    # export features
    # -------------------------
    def extract_features(
        self,
        X: np.ndarray,
        alpha_lasso: Optional[float] = None,
        recompute_alphas: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extrai e retorna (X_features, alphas_optional)
        technique: "energy", "topk", "pca", "pure_alpha", "RM"
        """
        if self.A_norm is None or self.col_norms is None:
            raise ValueError(
                "Estruturas CS não definidas: A_norm / col_norms faltando."
            )

        if (
            self.technique == "random_mesurements"
        ):  # Usa o vetor de medidas aleatório antes da otimização
            # if self._is_mask(self.Phi):
            #     # Phi é mask booleana (N,)
            #     return X[:, self.Phi], []

            # elif self._is_index_array(self.Phi):
            #     # Phi são índices (M,)
            #     return X[:, self.Phi], []

            # elif isinstance(self.Phi, np.ndarray) and self.Phi.ndim == 2:
            # Phi é matriz M×N
            return X.dot(self.Phi.T), []

        # compute alphas (em memória)
        alphas = self.compute_alphas(X, alpha_lasso=alpha_lasso)

        if self.technique == "energy":
            if self.shapes is None:
                raise ValueError(
                    "shapes não está definido — necessário para energy per band."
                )
            alpha_wave = alphas[:, self.N :]
            features = np.vstack(
                [
                    self.energy_per_band(alpha_wave[i], self.shapes)
                    for i in range(alpha_wave.shape[0])
                ]
            )
            return features, alphas

        elif self.technique == "topk":
            if self._topk_idx is None:
                self._topk_idx = self.extract_topk_idx_global(alphas, self.K_topk)
            features = alphas[:, self._topk_idx]
            return features, alphas

        elif self.technique == "pca":
            scaler = StandardScaler()
            A_scaled = scaler.fit_transform(alphas)
            pca = PCA(n_components=min(self.pca_components, alphas.shape[1]))
            features = pca.fit_transform(A_scaled)
            # opcionalmente armazenar pca para transform futuro (não feito aqui)
            return features, alphas

        elif self.technique == "pure_alpha":
            return alphas, alphas

        elif self.technique == "original_data":
            return X, alphas

        else:
            raise ValueError(
                f"Technique '{self.technique}' desconhecida. Escolha energy/topk/pca/pure_alpha/random_mesurements."
            )

    def save_feature_sets(
        self,
        output_dir: str,
        features: np.ndarray,
        y_labels: Optional[np.ndarray] = None,
        prefix: str = "features",
    ):
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(features)
        if y_labels is not None:
            df["label"] = y_labels
        out_path = os.path.join(output_dir, f"{prefix}.csv")
        df.to_csv(out_path, index=False)
        if self.verbose:
            print(f"[SAVE] Features salvas em: {out_path}")

    # -------------------------
    # sklearn API
    # -------------------------
    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        # transformer não precisa ajustar nada para as estruturas CS se elas já foram carregadas
        if self.A_norm is None or self.col_norms is None or self.Phi_ is None:
            try:
                self.load_cs_structures(self.cs_structures_path)
            except Exception as e:
                if self.verbose:
                    print(f"[INIT] Falha ao carregar cs_structures: {e}")
        return self

    def sliding_window_maker(self, Y_raw):
        # Y_raw shape esperado: (N_amostras, 50)

        # 1. Cria as janelas. Resultado: (N_janelas, 50, 12)
        Y_windows = sliding_window_view(Y_raw, window_shape=self.window_size, axis=0)[
            :: self.window_step
        ]

        # 2. Transpõe para o formato (N_janelas, 12, 50)
        Y_windows = Y_windows.transpose(0, 2, 1)

        # 3. ACHATAMENTO DINÂMICO
        # num_windows = total de janelas geradas
        # -1 faz o numpy calcular automaticamente: 12 * 50 = 600
        num_windows = Y_windows.shape[0]
        Y_all = Y_windows.reshape(num_windows, -1)

        return Y_all  # Retorna matriz (N_janelas, 600)

    def reverse_windowing(self, windows_3d, original_rows):
        """
        windows_3d: Array (N_janelas, 12, 100)
        original_rows: 17000 (quantidade de sinais originais)
        window_step: O passo usado no janelamento (ex: 6 para 50% de sobreposição)
        """
        num_janelas, window_size, num_samples = windows_3d.shape

        # Matriz para acumular os valores reconstruídos
        reconstructed_full = np.zeros((original_rows, num_samples))
        # Matriz para contar quantas vezes cada linha foi preenchida (para tirar a média)
        counts = np.zeros((original_rows, 1))

        for i in range(num_janelas):
            start_idx = i * self.window_step
            end_idx = start_idx + window_size

            # Caso a última janela ultrapasse o limite de 17000
            if end_idx > original_rows:
                overlap_end = original_rows - start_idx
                reconstructed_full[start_idx:original_rows] += windows_3d[
                    i, :overlap_end, :
                ]
                counts[start_idx:original_rows] += 1
            else:
                reconstructed_full[start_idx:end_idx] += windows_3d[i]
                counts[start_idx:end_idx] += 1

        # Divide pelo número de ocorrências para obter a média suave
        counts[counts == 0] = 1  # Evita divisão por zero
        final_X = reconstructed_full / counts

        return final_X

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Y: matriz de sinais subamostrados (n_samples x M)
        Retorna sinais reconstruídos (n_samples x N)
        """
        t0 = time.perf_counter()
        X_rec = []

        # Convertendo o Y de dataframe para array numpy
        Y = Y.to_numpy()
        t1 = time.perf_counter()

        # Conversão de Y em batchs de 12 sinais tal qual o feito no código de tunnig
        Y_batch = self.sliding_window_maker(Y_raw=Y)
        t2 = time.perf_counter()

        X_rec = []
        for i in range(Y_batch.shape[0]):
            x_hat = self.reconstruct_from_y(Y_batch[i])
            X_rec.append(x_hat)
        t3 = time.perf_counter()

        # Separar os sinais convertidos em distúrbios unitários novamente (dividir por 12)
        X_rec = np.array(X_rec)
        num_windows = X_rec.shape[0]
        X_rec_3d = X_rec.reshape(num_windows, 12, 100)
        X_resized = self.reverse_windowing(
            windows_3d=X_rec_3d, original_rows=Y.shape[0]
        )
        t4 = time.perf_counter()
        
        # Relatório de Tempos
        print(f"\n--- Profiling Transform ---")
        print(f"Conversão p/ Numpy: {t1-t0:.4f}s")
        print(f"Janelamento:        {t2-t1:.4f}s")
        print(f"Reconstrução (CS):  {t3-t2:.4f}s (Total de {Y_batch.shape[0]} janelas)")
        print(f"Fusão/Reverse:      {t4-t3:.4f}s")
        print(f"Tempo Total:        {t4-t0:.4f}s")

        return X_resized

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
