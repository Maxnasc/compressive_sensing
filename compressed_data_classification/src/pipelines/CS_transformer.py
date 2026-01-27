# CS_transformer_fixed.py
import os
import pickle
import json
from typing import Optional, Tuple, List, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import OrthogonalMatchingPursuit

# Verificar disponibilidade de GPU
try:
    import cupy as cp
    from cuml.linear_model import Lasso as LassoGPU
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


class CompressiveSensingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn para extrair features a partir de compressed sensing.
    Suporta CPU (Joblib) e GPU (CuPy + CuML) para aceleração.
    """

    def __init__(
        self,
        technique: str = "energy",
        cs_structures_path: str = "compressed_data_classification/src/cs/cs_constants",
        cs_metrics_path: str = "compressed_data_classification/src/cs/results/metrics/best_cs_tune_metrics.json",
        lasso_alpha: float = 1e-4,
        K_topk: int = 40,
        pca_components: int = 40,
        n_jobs: int = 1,
        use_gpu: bool = True,
        verbose: bool = False,
    ):
        self.cs_structures_path = cs_structures_path
        self.cs_metrics_path = cs_metrics_path
        self.lasso_alpha = lasso_alpha
        self.K_topk = K_topk
        self.pca_components = pca_components
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # GPU control
        self.use_gpu = use_gpu and HAS_GPU
        if use_gpu and not HAS_GPU:
            print("[WARN] GPU solicitada mas CuPy/CuML não disponível. Usando CPU.")
        
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
        
        # GPU caches
        self.A_norm_gpu = None
        self.col_norms_gpu = None
        self.Psi_wave_gpu = None

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
        with open(self.cs_metrics_path, 'r') as f:
            metrics = json.load(f)
        
        self.Phi = np.load(f'{path}/cs_best_result_Phi.npy')
        self.Psi_wave = np.load(f'{path}/cs_best_result_Psi_w.npy')
        self.A_norm = np.load(f'{path}/cs_best_result_A_norm.npy')
        self.col_norms = np.load(f'{path}/cs_best_result_col_norms.npy')
        self.N = metrics.get("config_parameters").get("N")
        self.lasso_alpha = metrics.get("config_parameters").get("PARAM_VAL")
        self.window_size = metrics.get("config_parameters").get("WINDOW_SIZE")
        self.window_step = metrics.get("config_parameters").get("WINDOW_STEP")
        
        # Transferir para GPU se disponível
        if self.use_gpu:
            self._transfer_to_gpu()
        
        if self.verbose:
            print(f"[LOAD] Estruturas CS carregadas de: {path}")
            if self.use_gpu:
                print(f"[GPU] Dados transferidos para GPU")

    def _transfer_to_gpu(self):
        """Transfere estruturas críticas para GPU uma única vez."""
        if not self.use_gpu or cp is None:
            return
        
        try:
            self.A_norm_gpu = cp.asarray(self.A_norm, dtype=cp.float32)
            self.col_norms_gpu = cp.asarray(self.col_norms, dtype=cp.float32)
            self.Psi_wave_gpu = cp.asarray(self.Psi_wave, dtype=cp.float32)
            if self.verbose:
                print("[GPU] Transferência concluída: A_norm, col_norms, Psi_wave")
        except Exception as e:
            print(f"[ERROR] Falha ao transferir para GPU: {e}")
            self.use_gpu = False

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
        - matrix densa (M x N)
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
    # Resolver alpha (Lasso) - CPU
    # -------------------------
    def _compute_alpha_single(
        self, signal: np.ndarray, alpha_lasso: Optional[float] = None
    ) -> np.ndarray:
        """Resolve Lasso para uma única amostra (CPU)."""
        if alpha_lasso is None:
            alpha_lasso = self.lasso_alpha
        if self.A_norm is None or self.col_norms is None:
            raise ValueError("A_norm e col_norms devem estar definidos.")
        
        y = self._y_from_Phi(signal)
        model = Lasso(alpha=alpha_lasso, max_iter=10000, fit_intercept=False)
        model.fit(self.A_norm, y)
        coef = model.coef_ / (self.col_norms + 1e-16)
        return coef

    def _compute_alpha_single_gpu(self, signal_gpu) -> np.ndarray:
        """Resolve Lasso para uma amostra em GPU."""
        if not self.use_gpu or cp is None:
            raise RuntimeError("GPU não disponível")
        
        try:
            # Calcular y em GPU
            if isinstance(self.Phi, np.ndarray) and self.Phi.ndim == 2:
                Phi_gpu = cp.asarray(self.Phi, dtype=cp.float32)
                y_gpu = Phi_gpu.dot(signal_gpu)
            else:
                # Para mask/index, fazer em CPU
                signal_np = cp.asnumpy(signal_gpu)
                y = self._y_from_Phi(signal_np)
                y_gpu = cp.asarray(y, dtype=cp.float32)
            
            # Solver Lasso em GPU
            model = LassoGPU(
                alpha=self.lasso_alpha,
                max_iter=10000,
                fit_intercept=False,
                output_type='numpy'  # Retorna numpy direto
            )
            model.fit(self.A_norm_gpu, y_gpu)
            
            # Desnormalizar
            coef = model.coef_ / (cp.asnumpy(self.col_norms_gpu) + 1e-16)
            return coef
            
        except Exception as e:
            print(f"[ERROR GPU] {e}. Voltando para CPU.")
            signal_np = cp.asnumpy(signal_gpu) if hasattr(signal_gpu, 'get') else signal_gpu
            return self._compute_alpha_single(signal_np)

    # -------------------------
    # Batch Alphas - com suporte GPU
    # -------------------------
    def _compute_alphas_batch_cpu(self, Y_batch: np.ndarray) -> np.ndarray:
        """Resolve Lasso em paralelo (CPU com Joblib)."""
        num_samples = Y_batch.shape[0]
        
        X_rec_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.reconstruct_from_y)(Y_batch[i])
            for i in range(num_samples)
        )
        
        return np.vstack(X_rec_list)

    def _compute_alphas_batch_gpu(self, Y_batch: np.ndarray) -> np.ndarray:
        """Resolve Lasso em GPU (vetorizado quando possível)."""
        if not self.use_gpu or cp is None:
            return self._compute_alphas_batch_cpu(Y_batch)
        
        try:
            num_samples = Y_batch.shape[0]
            
            # Transferir batch para GPU
            Y_batch_gpu = cp.asarray(Y_batch, dtype=cp.float32)
            
            # Processar em GPU
            X_rec_list = [
                self._reconstruct_from_y_gpu(Y_batch_gpu[i])
                for i in range(num_samples)
            ]
            
            return np.vstack(X_rec_list)
            
        except Exception as e:
            if self.verbose:
                print(f"[GPU FALLBACK] Erro em GPU: {e}. Usando CPU.")
            return self._compute_alphas_batch_cpu(Y_batch)

    def _compute_alphas_batch(self, Y_batch: np.ndarray) -> np.ndarray:
        """Wrapper que escolhe CPU ou GPU."""
        if self.use_gpu:
            return self._compute_alphas_batch_gpu(Y_batch)
        else:
            return self._compute_alphas_batch_cpu(Y_batch)

    def compute_alphas(
        self,
        X: np.ndarray,
        alpha_lasso: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ) -> np.ndarray:
        """Compute alphas - wrapper que escolhe CPU/GPU."""
        if n_jobs is None:
            n_jobs = self.n_jobs
        
        num = X.shape[0]
        
        if self.use_gpu:
            # GPU: processar tudo de uma vez
            return self._compute_alphas_batch(X)
        else:
            # CPU: usar joblib
            if n_jobs == 1:
                alphas = np.vstack(
                    [self._compute_alpha_single(X[i], alpha_lasso) for i in range(num)]
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
        return np.sort(topk_idx)
    
    def reconstruct_from_y(self, y: np.ndarray, method: str = "LASSO") -> np.ndarray:
        """Reconstrói sinal a partir de y (CPU)."""
        if self.A_norm is None or self.col_norms is None:
            raise ValueError("Estruturas CS não carregadas.")

        if method == "OMP":
            model = OrthogonalMatchingPursuit(n_nonzero_coefs=int(self.lasso_alpha))
            model.fit(self.A_norm, y)
        else:
            model = Lasso(alpha=self.lasso_alpha, max_iter=10000, fit_intercept=False)
            model.fit(self.A_norm, y)

        coef = model.coef_ / self.col_norms
        alpha_I = coef[:self.N]
        alpha_wave = coef[self.N:]
        x_rec = alpha_I + self.Psi_wave.dot(alpha_wave)

        return x_rec

    def _reconstruct_from_y_gpu(self, y_gpu) -> np.ndarray:
        """Reconstrói sinal a partir de y (GPU)."""
        if not self.use_gpu or cp is None:
            y_np = cp.asnumpy(y_gpu) if hasattr(y_gpu, 'get') else y_gpu
            return self.reconstruct_from_y(y_np)
        
        try:
            # Solver Lasso em GPU
            model = LassoGPU(
                alpha=self.lasso_alpha,
                max_iter=10000,
                fit_intercept=False,
                output_type='numpy'
            )
            model.fit(self.A_norm_gpu, y_gpu)
            
            # Desnormalizar
            coef = model.coef_ / (cp.asnumpy(self.col_norms_gpu) + 1e-16)
            alpha_I = coef[:self.N]
            alpha_wave = coef[self.N:]
            
            # Reconstrução (usar CPU para Psi_wave se não estiver em GPU)
            if self.Psi_wave_gpu is not None:
                alpha_wave_gpu = cp.asarray(alpha_wave)
                x_rec = alpha_I + cp.asnumpy(self.Psi_wave_gpu.dot(alpha_wave_gpu))
            else:
                x_rec = alpha_I + self.Psi_wave.dot(alpha_wave)
            
            return x_rec
            
        except Exception as e:
            print(f"[ERROR GPU Reconstruct] {e}. Usando CPU.")
            y_np = cp.asnumpy(y_gpu) if hasattr(y_gpu, 'get') else y_gpu
            return self.reconstruct_from_y(y_np)


    # -------------------------
    # export features
    # -------------------------
    def extract_features(
        self,
        X: np.ndarray,
        alpha_lasso: Optional[float] = None,
        recompute_alphas: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extrai features baseado na técnica configurada."""
        if self.A_norm is None or self.col_norms is None:
            raise ValueError("Estruturas CS não definidas.")

        if self.technique == "random_mesurements":
            return X.dot(self.Phi.T), []

        alphas = self.compute_alphas(X, alpha_lasso=alpha_lasso)

        if self.technique == "energy":
            if self.shapes is None:
                raise ValueError("shapes não definido — necessário para energy per band.")
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
            return features, alphas

        elif self.technique == "pure_alpha":
            return alphas, alphas
        
        elif self.technique == "original_data":
            return X, alphas

        else:
            raise ValueError(f"Technique '{self.technique}' desconhecida.")

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
            print(f"[SAVE] Features salvas em: {out_path}")

    # -------------------------
    # sklearn API
    # -------------------------
    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        if self.A_norm is None or self.col_norms is None:
            try:
                self.load_cs_structures(self.cs_structures_path)
            except Exception as e:
                if self.verbose:
                    print(f"[INIT] Falha ao carregar cs_structures: {e}")
        return self
    
    def _compute_alphas_batch(self, Y_batch: np.ndarray) -> np.ndarray:
        """
        Resolve Lasso para MÚLTIPLAS amostras em paralelo com joblib
        Y_batch: (N_janelas, M)
        Retorna: (N_janelas, N) onde N = self.N + len(shapes)
        """
        num_samples = Y_batch.shape[0]
        
        # Usar joblib.Parallel em vez de loop serial
        X_rec_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.reconstruct_from_y)(Y_batch[i])
            for i in range(num_samples)
        )
        
        return np.vstack(X_rec_list)
    
    def sliding_window_maker(self, Y_raw):
        """Cria janelas de Y_raw (N_amostras, 50) -> (N_janelas, 600)."""
        Y_windows = sliding_window_view(Y_raw, window_shape=self.window_size, axis=0)[::self.window_step]
        Y_windows = Y_windows.transpose(0, 2, 1)
        num_windows = Y_windows.shape[0]
        Y_all = Y_windows.reshape(num_windows, -1)
        return Y_all

    def reverse_windowing_optimized(self, windows_3d, original_rows):
        """
        Versão otimizada de reverse_windowing usando operações NumPy vetorizadas.
        """
        num_janelas, window_size, num_samples = windows_3d.shape
        
        reconstructed_full = np.zeros((original_rows, num_samples), dtype=np.float32)
        counts = np.zeros((original_rows, 1), dtype=np.float32)
        
        # Pré-calcular índices
        start_indices = np.arange(num_janelas) * self.window_step
        
        for i in range(num_janelas):
            start_idx = start_indices[i]
            end_idx = min(start_idx + window_size, original_rows)
            
            if start_idx < original_rows:
                rows_to_fill = end_idx - start_idx
                reconstructed_full[start_idx:end_idx] += windows_3d[i, :rows_to_fill, :]
                counts[start_idx:end_idx] += 1
        
        counts = np.maximum(counts, 1)  # Evita div por zero
        return reconstructed_full / counts

    def reverse_windowing(self, windows_3d, original_rows):
        """
        windows_3d: Array (N_janelas, 12, 100)
        original_rows: 17000 (quantidade de sinais originais)
        window_step: O passo usado no janelamento (ex: 6 para 50% de sobreposição)
        """
        return self.reverse_windowing_optimized(windows_3d, original_rows)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Y: matriz de sinais subamostrados (n_samples x M)
        Retorna sinais reconstruídos (n_samples x N)
        """
        Y = Y.to_numpy() if hasattr(Y, 'to_numpy') else Y
        
        if self.verbose:
            mode = "GPU" if self.use_gpu else "CPU"
            print(f"[TRANSFORM] Usando {mode} | Input shape: {Y.shape}")
    
        # 1. Janelamento
        Y_batch = self.sliding_window_maker(Y_raw=Y)
        
        # 2. Resolver Lasso (CPU ou GPU)
        X_rec = self._compute_alphas_batch(Y_batch)
        
        # 3. Desjanelamento otimizado
        num_windows = X_rec.shape[0]
        X_rec_3d = X_rec.reshape(num_windows, 12, 100)
        X_resized = self.reverse_windowing(windows_3d=X_rec_3d, original_rows=Y.shape[0])
        
        return X_resized

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
