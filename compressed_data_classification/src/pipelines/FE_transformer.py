"""
Module: pipelines/FE_transformer.py

XPQRS Feature Extractor for electrical signal classification.

This module implements the XPQRS feature extraction method that extracts 15 features
from electrical signals by computing features from the signal and its first four derivatives.
The features extracted are:
- Log Energy (LE)
- Shannon Energy (SE)
- Mobility (Mob) 

These features are extracted for each of the 5 signal representations (original + 4 derivatives),
yielding a total of 15 features per signal.

This transformer is compatible with scikit-learn pipelines and can be used with any classifier.

Author: Maxnasc7
License: MIT
Reference: XPQRS feature extraction from signal processing literature
"""

import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, TransformerMixin

# Para rodar o profiler
import builtins

if 'profile' not in builtins.__dict__:
    def profile(func): 
        return func
    builtins.__dict__['profile'] = profile

class XPQRSFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that extracts XPQRS features from signals.
    
    This transformer extracts 15 features (Log Energy, Shannon Energy, Mobility) from
    a signal and its first four derivatives. The extracted features are useful for
    electrical disturbance classification.
    
    Attributes
    ----------
    None (no state is maintained)
    
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.svm import SVC
    >>> pipe = Pipeline([
    ...     ('features', XPQRSFeatureExtractor()),
    ...     ('classifier', SVC())
    ... ])
    >>> pipe.fit(X_train, y_train)
    """
    
    def __init__(self):
        pass

    def _get_approximated_derivatives(self, x):
        """
        Compute the first four derivatives of a signal.
        
        Calculates the first four derivatives using finite differences (np.diff).
        This is used to obtain different representations of the signal for feature extraction.
        
        Parameters
        ----------
        x : np.ndarray
            1D signal array
        
        Returns
        -------
        list of np.ndarray
            [x, d1, d2, d3, d4] where:
            - x: Original signal
            - d1: First derivative (signal velocity)
            - d2: Second derivative (signal acceleration)
            - d3: Third derivative
            - d4: Fourth derivative
        
        Notes
        -----
        Each derivative reduces signal length by 1, so final arrays have different lengths.
        """
        d1 = np.diff(x)  # Primeira derivada
        d2 = np.diff(d1) # Segunda derivada
        d3 = np.diff(d2) # Terceira derivada
        d4 = np.diff(d3) # Quarta derivada
        return [x, d1, d2, d3, d4]

    def _extract_features(self, u):
        """
        Extract three features from a signal representation.
        
        Extracts the following features from a signal vector:
        1. Log Energy (LE): Sum of log of squared signal values
        2. Shannon Energy (SE): Negative sum of squared signal times log of squared signal
        3. Mobility (Mob): Square root of ratio of velocity variance to signal variance
        
        These features characterize the energy distribution and dynamics of the signal.
        
        Parameters
        ----------
        u : np.ndarray
            1D signal vector
        
        Returns
        -------
        list of float
            [log_energy, shannon_energy, mobility]
        
        Notes
        -----
        - A small epsilon is added to avoid log(0) errors
        - Mobility is useful for distinguishing different electrical disturbance types
        """
        # Evita log(0) adicionando uma constante pequena (epsilon)
        epsilon = 1e-10
        u_sq = u**2 + epsilon
        
        # 1. Log Energy (LE)
        le = np.sum(np.log(u_sq))
        
        # 2. Shannon Energy (SE)
        se = -np.sum(u_sq * np.log(u_sq))
        
        # 3. Mobility (Mob)
        # Mobility = sqrt(Var(du/dt) / Var(u))
        if len(u) > 1:
            var_u = np.var(u)
            du_dt = np.diff(u)
            mob = np.sqrt(np.var(du_dt) / (var_u + epsilon))
        else:
            mob = 0.0
        
        return [le, se, mob]

    @profile
    def transform(self, X):
        """
        Extract XPQRS features from multiple signals.
        
        Processes each signal in the input data to extract 15 XPQRS features:
        - 3 features (LE, SE, Mob) × 5 signal representations (original + 4 derivatives)
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input signals with shape (n_samples, n_features)
            Each row represents one signal
        
        Returns
        -------
        np.ndarray
            Extracted features with shape (n_samples, 15)
            Contains 15 features per signal
        
        Notes
        -----
        - If X is a numpy array, it's converted to DataFrame
        - Each signal is processed independently
        - Processing time is printed for monitoring
        """
        t0 = time.perf_counter()
        # Verificando se X é um dicionário ou um array numpy
        if type(X) == np.ndarray: # <- Veio do CS
            # Converte para dataframe
            X = pd.DataFrame(X)
        
        features_list = []
        for signal in X.values:
            # Gera o sinal original + 4 derivadas
            all_signals = self._get_approximated_derivatives(signal)
            
            # Para cada um dos 5 sinais, extrai as 3 características (Total = 15)
            sig_features = []
            for s in all_signals:
                sig_features.extend(self._extract_features(s))
            
            features_list.append(sig_features)
            
        t1 = time.perf_counter()
        print(f"Tempo total de transformação para {X.shape[0]}: {t1-t0:.4f}s")
        return np.array(features_list)

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op as this transformer is stateless).
        
        This method is required by scikit-learn but does not learn any parameters
        as feature extraction is deterministic.
        
        Parameters
        ----------
        X : array-like
            Input data (unused)
        y : array-like, optional
            Target values (unused)
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        return self

# --- Exemplo de Integração no Pipeline ---
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Configuração do Pipeline seguindo o artigo [cite: 141, 184, 188]
pipeline = Pipeline([
    ('feature_extraction', XPQRSFeatureExtractor()),
    ('classifier', SVC(kernel='poly', degree=2)) # Quadratic SVM
])

# Exemplo de uso com K-Fold (10 folds conforme o artigo [cite: 210])
# scores = cross_val_score(pipeline, X_raw_signals, y_labels, cv=10)