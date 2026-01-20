import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class XPQRSFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _get_approximated_derivatives(self, x):
        """
        Calcula as 4 primeiras derivadas aproximadas conforme Eq. 10-18[cite: 145, 147, 149].
        O cálculo é feito pela diferença entre elementos sucessivos.
        """
        d1 = np.diff(x)  # Primeira derivada
        d2 = np.diff(d1) # Segunda derivada
        d3 = np.diff(d2) # Terceira derivada
        d4 = np.diff(d3) # Quarta derivada
        return [x, d1, d2, d3, d4]

    def _extract_features(self, u):
        """
        Extrai Log Energy, Shannon Energy e Mobility de um vetor u.
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
        var_u = np.var(u)
        # Calcula a derivada temporal interna para a mobilidade
        du_dt = np.diff(u)
        var_du_dt = np.var(du_dt)
        mob = np.sqrt(var_du_dt / (var_u + epsilon))
        
        return [le, se, mob]

    def transform(self, X):
        """
        Processa cada sinal para gerar o vetor de 15 características[cite: 160, 161].
        """
        features_list = []
        for signal in X:
            # Gera o sinal original + 4 derivadas
            all_signals = self._get_approximated_derivatives(signal)
            
            # Para cada um dos 5 sinais, extrai as 3 características (Total = 15)
            sig_features = []
            for s in all_signals:
                sig_features.extend(self._extract_features(s))
            
            features_list.append(sig_features)
            
        return np.array(features_list)

    def fit(self, X, y=None):
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