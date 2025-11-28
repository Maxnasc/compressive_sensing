# supondo X_matrix (num_signals x N), y_labels, Phi, Psi_wave, shapes já carregados
from sklearn.linear_model import Lasso
import numpy as np

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
