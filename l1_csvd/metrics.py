import numpy as np


def l1_matrix_norm(A, exact=True):
    if not exact:
        U_, Sigma_, V_ = np.linalg.svd(A, full_matrices=False)
        return np.sqrt(A.shape[0]) * np.max(Sigma_)
    else:
        return np.max(np.abs(A).sum(axis=0))


def metric_11(X, U, S, V, matr_norm=False, exact=True):
    A = X.T @ U - V @ S
    if not matr_norm:
        return np.linalg.norm(A, 1)
    else:
        l1_matrix_norm(A, exact)


def join_sol(X, Q, matr_norm=False, exact=True):
    A = Q.T @ X
    if not matr_norm:
        return np.linalg.norm(A, 1)
    else:
        l1_matrix_norm(A, exact)

    return l1_matrix_norm(Q.T @ X, exact)


def Mp(X, U, S, V, matr_norm=False, exact=True):
    A = U.T @ X - S @ V.T
    B = U.T @ X

    if not matr_norm:
        return np.linalg.norm(A, 1) / np.linalg.norm(B, 1)
    else:
        return l1_matrix_norm(A, exact) / l1_matrix_norm(B, exact)


def Mah_dist(U, M, S, y):
    y = y.reshape(-1, 1)
    N = y.shape[1]
    return np.sqrt((U.T @ ((-1) * (M - y)) / np.diag(S) * np.sqrt(N)).sum(axis=1))


def NR(N, S):
    return (S ** 2).sum() / (N ** 2).sum()

def R_sv(S_est, S_clean):
    return np.sqrt(((S_est - S_clean) ** 2).sum()) / np.sqrt((S_clean ** 2).sum())

def rate_to_db(x):
    return 10 * np.log10(x)
