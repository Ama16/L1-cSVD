import numpy as np
from scipy.stats import ortho_group

from l1_csvd.utils import L1pca


def L1_cSVD(X, K, num_iter=10):
    """Get L1 cSVD decomposition.

    Parameters
    ----------
    X:
        data matrix
    K:
        the number of components
    num_iter:
        number of iterations
    """
    D, N = X.shape
    U = L1pca(X, K)
    A = np.dot(X.T, U)
    Sigma = np.zeros((K, K))
    V = ortho_group.rvs(dim=N)[:K].T

    for _ in range(num_iter):
        for i in range(K):
            s = np.empty(N)
            M = np.empty(N)
            for j in range(N):
                s[j] = A[j][i] / V[j][i]

                tmp = A[:, i] - s[j] * V[:, i]
                M[j] = np.linalg.norm(tmp, ord=1)

            j_opt = np.argmin(M)
            Sigma[i][i] = s[j_opt]
        tmp = np.diag(1 / np.diagonal(Sigma))
        U_, Sigma_, V_ = np.linalg.svd(np.dot(A, tmp), full_matrices=False)
        V = np.dot(U_, V_)
    return U, Sigma, V
