import numpy as np
import numpy.linalg as LA
from time import time

def NextComb(l):
    """Get next value of binary value represented as vector {+-1}^{n}.

    Parameters
    ----------
    l:
        vector with +1 or -1 values
    """
    Last_one = np.where(l == 1)[0][-1]
    if Last_one == len(l) - 1:
        l[-1] = -1
        return l
    else:
        l[Last_one:] = -1 * l[Last_one:]
        return l


def BNM(X, k):
    """Get binary matrix B which maximizes ||XB||_*^2.

    Parameters
    ----------
    X:
        data matrix
    k:
        the number of components
    """
    n = X.shape[1]
    tempB = np.array([1] * (n * k))
    sum_sv = -1e6
    Last_B = np.array([-1] * (n * k))
    
    i = 0
    last = 0
    # print("L1pca start")
    b = time()
    for _ in range(10 ** 4):
        i += 1
        tempB = np.random.choice([1, -1], size=n*k)
        tempB.shape = (n, k)
        temp_sum_sv = sum(LA.svd(np.dot(X, tempB))[1])
        if sum_sv < temp_sum_sv:
            sum_sv = temp_sum_sv
            B = tempB.copy()
            cur_time = time() - b
            # print("BNM Iter: {} | -Loss: {} | Time spend (min): {}".format(i, sum_sv, cur_time / 60))
    # print()
    return B


def L1pca(X, k):
    """Get L1 PCA.

    Parameters
    ----------
    X:
        data matrix
    k:
        the number of components
    """
    p = X.shape[0]

    median = np.median(X, axis=1)
    median.shape = (p, 1)
    X = X - median

    B_BNM = BNM(X, k)
    U, D, Vt = LA.svd(np.dot(X, B_BNM))
    Q = np.dot(U[:, :k], Vt)

    return Q
