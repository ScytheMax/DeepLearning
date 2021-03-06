#[1] pages 24-29
#[2] folder 01_foundations

import helper as ms
import numpy as np
from numpy import ndarray
from typing import Callable

def matmul_forward(X: ndarray, # m x n
                   W: ndarray, # n x p
                   sigma: Callable) -> ndarray:
    assert(X.shape[1] == W.shape[0]) # n == n

    N = np.dot(X, W)
    return sigma(N)

# maybe not complete correctly, look for next exercise with sum
def derivMulMatSigma(X: ndarray, # m x n
                     W: ndarray, # n x p
                     sigma: Callable) -> ndarray:
    assert(X.shape[1] == W.shape[0])

    N = np.dot(X, W)
    dSdu = ms.deriv(sigma, N)
    return np.dot(dSdu, np.transpose(W))

X = np.array([[1,2,3], # 2 x 3
             [4,5,6]])
W = np.array([[1,1,2,3], # 3 x 4
              [-1,1,2,3],
              [1,-1,2,3]])

print("X ", X)
print("W ", W)
print("mulMatSig(X, W, parabel): ", matmul_forward(X, W, ms.parabel)) # 2 x 4
print("mulMatSig(X, W, parabel): ", derivMulMatSigma(X, W, ms.parabel)) # 2 x 3, same like X

