#[1] pages 30-39
#[2] folder 01_foundations\Code.ipynb "Basic functions"

import helper as ms
import numpy as np
from numpy import ndarray
from typing import Callable

def matmul_forward_sum(X: ndarray, # m x n
                       W: ndarray, # n x p
                       sigma: Callable) -> ndarray:

    S = ms.matmul_forward(X, W, sigma)
    return np.sum(S)

# backward means derivation
def matmul_backward_sum(X: ndarray, # m x n 
                        W: ndarray, # n x p
                        sigma: Callable) -> ndarray:
    N = np.dot(X, W) # m x p
    S = sigma(N)     # m x p
    # would be the same calc how, but not so efficient: S = ms.matmul_forward(X, W, sigma) 
    
    dSdN = ms.deriv(sigma, N) # delta = 0.001
    # dLdN = dSdN * dLdS = dSdN

    dLdX = np.dot(dSdN, np.transpose(W))  # (m x p) * (p x n) = m x n
    dLdW = np.dot(np.transpose(X), dSdN)  # (n x m) * (m x p) = n x p
    
    return dLdX, dLdW

X = np.array([[1,2,3], # 2 x 3
             [4,5,6]])
W = np.array([[1,1,2,3], # 3 x 4
              [-1,1,2,3],
              [1,-1,2,3]])

print("X ", X)
print("W ", W)
print("N ", np.dot(X, W)) # 2 x 4
print("matmul_forward_sum ", matmul_forward_sum(X, W, ms.parabel))
print("matmul_backward_sum ", matmul_backward_sum(X, W, ms.parabel))