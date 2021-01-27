#[1] pages 41-50
#[2] folder 02_fundamentals

import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple

def forward_lin_reg(X: ndarray, # m x n
                    Y: ndarray, # m x 1
                    weights: Dict[str, ndarray]
                    ) -> Tuple[float, Dict[str, ndarray]]:
    assert(X.shape[0] == Y.shape[0])
    assert(X.shape[1] == weights['W'].shape[0])
    assert(weights['B'].shape[0] == weights['B'].shape[1] == 1)

    W = weights['W']    # n x 1
    B = weights['B']
    
    N = np.dot(X, W)    # m x 1 
    P = N + B           # m x 1
    S = P - Y
    L = np.mean(np.power(S, 2))
    
    forward_info : Dict[str, ndarray] = {}
    forward_info['X'] = X
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['Y'] = Y
    
    return L, forward_info

X = np.array([[1, 1, 1],
              [1, 2, 1]])
Y = np.array([[1],
              [1.5]])

W = np.array([[0.4],
              [0.75],
              [0.3]])
B = np.array([[-0.3]])
weights : Dict[str, ndarray] = {}
weights['W'] = W
weights['B'] = B

print("res of forward lin reg", forward_lin_reg(X, Y, weights))