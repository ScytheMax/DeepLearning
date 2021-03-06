import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List

#from exercise 2
def deriv(func: Callable[[ndarray], ndarray],
          x: ndarray,
          delta: float = 0.001) -> ndarray:

    return (func(x + delta) - func(x - delta)) / (2 * delta)

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def chainDeriv(chain: Chain, #[f, g]
               x: ndarray,
               delta: float = 0.001) -> ndarray:
    assert len(chain) == 2
    return deriv(chain[0], chain[1](x), delta) * deriv(chain[1], x, delta)

def parabel(x: ndarray) -> ndarray:
    return x * x

#from exercise 5
def matmul_forward(X: ndarray, # m x n
                   W: ndarray, # n x p
                   sigma: Callable) -> ndarray:
    assert(X.shape[1] == W.shape[0]) # n == n

    N = np.dot(X, W) # m x p
    return sigma(N)

#from exercise 7
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

#from exercise 8
def backward_lin_reg(forward_info : Dict[str, ndarray], # X, N, P, Y
                     weights: Dict[str, ndarray] # W, B
                     ) -> Dict[str, ndarray]:

    # N = X * W
    # P = N + B
    # L = mean((P - Y)^2)

    batch_size = forward_info['X'].shape[0]
    
    dLdP = 2 * (forward_info['P'] - forward_info['Y']) / batch_size
    dPdN = np.ones_like(forward_info['N'])
    dNdW = np.transpose(forward_info['X'])

    dLdW = np.dot(dNdW, dLdP * dPdN)

    dPdB = 1

    dLdB = (dLdP * dPdB).sum(axis = 0)
    
    loss_gradients : Dict[str, ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB
    
    return loss_gradients
