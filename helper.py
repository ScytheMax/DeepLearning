import numpy as np
from numpy import ndarray
from typing import Callable
from typing import List

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
