#[1] pages 19-23
#[2] folder 01_foundations\Code.ipynb "Basic functions"

import helper as ms
import numpy as np
from numpy import ndarray
from typing import Callable

def multipleParameters(x: ndarray,
                       y: ndarray,
                       sigma: Callable) -> ndarray:
    assert(x.shape == y.shape)
    # in book: x + y
    alpha = x * y
    return sigma(alpha)

def derivMulPar(x: ndarray,
                y: ndarray,
                sigma: Callable) -> ndarray:
    assert(x.shape == y.shape)
    #in book: deriv of (x+y)/dx is 1
    alpha = x * y
    dx = ms.deriv(sigma, alpha) * y
    dy = ms.deriv(sigma, alpha) * x

    return dx, dy

def f_sigma(a: ndarray) -> ndarray:
    return a * a - 3

x = np.array([1,2,3,4])
y = np.array([1.5,2.2,3.5,4.5])
print("x: ", x)
print("y: ", y)
print("sigma(x + y): ", multipleParameters(x, y, f_sigma))
print("dx, dy of sigma(x + y): ", derivMulPar(x, y, f_sigma))