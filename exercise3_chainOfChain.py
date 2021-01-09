#[1] pages 13-18
#[2] folder 01_foundations\Code.ipynb "Basic functions"

import helper as ms
import numpy as np
from numpy import ndarray
from typing import Callable
from typing import List

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def chainDeriv_3(chain: Chain, #[f, g, h]
                 input_: ndarray,
                 diff: float = 0.001) -> ndarray:
    assert len(chain) == 3
    # d[f(g(h(x)))] / dx = f'(g(h(x))) * d[g(h(x))] / dx = f'(g(h(x))) * g'(h(x)) * h'(x)
    return ms.deriv(chain[0], chain[1](chain[2](input_)), diff) * \
           ms.deriv(chain[1], chain[2](input_), diff) * \
           ms.deriv(chain[2], input_, diff)

def chainDeriv_3_with_chainDeriv_2(chain: Chain, #[f, g, h]
                 input_: ndarray,
                 diff: float = 0.001) -> ndarray:
    assert len(chain) == 3
    f = chain[0]
    g_h = lambda input_ : chain[1](chain[2](input_))

    # d[f(g(h(x)))] / dx = f'(g(h(x))) * d[g(h(x))] / dx
    return ms.deriv(f, g_h(input_), diff) * ms.chainDeriv([chain[1], chain[2]], input_, diff)

def g(input_: ndarray) -> ndarray: # x + 3
    return input_ + 3

def h(input_: ndarray) -> ndarray: # x^3 - 5
    return np.power(input_, 3) - 5

def f_g_h(input_: ndarray) -> ndarray: # f(g(h(x))) = (x^3 - 5 + 3)^2 =
                                       # (x^3 - 2)^2= x^6 - 4x^3 + 4
    return np.power(input_, 6) - 4 * np.power(input_, 3) + 4


x = np.array([1, 2, 3, 4])
print("x values", x, ", diff is 0.001")
print("f_g_h(x)", f_g_h(x))
print("f(g(h(x)))", ms.parabel(g(h(x))), '\n')

print("deriv of f_g_h at x", ms.deriv(f_g_h, x))
print("chainDeriv_3 of [f, g, h] at x", chainDeriv_3([ms.parabel, g, h], x))
print("chainDeriv_3 with chainDeriv_2 of [f, g, h] at x", 
      chainDeriv_3_with_chainDeriv_2([ms.parabel, g, h], x))