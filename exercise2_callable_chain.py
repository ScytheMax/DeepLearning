#[1] pages 7-12
#[2] folder 01_foundations

import numpy as np
from numpy import ndarray
from typing import Callable
from typing import List

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          diff: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the "input_" array.
    '''
    return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def chainDeriv(chain: Chain, #[f, g]
               input_: ndarray,
               diff: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a two element chain function at every element in the "input_" array.
    h(x) = (f o g)(x)
    h'(x) = f'(g(x)) * g'(x)
    '''
    return deriv(chain[0], chain[1](input_)) * deriv(chain[1], input_)

def parabel(input_: ndarray) -> ndarray:
    return input_ * input_

def polynomFourthDegree(input_: ndarray) -> ndarray:
    return parabel(input_) * parabel(input_)

x = np.array([1, 2, 3, 4])
print("x values", x)
print("parabel of x values", parabel(x))
print("fourth degree of x values", polynomFourthDegree(x), '\n')

print("approximately derivate of parabel at x values with diff = 0.001\n", 
      deriv(parabel, x))
# is the same, because derivate of the parabel is linear
print("approximately derivate of parabel at x values with diff = 0.000001\n", 
      deriv(parabel, x, 0.000001), '\n')

print("approximately derivate of polynom of fourth degree at x values with diff = 0.001\n", 
      deriv(polynomFourthDegree, x, 0.001))

chain = [parabel, parabel]
print("approximately derivate of polynom of fourth degree at x values with diff = 0.001\n with chain rule of parabel o parabel", 
      chainDeriv(chain, x, 0.001), '\n')