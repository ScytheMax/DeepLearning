#[1] pages 51-56
#[2] folder 02_fundamentals

import helper as ms
import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple

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

X = np.array([[1, 1, 1],
              [1, 2, 1]])
Y = np.array([[1],
              [3]])

W = np.array([[0.4],
              [0.5],
              [0.3]])
B = np.array([[-0.3]])
weights : Dict[str, ndarray] = {}
weights['W'] = W
weights['B'] = B

learning_rate = 0.001

for i in range(10):
    res = ms.forward_lin_reg(X, Y, weights)
    print('L:', res[0])
    print('W: ', weights['W'])
    print('B: ', weights['B'])

    loss_grads = backward_lin_reg(res[1], weights)
    for key in weights.keys():
        weights[key] -= learning_rate * loss_grads[key]