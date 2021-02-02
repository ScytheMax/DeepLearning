#[1] pages 51-56
#[2] folder 02_fundamentals

import helper as ms
import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple
import copy

def train_lin_reg(X: ndarray,
                  Y: ndarray,
                  weights: Dict[str, ndarray],
                  iteration: int = 1000,
                  learning_rate: float = 0.001
                  ) -> Dict[str, ndarray]:
    
    weights_updated = copy.deepcopy(weights)

    for i in range(iteration):
        [L, forward_info] = ms.forward_lin_reg(X, Y, weights_updated)
        loss_grads = ms.backward_lin_reg(forward_info, weights_updated)
        for key in weights_updated.keys():
            weights_updated[key] -= learning_rate * loss_grads[key]

    return weights_updated

def train_lin_reg_rand_w(X: ndarray,
                         Y: ndarray,
                         iteration: int = 1000,
                         learning_rate: float = 0.001
                         ) -> Dict[str, ndarray]:
    
    W = np.random.randn(X.shape[1], 1)
    B = np.random.randn(1, 1)
    weights : Dict[str, ndarray] = {}
    weights['W'] = W
    weights['B'] = B

    for i in range(iteration):
        [L, forward_info] = ms.forward_lin_reg(X, Y, weights)
        loss_grads = ms.backward_lin_reg(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    return weights

def train_examples_w_param_vs_rand(X: ndarray,
                                   Y: ndarray,
                                   weights: Dict[str, ndarray],
                                   iteration: int = 1000,
                                   learning_rate: float = 0.001
                                   ):

    print('##################################')
    print("train with learning_rate", learning_rate, "and", iteration, "iterations")

    weights_updated = train_lin_reg(X, Y, weights, iteration, learning_rate)    
    [L, forward_info] = ms.forward_lin_reg(X, Y, weights_updated) 
    print("L", L)
    print("W_upt", weights_updated['W'])
    print("B_upt", weights_updated['B'], '\n')

    weights_updated_rand = train_lin_reg_rand_w(X, Y, iteration, learning_rate)
    [L_rand, forward_info] = ms.forward_lin_reg(X, Y, weights_updated_rand)
    print("L_rand", L_rand)
    print("W_upt_rand", weights_updated_rand['W'])
    print("B_upt_rand", weights_updated_rand['B'], '\n')

X = np.array([[1, 1, 1], [1, 2, 1]])
Y = np.array([[1], [3]])
W = np.array([[0.4], [0.5], [0.3]])
B = np.array([[-0.3]])
weights : Dict[str, ndarray] = {}
weights['W'] = W
weights['B'] = B
iteration = 1000
learning_rate = 0.001

[L, forward_info] = ms.forward_lin_reg(X, Y, weights)
print("L", L)
print("W", weights['W'])
print("B", weights['B'])

train_examples_w_param_vs_rand(X, Y, weights, 100, 0.01)
train_examples_w_param_vs_rand(X, Y, weights, 100, 0.1)

train_examples_w_param_vs_rand(X, Y, weights, 1000, 0.001)
train_examples_w_param_vs_rand(X, Y, weights, 1000, 0.01)
train_examples_w_param_vs_rand(X, Y, weights, 1000, 0.1)

train_examples_w_param_vs_rand(X, Y, weights, 10000, 0.001)
train_examples_w_param_vs_rand(X, Y, weights, 10000, 0.01)