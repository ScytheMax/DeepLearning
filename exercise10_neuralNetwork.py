#[1] pages 57-64
#[2] folder 02_fundamentals

import helper as ms
import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-1.0 * x / 1000)) # compression of x values, because x become very high

def init_weights(x_param_size: int, 
                 lin_reg_size: int) -> Dict[str, ndarray]:

    weights: Dict[str, ndarray] = {}
    weights['W1'] = np.random.randn(x_param_size, lin_reg_size) # 3, 13
    weights['B1'] = np.random.randn(1, lin_reg_size)            # 1, 13
    weights['W2'] = np.random.randn(lin_reg_size, 1)            # 13, 1
    weights['B2'] = np.random.randn(1, 1)                       # 1, 1
    return weights

def forward_neural_network(X: ndarray,
                           y: ndarray,
                           weights: Dict[str, ndarray]
                           ) -> Tuple[float, Dict[str, ndarray]]:

    M1 = np.dot(X, weights['W1'])   # 5, 3 x 3, 13

    N1 = M1 + weights['B1']         # 5, 13

    O1 = sigmoid(N1)                # 5, 13
    
    M2 = np.dot(O1, weights['W2'])  # 5, 1

    P = M2 + weights['B2']          # 5, 1

    loss = np.mean(np.power(P - y, 2))

    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P'] = P
    forward_info['y'] = y

    return loss, forward_info

def backward_neural_network(forward_info: Dict[str, ndarray], 
                            weights: Dict[str, ndarray]) -> Dict[str, ndarray]:

    batch_size = forward_info['X'].shape[0]                             # 5

    dLdP = 2 * (forward_info['P'] - forward_info['y']) / batch_size     # 5
    
    dPdM2 = np.ones_like(forward_info['M2'])                            # 5

    dLdM2 = dLdP * dPdM2                                                # 5, 1
  
    dPdB2 = np.ones_like(weights['B2'])                                 # 1, 1

    dLdB2 = (dLdP * dPdB2).sum(axis=0)                                  # 1,
    
    dM2dW2 = np.transpose(forward_info['O1'], (1, 0))                   # 13, 5
    
    dLdW2 = np.dot(dM2dW2, dLdP)                                        # 13, 1

    dM2dO1 = np.transpose(weights['W2'])                                # 1, 13

    dLdO1 = np.dot(dLdM2, dM2dO1)                                       # 5, 13
    
    dO1dN1 = sigmoid(forward_info['N1']) * (1- sigmoid(forward_info['N1'])) # 5, 13
    
    dLdN1 = dLdO1 * dO1dN1                                              # 5, 13
    
    dN1dB1 = np.ones_like(weights['B1'])                                # 1, 13
    
    dN1dM1 = np.ones_like(forward_info['M1'])                           # 5, 13
    
    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)                                # 13,
    
    dLdM1 = dLdN1 * dN1dM1                                              # 5, 13
    
    dM1dW1 = np.transpose(forward_info['X'], (1, 0))                    # 3, 5

    dLdW1 = np.dot(dM1dW1, dLdM1)                                       # 3, 13

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W2'] = dLdW2                # 13, 1
    loss_gradients['B2'] = dLdB2.sum(axis=0)    # 1,
    loss_gradients['W1'] = dLdW1                # 3, 13
    #
    # attention: example code in [2] sum over all 13 entries, but B1 is of dimension 13, 1
    #
    loss_gradients['B1'] = dLdB1                # 13, 1
    
    return loss_gradients

def train_neural_network(X_train: ndarray, 
                         y_train: ndarray,
                         n_iter: int = 1000,
                         learning_rate: float = 0.01
                         ) -> Dict[str, ndarray]:

    weights = init_weights(X_train.shape[1], 13)
    loss, forward_info = forward_neural_network(X_train, y_train, weights)
    print('L before training', loss)

    for i in range(n_iter):
        loss, forward_info = forward_neural_network(X_train, y_train, weights)
        loss_grads = backward_neural_network(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]  

    return weights

# example real estate market

#             rooms,   rent,  dist to supermarket
X = np.array([[1.5,     10,     4],
              [3.5,     20,     7],
              [3,       15,     1],
              [5,       22,     14],
              [2,       16,     3.5]])

#           purchase price
Y = np.array([[100], 
              [250], 
              [300], 
              [400], 
              [120]])



weights = train_neural_network(X, Y)
print('weights', weights)
loss, forward_info = forward_neural_network(X, Y, weights)
print('L trained', loss)