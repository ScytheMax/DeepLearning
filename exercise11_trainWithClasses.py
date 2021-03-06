#[1] pages 96-97
#[2] folder 03_dlfs

from Classes.NeuralNetwork import NeuralNetwork
from Classes.SGD import SGD
from Classes.Dense import Dense
from Classes.Sigmoid import Sigmoid
from Classes.Linear import Linear
from Classes.MeanSquaredError import MeanSquaredError
from Classes.Trainer import Trainer
import numpy as np
from numpy import ndarray

### test data ###
from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target

# Scaling the data
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

def to_2d_np(a: ndarray, 
          type: str="col") -> ndarray:

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

def mae(y_true: ndarray, y_pred: ndarray):
    '''
    Compute mean absolute error for a neural network.
    '''    
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray):
    '''
    Compute root mean squared error for a neural network.
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):

    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))
### ###

optimizer = SGD(lr = 0.01)

lin_reg = NeuralNetwork(
    layers = [Dense(neurons = 1,
                    activation = Linear())],
    loss = MeanSquaredError(),
    seed = 20190501
)

trainer = Trainer(lin_reg, optimizer)

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(lin_reg, X_test, y_test)


nn = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

trainer = Trainer(nn, optimizer)

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(nn, X_test, y_test)


deep_neural_network = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

trainer = Trainer(deep_neural_network, optimizer)

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(deep_neural_network, X_test, y_test)