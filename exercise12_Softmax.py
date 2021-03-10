#[1] pages 111f
#[2] folder 04_extensions

import numpy as np
from Extensions.layers import Dense
from Extensions.losses import SoftmaxCrossEntropy, MeanSquaredError
from Extensions.optimizers import Optimizer, SGD, SGDMomentum
from Extensions.activations import Sigmoid, Tanh, Linear, ReLU
from Extensions.network import NeuralNetwork
from Extensions.train import Trainer
from Extensions.utils import mnist
from Extensions.utils.np_utils import softmax

# had to download it manually, because of  http 403 message: ' The request was a legal request, but the server is refusing to respond to it.'
# mnist.download_mnist()
# only save the downloaded files once at the beginning in a pkl file
# mnist.save_mnist()

X_train, y_train, X_test, y_test = mnist.load()
# use only one tenth of the input data
X_train = X_train[1:6000]
y_train = y_train[1:6000]
X_test = X_test[1:1000]
y_test = y_test[1:1000]

num_labels = len(y_train)
print('num_label y_train', num_labels)
train_labels = np.zeros((num_labels, 10))

for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
print('num_labels y_test', num_labels)

test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1

#Scale data to mean 0, variance 1
X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
print('np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test)', np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test))

std_x_train = np.std(X_train)
X_train, X_test = X_train / std_x_train, X_test / std_x_train
print('np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test)', np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test))

# maybe the main weigth parameter is calculated?
def calc_accuracy_model(model, test_set):
    #print(np.argmax(model.forward(test_set, inference=True), axis=1))
    #print(np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test))
    #print(np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum())
    #print(np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0)
    return print(f'''The model validation accuracy is: {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')

# Softmax cross entropy vs MeanSquaredError
# with sigmoid activation

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Sigmoid())],
            loss = MeanSquaredError(normalize=False), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

print()
print()

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Sigmoid()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 130,
            eval_every = 1,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

print()
print()

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=ReLU()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 130,
            eval_every = 1,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

print()
print()

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 130,
            eval_every = 1,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)


