from Classes.Operation import Operation
import numpy as np
from numpy import ndarray

class Sigmoid(Operation):

    def __init__(self) -> None:

        super().__init__()

    def _output(self) -> ndarray:

        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


