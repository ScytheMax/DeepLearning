from Classes.Operation import Operation
import numpy as np
from numpy import ndarray
from Classes.Operation import assert_same_shape

class ParamOperation(Operation):

    def __init__(self, param: ndarray) -> ndarray:

        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError()

