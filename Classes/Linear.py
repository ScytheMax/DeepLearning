from Classes.Operation import Operation
from numpy import ndarray

class Linear(Operation):

    def __init__(self) -> None:
     
        super().__init__()

    def _output(self) -> ndarray:

        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad