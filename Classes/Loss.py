
class Loss(object):

    def __init__(self):
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:

        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:

        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:

        raise NotImplementedError()

    def _input_grad(self) -> ndarray:

        raise NotImplementedError()