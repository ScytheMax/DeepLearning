import Loss

class MeanSquaredError(Loss):

    def __init__(self) -> None:

        super().__init__()

    def _output(self) -> float:

        loss = (
            np.sum(np.power(self.prediction - self.target, 2)) / 
            self.prediction.shape[0]
        )

        return loss

    def _input_grad(self) -> ndarray:     

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]