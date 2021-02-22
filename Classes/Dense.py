import Layer
import Operation

class Dense(Layer):

    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()):

        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:

        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None

