import numpy as np
from functions import relu, sigmoid, softmax


class MLP:
    def __init__(
        self,
        activation_function,
        input_size=784,
        hidden_layer_sizes=[16, 16],
        output_size=10,
    ):
        """
        Initialize the MLP with given layer sizes.

        Parameters:
        - input_size: Number of input neurons.
        - hidden_layer_sizes: List of integers representing the size of each hidden layer.
        - output_size: Number of output neurons.
        """
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activation_function = activation_function

        self.layers = [input_size] + hidden_layer_sizes + [output_size]
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_weights_and_biases()  # Initialize weights and biases

    def _initialize_weights_and_biases(self):
        """
        Initialize weights and biases for each layer.
        Weights are initialized with small random values, and biases are set to zero.
        """
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _apply_activation(self, z):
        if self.activation_function == "relu":
            return relu(z)
        elif self.activation_function == "sigmoid":
            return sigmoid(z)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
        - x: Input data (numpy array).

        Returns:
        - Activations of the output layer.
        """
        activations = x
        for i in range(len(self.weights) - 1):  # loop through layers
            activations = self._apply_activation(
                np.dot(activations, self.weights[i]) + self.biases[i]
            )

        # Output layer
        output = softmax(np.dot(activations, self.weights[-1]) + self.biases[-1])
        return output
