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
        - activation_function: String representing the activation function ("relu", "sigmoid").
        - input_size: Number of input neurons.
        - hidden_layer_sizes: List of integers representing the size of each hidden layer.
        - output_size: Number of output neurons.
        """
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activation_function = activation_function

        # Define the structure of the network
        self.layers = [input_size] + hidden_layer_sizes + [output_size]

        # Initialize weights and biases as NumPy arrays
        self.weights = []
        self.biases = []
        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        """
        Initialize weights and biases for each layer using Xavier initialization.
        Weights are scaled by sqrt(2 / (fan_in + fan_out)), and biases are set to zero.
        """
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]  # Number of input neurons
            fan_out = self.layers[i + 1]  # Number of output neurons
            # Xavier initialization for weights
            weight = np.random.randn(fan_in, fan_out) * np.sqrt(2 / (fan_in + fan_out))
            # Initialize biases to zero
            bias = np.zeros((1, fan_out))
            self.weights.append(weight)
            self.biases.append(bias)

    def _apply_activation(self, z):
        """
        Apply the activation function based on the specified type.

        Parameters:
        - z: Pre-activation value (numpy array).

        Returns:
        - Post-activation value (numpy array).
        """
        if self.activation_function == "relu":
            return relu(z)
        elif self.activation_function == "sigmoid":
            return sigmoid(z)
        else:
            raise ValueError(
                "Unsupported activation function. Use 'relu' or 'sigmoid'."
            )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
        - x: Input data (numpy array).

        Returns:
        - Activations of the output layer (numpy array).
        """
        activations = x
        for i in range(len(self.weights) - 1):  # Loop through hidden layers
            activations = self._apply_activation(
                np.dot(activations, self.weights[i]) + self.biases[i]
            )

        # Output layer (softmax activation)
        output = softmax(np.dot(activations, self.weights[-1]) + self.biases[-1])
        return output

    def back_propagation(self, x, labels):
        """
        Perform backpropagation to compute gradients for weights and biases.

        Parameters:
        - x: Input data (numpy array of shape [n_samples, n_features]).
        - y: True labels (one-hot encoded numpy array of shape [n_samples, n_classes]).

        Returns:
        - gradients_w: List of gradients for weights (same structure as self.weights).
        - gradients_b: List of gradients for biases (same structure as self.biases).
        """
