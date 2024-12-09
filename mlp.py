import numpy as np
from functions import *


class MLP:
    def __init__(
        self,
        input_size=784,
        hidden_layer_sizes=[16, 16],
        output_size=10,
        activation_function="relu",
        learning_rate=0.01,
    ):
        """
        Initialize the Multi-Layer Perceptron (MLP) with specified architecture.

        Parameters:
        - input_size: Number of input neurons
        - hidden_layer_sizes: List of integers specifying the size of each hidden layer
        - output_size: Number of output neurons
        - activation_function: Activation function to use ('relu' or 'sigmoid')
        - learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            )
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        # Set activation function and its derivative
        if activation_function == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation_function == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError(
                "Unsupported activation function. Choose 'relu' or 'sigmoid'."
            )

    def forward(self, x):
        """
        Perform forward propagation through the network.

        Parameters:
        - x: Input data (numpy array)

        Returns:
        - Output of the network
        """

        # Reset activations and linear transformations for each new forward pass
        self.a = []  # Store activations
        self.z = []  # Store linear transformations

        current_activation = x
        self.a.append(current_activation)  # Input layer activation

        for i in range(len(self.weights) - 1):  # For all layers except the output layer
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = self.activation(z)
            self.z.append(z)
            self.a.append(current_activation)

        # Output layer with softmax activation
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        output = softmax(z)
        self.a.append(output)

        return output

    def back_propagation(self, x, y):
        """
        Perform backpropagation and compute gradients.

        Parameters:
        - x: Input data (numpy array)
        - y: True labels (numpy array)

        Returns:
        - Gradients for weights and biases
        """
        m = x.shape[0]  # Number of samples
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Perform forward propagation
        self.forward(x)

        # Compute gradients for the output layer
        delta = self.a[-1] - y  # When using cross-entropy + softmax
        gradients_w[-1] = np.dot(self.a[-2].T, delta) / m
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l + 1].T) * self.activation_derivative(
                self.z[l]
            )
            gradients_w[l] = np.dot(self.a[l].T, delta) / m
            gradients_b[l] = np.sum(delta, axis=0, keepdims=True) / m

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        """
        Update weights and biases using the computed gradients.

        Parameters:
        - gradients_w: Gradients for weights
        - gradients_b: Gradients for biases
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
