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
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier initialization for weights
            std_dev = np.sqrt(2 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * std_dev)
            # Initialize biases to zero
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        # Set activation function and its derivative
        if activation_function == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation_function == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

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

        self.a.append(x)  # Input layer activation

        # Iterate over all layers except last one
        for i in range(len(self.weights) - 1):
            # Compute the linear transformation (z) for the current layer
            z = np.dot(x, self.weights[i]) + self.biases[i]  # z = x * W + b
            x = self.activation(z)  # Apply the activation function
            self.z.append(z)  # Store the linear transformation
            self.a.append(x)  # Store the activation

        # Output layer with softmax activation
        z = np.dot(x, self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        output = softmax(z)
        self.a.append(output)

        return output

    def back_propagation(self, x, y):
        """
        Perform backpropagation and compute gradients for weights and biases.

        Parameters:
        - x: Input data (numpy array)
            Shape: (batch_size, input_features)
        - y: True labels (numpy array)
            Shape: (batch_size, output_classes)

        Returns:
        - gradients_w: List of gradients for each weight matrix
                    Each element has shape corresponding to its weight matrix
        - gradients_b: List of gradients for each bias vector
                    Each element has shape corresponding to its bias vector
        """
        m = x.shape[0]  # batch_size

        # The variables gradients with zeros
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Perform forward propagation
        # self.forward(x)

        # Compute gradients for the output layer
        delta = self.a[-1] - y  # When using cross-entropy + softmax
        gradients_w[-1] = (
            np.dot(self.a[-2].T, delta) / m
        )  # Gradient of weights: (a_prev)^T * delta / m
        gradients_b[-1] = (
            np.sum(delta, axis=0, keepdims=True) / m
        )  # Gradient of biases: sum(delta) / m

        # Backpropagate through hidden layers
        for l in range(
            len(self.weights) - 2, -1, -1
        ):  # Starting from the second last layer down to the first hidden layer
            delta = np.dot(delta, self.weights[l + 1].T) * self.activation_derivative(
                self.z[l]
            )  # delta = (delta_next_layer * W_next_layer^T) * activation_derivative(z_current_layer)
            gradients_w[l] = np.dot(self.a[l].T, delta) / m
            gradients_b[l] = np.sum(delta, axis=0, keepdims=True) / m

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        """
        Update weights and biases using the obtained gradients.

        Parameters:
        - gradients_w: Gradients for weights
        - gradients_b: Gradients for biases
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def evaluate(self, x, y):
        """
        Evaluate the trained model on given data.

        Parameters:
        - x: Input data (numpy array of shape [n_samples, input_size])
        - y: True labels (numpy array of shape [n_samples], integer labels)

        Returns:
        - accuracy: the accuracy of the model on the provided dataset
        """
        output = self.forward(x)  # forward pass
        predictions = np.argmax(output, axis=1)  # predicted labels
        accuracy = np.mean(predictions == y)  # compute accuracy
        return accuracy
