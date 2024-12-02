import numpy as np


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the Sigmoid function."""
    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * (1 - sigmoid_x)


def softmax(array: np.ndarray) -> np.ndarray:
    return np.exp(array) / np.sum(np.exp(array))


def process_image(img):
    img = np.array(img)  # Convert the image to a numpy array
    img = img.reshape(784, 1)  # Reshape it to a 784x1 column vector
    img = img.flatten()

    return img


def one_hot_encoder(label):
    """
    Convert an integer label in the range [0, 9] to a one-hot encoded vector.

    Parameters:
    - label: Integer in the range 0-9.

    Returns:
    - A one-hot encoded column vector as a numpy array.
    """

    one_hot_vector = np.zeros(10)  # This creates a 1D array of size 10
    one_hot_vector[label] = 1  # Set the element at the index 'label' to 1

    return one_hot_vector.reshape(1, -1)  # Reshape to 1 row and 10 columns


def cross_entropy_loss(output, one_hot_label):
    """
    Calculate the cross-entropy loss between the network output and the one-hot encoded label.

    Parameters:
    - output: The output of the forward pass (probabilities from softmax).
    - one_hot_label: The one-hot encoded label (a column vector).

    Returns:
    - The cross-entropy loss (scalar value).
    """
    # Ensure output and one_hot_label have the same shape
    if output.shape != one_hot_label.shape:
        raise ValueError("Output and one-hot encoded label must have the same shape.")

    # Apply the cross-entropy loss formula
    loss = -np.sum(
        one_hot_label * np.log(output + 1e-9)
    )  # Added epsilon to avoid log(0)

    return loss
