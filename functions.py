import numpy as np
from torchvision import datasets


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


def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) loss.

    Parameters:
        y_true (numpy.ndarray): Array of true values (targets).
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: The MSE loss.
    """
    # Ensure inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the squared differences
    squared_errors = (y_true - y_pred) ** 2

    # Return the mean of squared errors
    return np.mean(squared_errors)


def load_mnist():
    # Download and load the MNIST training dataset
    train_dataset = datasets.MNIST(
        root="./data",  # Directory to save the dataset
        train=True,  # Load training data
        transform=None,  # No transformations, keep raw PIL images
        download=True,  # Download the dataset if not already available
    )

    # Similarly, load the test dataset
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=None, download=True  # Load test data
    )

    return train_dataset, test_dataset
