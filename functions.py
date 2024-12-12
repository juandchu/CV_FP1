import numpy as np
from torchvision import datasets
from torchvision import transforms


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


def softmax(x):
    # x shape: (batch_size, output_size)
    shifted = x - np.max(x, axis=1, keepdims=True)  # Numeric stability
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def process_image(img):
    img = np.array(img)  # Convert the image to a numpy array
    img = img.reshape(784, 1)  # Reshape it to a 784x1 column vector
    img = img.flatten()

    return img


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
    # Define the transformation to convert PIL images to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,)),  # Optional: normalize the data
        ]
    )

    # Download and load the MNIST training dataset with the transformation
    train_dataset = datasets.MNIST(
        root="./data",  # Directory to save the dataset
        train=True,  # Load training data
        transform=transform,  # Apply transformation
        download=True,  # Download the dataset if not already available
    )

    # Similarly, load the test dataset
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True  # Load test data
    )

    return train_dataset, test_dataset


def cross_entropy_loss(y_true, y_pred):
    # y_true and y_pred are shape (batch_size, number_of_classes)
    # Avoid log(0) with a small epsilon
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
