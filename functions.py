import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * (1 - sigmoid_x)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def load_mnist(batch_size=64):
    """
    Load the MNIST dataset and return both the training and test data loaders.

    Parameters:
    - batch_size: The number of samples per batch to load.

    Returns:
    - train_loader: DataLoader for the training dataset
    - test_loader: DataLoader for the test dataset
    """
    # Define the transformation to convert PIL images to tensors and optionally normalize them
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize pixel values to [-1, 1]
        ]
    )

    # Download and load the MNIST training dataset with the transformations
    train_dataset = datasets.MNIST(
        root="./data",  # Directory to save/load the dataset
        train=True,  # Load training data
        transform=transform,
        download=True,  # Download the dataset if not already available
    )

    # Download and load the MNIST test dataset with the transformations
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True  # Load test data
    )

    # Create data loaders for both training and testing datasets
    # shuffle=True ensures the data is shuffled at every epoch for the training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
