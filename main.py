import warnings

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

from mlp import MLP
from functions import *
from torchvision import datasets


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


def main():
    train_dataset, test_dataset = load_mnist()  # Load the MNIST dataset
    img, label = train_dataset[0]  # Get the first image and its label

    # Process the image (flatten it)
    test_img = process_image(img)
    test_label = one_hot_encoder(label)

    # Create the MLP
    mlp = MLP(activation_function="relu")

    # Perform a forward pass
    output = mlp.forward(test_img)

    loss = cross_entropy_loss(output, test_label)

    # print("Output:", output.shape)
    # print("label:", test_label.shape)
    print(loss)


if __name__ == "__main__":
    main()
