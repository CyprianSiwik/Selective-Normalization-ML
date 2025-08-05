# MNIST dataset loader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataset(train=True, batch_size=64, download=True, normalize=True):
    """
    Load the MNIST dataset.

    Args:
        train (bool): If True, returns the training set. Else, the test set.
        batch_size (int): Size of each batch.
        download (bool): If True, downloads the dataset if not found locally.
        normalize (bool): If True, normalizes images to mean=0.1307, std=0.3081.

    Returns:
        DataLoader: PyTorch DataLoader for MNIST.
    """

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=download,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )

    return dataloader


if __name__ == '__main__':
    train_loader = get_mnist_dataset(train=True)
    test_loader = get_mnist_dataset(train=False)

    for images, labels in train_loader:
        print("Batch of images shape:", images.shape)  # [batch_size, 1, 28, 28]
        print("Batch of labels shape:", labels.shape)  # [batch_size]
        break
