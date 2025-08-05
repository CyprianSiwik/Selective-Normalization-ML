# CIFAR dataset loader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_cifar_dataset(name='cifar10', train=True, batch_size=128, download=True):
    """
    Load CIFAR-10 or CIFAR-100 dataset.

    Args:
        name (str): 'cifar10' or 'cifar100'
        train (bool): Whether to load the training set or test set.
        batch_size (int): Batch size for the DataLoader.
        download (bool): Whether to download the dataset if not present.

    Returns:
        DataLoader: DataLoader for the requested dataset.
    """
    # Define transform pipeline (you can customize as needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=train, transform=transform, download=download)
    elif name.lower() == 'cifar100':
        dataset = datasets.CIFAR100(root='./data', train=train, transform=transform, download=download)
    else:
        raise ValueError("Dataset name must be 'cifar10' or 'cifar100'")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
    return dataloader


if __name__ == '__main__':
    train_loader = get_cifar_dataset('cifar10', train=True)
    test_loader = get_cifar_dataset('cifar100', train=False)

    # Example usage
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
