import torch
from torchvision import datasets, transforms


def get_transforms(model_name: str | None = None):
    if model_name == "model3":
        train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
            transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15.0, 15.0), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    return train_transforms, test_transforms


def get_datasets(data_dir: str = "./data", model_name: str | None = None):
    train_transforms, test_transforms = get_transforms(model_name)
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=test_transforms)
    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    model_name: str | None = None,
):
    train_dataset, test_dataset = get_datasets(data_dir, model_name)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader



