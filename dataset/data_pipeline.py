import torchvision.transforms.v2 as transforms
import torch


def get_datasets():
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
        ]
    )
