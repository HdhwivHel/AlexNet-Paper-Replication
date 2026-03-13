import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from dataset.data_pipeline import get_datasets
from models.alexnet import AlexNet
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["testing"]["batch_size"]
workers = config["testing"]["num_workers"]


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_dataset = get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = AlexNet().to(device)
    model.load_state_dict(torch.load("results/checkpoints/AlexNet.pth"))
    model.eval()

    with torch.inference_mode():
        correct = 0
        total = 0

        for X, y in tqdm(test_loader):
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    test()
