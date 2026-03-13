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

num_epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
workers = config["train_dataset"]["num_workers"]


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, _ = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )
    model = AlexNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            predicted = torch.argmax(outputs, dim=1)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
        )
        torch.save(model.state_dict(), "results/checkpoints/AlexNet.pth")


if __name__ == "__main__":
    train()
