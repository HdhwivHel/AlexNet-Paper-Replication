from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
from dataset.data_pipeline import get_datasets
from models.alexnet import AlexNet
import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
CHECKPOINT_PATH = BASE_DIR / "results" / "checkpoints" / "AlexNet.pth"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

num_epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
workers = config["train_dataset"]["num_workers"]


def train():
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Windows can hit CUDA pin-memory thread issues with multi-worker DataLoader.
    use_pin_memory = device.type == "cuda" and not (os.name == "nt" and workers > 0)

    train_dataset, _ = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_pin_memory,
        persistent_workers=workers > 0,
    )
    model = AlexNet().to(device)
    should_compile = hasattr(torch, "compile") and device.type == "cuda"
    if should_compile:
        model = torch.compile(model)
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
            X = X.to(device, non_blocking=use_pin_memory)
            y = y.to(device, non_blocking=use_pin_memory)
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
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(model_to_save.state_dict(), CHECKPOINT_PATH)


if __name__ == "__main__":
    train()
