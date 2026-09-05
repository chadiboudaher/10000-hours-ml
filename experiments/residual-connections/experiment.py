import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)


class PlainBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)

        return out

class CNN(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            *[block(64) for _ in range(num_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def train_model(model):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "loss": [],
        "accuracy": []
    }

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Loss: {epoch_loss:.4f} "
            f"Acc: {epoch_accuracy:.4f}"
        )

    return history

def evaluate_model(model):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / total
    test_accuracy = correct / total

    return test_loss, test_accuracy

experiments = {
    "Plain-4": (PlainBlock, 4),
    "Plain-16": (PlainBlock, 16),
    "Residual-4": (ResidualBlock, 4),
    "Residual-16": (ResidualBlock, 16)
}

histories = {}
results = []

for name, (block, depth) in experiments.items():

    print(f"\n{'=' * 50}")
    print(f"Training {name}")
    print(f"{'=' * 50}")

    # Reset the random seed before model initialization
    set_seed(SEED)

    model = CNN(
        block=block,
        num_blocks=depth
    ).to(device)

    history = train_model(model)

    test_loss, test_accuracy = evaluate_model(model)

    histories[name] = history

    results.append({
        "model": name,
        "blocks": depth,
        "train_loss": history["loss"][-1],
        "train_accuracy": history["accuracy"][-1],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    print(
        f"{name} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Accuracy: {test_accuracy:.4f}"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/results.csv",
    index=False
)

print("\nFinal Results")
print(results_df)