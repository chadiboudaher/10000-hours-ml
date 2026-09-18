import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import DelayedRecallDataset
from models import VanillaRNN, LSTM, GRU

NUM_SAMPLES = 10_000
RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
NUM_CLASSES = 10
NUM_EMBEDDINGS = 10
EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10
BATCH_SIZE = 64
RANDOM_SEED = 42

dataset = DelayedRecallDataset(
    NUM_SAMPLES,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    RANDOM_SEED
)

train_data, val_data, test_data = random_split(
    dataset,
    [8000, 1000, 1000]
)

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(val_data,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

test_loader = DataLoader(test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

model_0 = VanillaRNN(
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_embedding=NUM_EMBEDDINGS,
    embedding_dim=EMBEDDING_DIM
)

model_1 = LSTM(
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_embedding=NUM_EMBEDDINGS,
    embedding_dim=EMBEDDING_DIM
)

model_2 = GRU(
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_embedding=NUM_EMBEDDINGS,
    embedding_dim=EMBEDDING_DIM
)

criterion = nn.CrossEntropyLoss()
optimizer_0 = optim.Adam(
    model_0.parameters(),
    lr=1e-3
)

optimizer_1 = optim.Adam(
    model_1.parameters(),
    lr=1e-3
)

optimizer_2 = optim.Adam(
    model_2.parameters(),
    lr=1e-3
)

# X, y = next(iter(loader))

# logits = model(X)

# res = criterion(logits, y)
# print(res)

def train_model(
        epochs,
        dataloader,
        val_dataloader,
        criterion: nn,
        optimizer: optim,
        model: nn.Module
):
    loss_history = []
    accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for X, y in dataloader:
            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        epoch_accuracy = correct / total
        val_average_loss, val_epoch_accuracy = model_eval(model, val_dataloader, criterion)

        loss_history.append(average_loss)
        accuracy_history.append(epoch_accuracy)
        val_loss_history.append(val_average_loss)
        val_accuracy_history.append(val_epoch_accuracy)

        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Train Accuracy: {epoch_accuracy:.2f} | "
            f"Val Loss: {val_average_loss:.4f} | "
            f"Val Accuracy: {val_epoch_accuracy:.2f} | "
        )

    return (
        loss_history, 
        accuracy_history,
        val_loss_history,
        val_accuracy_history
    )

def model_eval(
        model: nn.Module,
        dataloader,
        criterion: nn,
):
    model.eval()
    with torch.inference_mode():
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X, y in dataloader:
            logits = model(X)
            loss = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
            
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        epoch_accuracy = correct / total

    return average_loss, epoch_accuracy