import torch
import torch.nn as nn

from src.data.tokenizer import vocab
from src.data.dataset import batch


class ArithmeticLSTM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):
        embedded = self.embedding(x)

        output, _ = self.lstm(embedded)

        logits = self.output(output)

        return logits


model = ArithmeticLSTM(
    vocab_size=len(vocab),
    embedding_dim=32,
    hidden_dim=64
)

x = batch["input_ids"]

logits = model(x)

print("Input shape:", x.shape)
print("Output shape:", logits.shape)