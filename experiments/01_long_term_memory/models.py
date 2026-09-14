"""
Important Questions:
    1. Why do we even need (not necessarly in this situation) an embedding here 
    instead of feeding the digit directly into the RNN.
    Answer: The embedding learns useful feature representations for each token 
    (integer values),while the RNN learns how to use and preserve those features 
    across the sequence to make the final predictions.

nnEmbedding:
    - num_embeddings (int): size of the dictionary of embeddings in our case 10.
    - embedding_dim (int): the size of each embedding vector.
"""


import torch
import torch.nn as nn
from dataset import DelayedRecallDataset, loader

NUM_EMBEDDINGS = 10
EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10
RANDOM_SEED = 42

embedding = nn.Embedding(
    NUM_EMBEDDINGS,
    EMBEDDING_DIM
)

class VanillaRNN(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 output_size, 
                 num_embedding,
                 embedding_dim,
                 num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embedding,
            embedding_dim
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh"
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = VanillaRNN(
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_embedding=NUM_EMBEDDINGS,
    embedding_dim=EMBEDDING_DIM
)

dataset = DelayedRecallDataset(
    num_samples=100,
    num_classes=10,
    seq_length=50,
    random_seed=RANDOM_SEED
)

X, y = next(iter(loader))

logits = model(X)
print(logits)