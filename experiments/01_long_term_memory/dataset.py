"""
Implement a dataset that generate a sequence of length
SEQUENCE_LENGTH and a number of classes NUM_CLASSES.

Inputs:
    - sequence_length: The length of the sequence.
    - num_classes: The different classes that can exist in a 
    sequence (for example: 0-9, etc)
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

NUM_SAMPLES = 100
RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
NUM_CLASSES = 10
BATCH_SIZE = 16

class DelayedRecallDataset(Dataset):
    def __init__(self,
                 num_samples: int,
                 num_classes: int,
                 seq_length: int,
                 random_seed: int):
        super().__init__()

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)

        self.X = torch.randint(0,
                          self.num_classes,
                          size=(self.num_samples, self.seq_length))
        self.y = self.X[:, 0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = DelayedRecallDataset(
    NUM_SAMPLES,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    RANDOM_SEED
)

loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True)

X_batch, y_batch = next(iter(loader))
