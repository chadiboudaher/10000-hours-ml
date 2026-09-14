"""
Implement a dataset that generate a sequence of length
SEQUENCE_LENGTH and a number of classes NUM_CLASSES.

Inputs:
    - sequence_length: The length of the sequence.
    - num_classes: The different classes that can exist in a 
    sequence (for example: 0-9, etc)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

NUM_SAMPLES = 100
RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
NUM_CLASSES = 10
BATCH_SIZE = 16

np.random.seed(RANDOM_SEED)

# Create an example

# X = np.random.randint(0, NUM_CLASSES, SEQUENCE_LENGTH)
# y = X[0]

# print(f"X shape: {X.shape}")
# print(f"X sample: {X[:5]}")
# X = torch.from_numpy(X)
# print(f"X tensor: {X[:5]}")

# print(f"y value: {y}")

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

# X, y = dataset[0]

# print(f"data shape: {X.shape}")
# print(f"X sample: {X}")
# print(f"y value: {y}")

# X_res, y_res = dataset.__getitem__(45)

# print(f"X res sample: {X_res}")
# print(f"y res value: {y_res}")

# dataset_length = dataset.__len__()
# print(f"dataset length: {dataset_length}")

loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True)

X_batch, y_batch = next(iter(loader))

print(f"x batch shape: {X_batch.shape}")
print(f"y batch shape: {y_batch.shape}")
print(f"Number of batch collections: {len(loader)}")