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

RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
NUM_CLASSES = 10

np.random.seed(RANDOM_SEED)

# Create an example

X = np.random.randint(0, NUM_CLASSES, SEQUENCE_LENGTH)
y = X[0]

print(f"X shape: {X.shape}")
print(f"X sample: {X[:5]}")
X = torch.from_numpy(X)
print(f"X tensor: {X[:5]}")

print(f"y value: {y}")