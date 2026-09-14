"""
Implement a dataset that generate a sequence of length
SEQUENCE_LENGTH and a number of classes NUM_CLASSES.

Inputs:
    - sequence_length: The length of the sequence.
    - num_classes: The different classes that can exist in a 
    sequence (for example: 0-9, etc)
"""

import numpy as np

RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
NUM_CLASSES = 10

np.random.seed(RANDOM_SEED)

# Create an example

data = np.random.randint(0, NUM_CLASSES, SEQUENCE_LENGTH)

print(f"data shape: {data.shape}")
print(f"data sample: {data[:5]}")