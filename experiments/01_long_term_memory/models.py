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

NUM_EMBEDDINGS = 10
EMBEDDING_DIM = 32

embedding = nn.Embedding(
    NUM_EMBEDDINGS,
    EMBEDDING_DIM
)