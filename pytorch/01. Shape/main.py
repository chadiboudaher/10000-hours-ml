import torch

# Multiple shape found in Deep Learning

# Images
# [batch, channels, height, width]

# Sequence
# [batch, time, features]

# Video
# [batch, time, channels, height, width]

# Section 1 - Reading Shapes

x = torch.randn(32, 100)

print(f"Dimension is {x.ndim}")
print(f"Shape is {x.shape}")
# If this is tabular data, what could 32 represent?
# It could represent the number of rows (number of samples).
print(f"The number of features: {x.shape[1]}")

# Section 2 - Indexing and Slicing

x = torch.randn(32, 3, 64, 64)

print("="*50)

print(f"The shape of x is {x.shape}")
print(f"the shape of x[0] is {x[0].shape}")
print(f"the shape of x[1] is {x[1].shape}")
print("*"*20)
print(f"the shape of x[0, 1] is {x[0, 1].shape}")
print(f"the shape of x[0, 1, 2] is {x[0, 1, 2].shape}")
print("*"*20)
print(f"the shape of x[:5] is {x[:5].shape}")
print(f"the shape of x[:5, 1] is {x[:5, 1].shape}")
print(f"the shape of x[:5, 0, 0] is {x[:5, 0, 0].shape}")
print("#"*20)
print(f"the shape of x[:, 0] is {x[:, 0].shape}")
# Expected: [32, 64, 64]
# True shape: [32, 64, 64]
print(f"the shape of x[:, 0:1] is {x[:, 0:1].shape}")
# Expected: [32, 1, 64, 64]
# True shape: [32, 1, 64, 64]
print(f"the shape of x[0, :, :32, :32] is {x[0, :, :32, :32].shape}")
# Expected: [3, 32, 32]
# True shape: [3, 32, 32]