import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.data.tokenizer import vocab
from src.data.tokenizer import encode
from src.data.generator import generator

NUM_SAMPLES = 10
MIN_DIGITS = 1
MAX_DIGITS = 3
OPERATION = "+"

class ArithmeticDataset(Dataset):

    def __init__(
        self,
        num_samples: int,
        min_digits: int,
        max_digits: int,
        operation: str = "+"
    ):
        
        self.samples = generator(
            num_samples=num_samples,
            min_digits=min_digits,
            max_digits=max_digits,
            operation=operation
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_ids = encode(
            sample["expression"]
        )

        target_ids = encode(
            sample["target"],
            add_special_tokens=True
        )

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "carry_count": sample["carry_count"],
            "max_carry_chain": sample["max_carry_chain"],
        }

def collate_fn(batch):
    input_sequences = [
        sample["input_ids"]
        for sample in batch
    ]

    target_sequences = [
        sample["target_ids"]
        for sample in batch
    ]

    padded_inputs = pad_sequence(
        input_sequences,
        batch_first=True,
        padding_value=vocab["<PAD>"]
    )

    padded_targets = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=vocab["<PAD>"]
    )

    return {
        "input_ids": padded_inputs,
        "target_ids": padded_targets,
    }

dataset = ArithmeticDataset(
    num_samples=NUM_SAMPLES,
    min_digits=MIN_DIGITS,
    max_digits=MAX_DIGITS,
    operation=OPERATION
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

batch = next(iter(loader))

print(batch["input_ids"])
print(batch["input_ids"].shape)

print(batch["target_ids"])
print(batch["target_ids"].shape)