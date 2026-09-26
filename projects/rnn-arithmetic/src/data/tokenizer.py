import torch

vocab = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "<PAD>": 11,
    "<SOS>": 12,
    "<EOS>": 13,
}

inverse_vocab = {idx: token for token, idx in vocab.items()}

def encode(
        text: str,
        add_special_tokens: bool = False
) -> torch.Tensor:
    token_ids = []

    if add_special_tokens:
        token_ids.append(vocab["<SOS>"])

    for token in text:
        if token not in vocab:
            raise ValueError(f"Unknown token: {token}")
            
        token_ids.append(vocab[token])

    if add_special_tokens:
        token_ids.append(vocab["<EOS>"])

    return torch.tensor(token_ids, dtype=torch.long)

def decode(encoded: torch.Tensor) -> str:
    decoded = ""

    for token_id in encoded:
        token_id = token_id.item()

        if token_id not in inverse_vocab:
            raise ValueError(f"Unknown token id: {token_id}")

        decoded += inverse_vocab[token_id]

    return decoded 



text = "76+68"

encoded = encode(text)
decoded = decode(encoded)

print(text)
print(encoded)
print(decoded)

print(encode("144"))
print(encode("144", add_special_tokens=True))