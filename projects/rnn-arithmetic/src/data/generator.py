import json
import torch
import random

NUM_SAMPLES = 10
MIN_DIGITS = 1
MAX_DIGITS = 3
OPERATION = "+"

def carry_stats(a: int, b: int):
    carry = 0
    carry_count = 0
    current_chain = 0
    max_chain = 0

    while a > 0 or b > 0:
        digit_a = a % 10
        digit_b = b % 10

        total = digit_a + digit_b + carry
        if total >= 10:
            carry = 1
            carry_count += 1

            current_chain += 1
            max_chain = max(max_chain, current_chain)
        else:
            carry = 0
            current_chain = 0
        a //= 10
        b //= 10

    return carry_count, max_chain

def random_number_with_digits(num_digits: int):

    if num_digits == 1:
        return random.randint(0, 9)

    low = 10 ** (num_digits - 1)
    high = (10 ** num_digits) - 1

    return random.randint(low, high)

def generator(
        num_samples,
        min_digits,
        max_digits,
        operation: str = "+"
):
    samples = []
    for _ in range(num_samples):
        digits_a = random.randint(min_digits, max_digits)
        digits_b = random.randint(min_digits, max_digits)

        a = random_number_with_digits(digits_a)
        b = random_number_with_digits(digits_b)

        carry_count, max_chain = carry_stats(
            a,
            b
        )

        if operation == "+":
            result = a + b
        else:
            raise ValueError(
                f"Unsupported operation: {operation}"
            )

        sample = {
            "a": a,
            "b": b,
            "expression": f"{a}{operation}{b}",
            "target": str(result),
            "digits_a": digits_a,
            "digits_b": digits_b,
            "carry_count": carry_count,
            "max_carry_chain": max_chain,
        }

        samples.append(sample)

    return samples


samples = generator(
    num_samples=NUM_SAMPLES,
    min_digits=MIN_DIGITS,
    max_digits=MAX_DIGITS,
    operation=OPERATION
)

for sample in samples:
    print(json.dumps(sample, indent=2))
