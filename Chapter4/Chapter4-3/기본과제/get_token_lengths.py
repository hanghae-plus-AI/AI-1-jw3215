import numpy as np

def get_token_lengths(dataset, tokenizer):
    lengths = []
    for example in dataset:
        formatted_text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}"
        length = len(tokenizer(formatted_text, truncation=False)["input_ids"])
        lengths.append(length)
    return lengths
