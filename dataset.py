import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import os

class StoryDataset(Dataset):
    def __init__(self, tokenizer_dir, file_path, block_size=256):
        print("➡ Loading tokenizer")
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_dir, "vocab.json"),
            os.path.join(tokenizer_dir, "merges.txt")
        )

        print("➡ Reading text file")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print("➡ Tokenizing text")

        # Encode in chunks so tqdm can show progress
        chunk_size = 1_000  # characters per chunk
        token_ids = []

        for i in tqdm(
            range(0, len(text), chunk_size),
            desc="Tokenizing",
            unit="chunk"
        ):
            chunk = text[i : i + chunk_size]
            token_ids.extend(self.tokenizer.encode(chunk).ids)

        self.tokens = token_ids
        self.block_size = block_size

        print(f"➡ Total tokens: {len(self.tokens):,}")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
