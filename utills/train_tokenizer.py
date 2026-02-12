from tokenizers import ByteLevelBPETokenizer
import os

# -----------------------
# Paths
# -----------------------
data_files = [
    "../data_clean/train.txt",
    "../data_clean/eval.txt",
    "../data_clean/test.txt"
]

tokenizer_dir = "../tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

# -----------------------
# Text normalization (important for better merges)
# -----------------------
def normalize_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace("\r\n", "\n").strip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

for f in data_files:
    normalize_file(f)

# -----------------------
# Special tokens
# -----------------------
special_tokens = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<MASK>"
]

# -----------------------
# Initialize tokenizer
# -----------------------
tokenizer = ByteLevelBPETokenizer(
    lowercase=False,
    add_prefix_space=True
)

# -----------------------
# Train tokenizer
# -----------------------
tokenizer.train(
    files=data_files,
    vocab_size=8000,
    min_frequency=2,
    special_tokens=special_tokens
)

# -----------------------
# Padding & truncation
# -----------------------
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("<PAD>"),
    pad_token="<PAD>"
)

tokenizer.enable_truncation(max_length=512)

# -----------------------
# Save tokenizer
# -----------------------
tokenizer.save_model(tokenizer_dir)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

# -----------------------
# Save README for reproducibility
# -----------------------
with open(os.path.join(tokenizer_dir, "README.txt"), "w") as f:
    f.write(
        "ByteLevel BPE Tokenizer\n"
        "vocab_size=15000\n"
        "min_frequency=2\n"
        "special_tokens=<PAD>, <BOS>, <EOS>, <UNK>, <MASK>\n"
        f"pad_id={tokenizer.token_to_id('<PAD>')}\n"
        "lowercase=False\n"
        "add_prefix_space=True\n"
        "max_length=512\n"
    )

# -----------------------
# Sanity check
# -----------------------
sample = "Hello world! This is a test."
encoded = tokenizer.encode(sample)

print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)
print("Decoded:", tokenizer.decode(encoded.ids))

print(f"\n✅ Tokenizer trained and saved to {tokenizer_dir}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
