from tokenizers import ByteLevelBPETokenizer
import os

# --- Path to new tokenizer ---
tokenizer_dir = "../tokenizer"

# --- Load tokenizer ---
tokenizer = ByteLevelBPETokenizer(
    os.path.join(tokenizer_dir, "vocab.json"),
    os.path.join(tokenizer_dir, "merges.txt")
)

# --- Sample texts for evaluation ---
sample_texts = [
    "Once upon a time in a faraway land.",
    "The year 2024 brings many changes!",
    "“Hello,” she said, smiling.",
    "12345 is a number.",
    "In 1861, history changed drastically.",
    "He exclaimed: 'Wow! That’s amazing.'"
]

# --- Evaluation function ---
def evaluate_tokenizer(tokenizer, text):
    encoded = tokenizer.encode(text)
    print(f"Original: {text}")
    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)
    print("Decoded:", tokenizer.decode(encoded.ids))
    print("-" * 50)

# --- Run evaluation ---
print("=== NEW TOKENIZER EVALUATION ===")
for text in sample_texts:
    evaluate_tokenizer(tokenizer, text)
