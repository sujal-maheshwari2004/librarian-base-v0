---
license: mit
language: en
---

# My Custom GPT Model (approx. 43M parameters)

This repository contains a custom-trained GPT-style language model with approximately 43 million parameters. It was built from scratch using PyTorch and converted to the Hugging Face `transformers` format for easy use and sharing.

## Model Details

*   **Architecture:** GPT-2 Style Transformer
*   **Parameters:** ~43 Million
*   **Context Length:** 256 tokens
*   **Vocabulary Size:** 30,000
*   **Tokenizer:** Custom Byte-Level BPE tokenizer trained alongside the model.

## Training Data

The model was trained on a corpus of classic English literature obtained from Project Gutenberg. The training data consists of the following books:

*   *Pride and Prejudice* by Jane Austen (Book ID: 1342)
*   *Alice's Adventures in Wonderland* by Lewis Carroll (Book ID: 11)
*   *The Adventures of Sherlock Holmes* by Arthur Conan Doyle (Book ID: 1661)
*   *A Tale of Two Cities* by Charles Dickens (Book ID: 84)
*   *Moby Dick; or The Whale* by Herman Melville (Book ID: 1952)

The text was cleaned, normalized, and combined into a single dataset for training.

## How to Use

You can easily use this model for text generation using the `transformers` library pipeline.

First, make sure you have `transformers` and `torch` installed:
```bash
pip install transformers torch
```

Then, you can use the following Python code to load the model and generate text. **Remember to replace `"YourUsername/YourModelName"` with your actual model repository ID on the Hugging Face Hub.**

```python
from transformers import pipeline

# Replace with your model's repository ID on the Hugging Face Hub
repo_id = "YourUsername/YourModelName"

# Load the text generation pipeline
try:
    generator = pipeline('text-generation', model=repo_id)
    print(f"Successfully loaded model from {repo_id}")
except Exception as e:
    print(f"Failed to load model. Is the repo_id '{repo_id}' correct and public?")
    print(e)
    exit()

# The prompt you want the model to continue
prompt = "Once upon a time in a land far, far away"

# Generate text
# You can adjust max_length, temperature, and top_k for different results
generated_text = generator(
    prompt,
    max_length=150,          # Max total length of the output
    num_return_sequences=1,
    temperature=0.8,         # Controls randomness: lower is less random
    top_k=50,                # Samples from the top K most likely tokens
    pad_token_id=generator.tokenizer.eos_token_id # Suppress padding warning
)

print("--- Generated Text ---")
print(generated_text[0]['generated_text'])
```

### Example Output

```
--- Generated Text ---a
Once upon a time in a land far, far away, there was a little girl who lived in a great forest. She was a very good little girl, and she was very fond of her mother.
```

t.*
