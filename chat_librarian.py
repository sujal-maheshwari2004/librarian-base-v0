import torch
import torch.nn.functional as F
from model import GPT
from dataset import StoryDataset

# ============================================================
# Config (must match training)
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 8000
BLOCK_SIZE = 128

EMBED_DIM = 128
N_LAYER = 2
N_HEAD = 4
FF_DIM = 256

CHECKPOINT_PATH = "model/checkpoints/gpt_best.pt"
TOKENIZER_DIR = "tokenizer"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9

# ============================================================
# Load Model
# ============================================================

model = GPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    ff_dim=FF_DIM,
    context_length=BLOCK_SIZE
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded.")

# ============================================================
# Load tokenizer (dataset requires a file path)
# ============================================================

DUMMY_FILE = "data_clean/eval.txt"  # any valid text file

dataset = StoryDataset(TOKENIZER_DIR, DUMMY_FILE, BLOCK_SIZE)
tokenizer = dataset.tokenizer

print("✅ Tokenizer loaded.")

# ============================================================
# Sampling helpers
# ============================================================

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    batch_size, vocab_size = logits.shape

    # Top-K
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, -float("Inf")),
            logits
        )

    # Top-P (nucleus sampling)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1),
            dim=-1
        )

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = \
            sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for b in range(batch_size):
            indices = sorted_indices[b][sorted_indices_to_remove[b]]
            logits[b, indices] = -float("Inf")

    return logits


# ============================================================
# Generation Function
# ============================================================

@torch.no_grad()
def generate(prompt, max_new_tokens=100):
    encoding = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        encoding.ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    for _ in range(max_new_tokens):

        input_cond = input_ids[:, -BLOCK_SIZE:]

        logits = model(input_cond)
        logits = logits[:, -1, :]

        logits = logits / TEMPERATURE
        logits = top_k_top_p_filtering(logits, TOP_K, TOP_P)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())



# ============================================================
# Chat Loop
# ============================================================

print("\n💬 Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"User: {user_input}\nAssistant:"

    output = generate(prompt, MAX_NEW_TOKENS)

    print("\nAssistant:")
    print(output)
    print("\n" + "-" * 60)
