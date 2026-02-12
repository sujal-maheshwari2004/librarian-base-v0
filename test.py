import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from model import GPT
from dataset import StoryDataset

# ============================================================
# Config
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 8000
BLOCK_SIZE = 128

EMBED_DIM = 128
N_LAYER = 2
N_HEAD = 4
FF_DIM = 256

BATCH_SIZE = 32

TOKENIZER_DIR = "tokenizer"
EVAL_FILE = "data_clean/eval.txt"
CHECKPOINT_PATH = "model/checkpoints/gpt_best.pt"


def main():

    # ============================================================
    # Load dataset
    # ============================================================

    eval_ds = StoryDataset(TOKENIZER_DIR, EVAL_FILE, BLOCK_SIZE)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,   # works now
        pin_memory=True
    )

    # ============================================================
    # Load model
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

    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # Evaluation
    # ============================================================

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0

    with torch.no_grad():
        for x, y in tqdm(eval_loader, desc="Evaluating"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1)
            )

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)

            top1 = torch.argmax(probs, dim=-1)
            top5 = torch.topk(probs, 5, dim=-1).indices
            top10 = torch.topk(probs, 10, dim=-1).indices

            correct_top1 += (top1 == y).sum().item()

            correct_top5 += (
                (top5 == y.unsqueeze(-1)).any(dim=-1)
            ).sum().item()

            correct_top10 += (
                (top10 == y.unsqueeze(-1)).any(dim=-1)
            ).sum().item()

            total_tokens += y.numel()

    # ============================================================
    # Final Metrics
    # ============================================================

    avg_loss = total_loss / len(eval_loader)
    perplexity = math.exp(avg_loss)

    top1_acc = correct_top1 / total_tokens
    top5_acc = correct_top5 / total_tokens
    top10_acc = correct_top10 / total_tokens

    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    print(f"Cross-Entropy Loss : {avg_loss:.4f}")
    print(f"Perplexity         : {perplexity:.4f}")
    print(f"Top-1 Accuracy     : {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy     : {top5_acc * 100:.2f}%")
    print(f"Top-10 Accuracy    : {top10_acc * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
