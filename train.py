import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GPT
from dataset import StoryDataset
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# Training config (8GB GPU friendly)
# ============================================================
VOCAB_SIZE = 8000
BLOCK_SIZE = 128

EMBED_DIM = 128
N_LAYER = 2
N_HEAD = 4
FF_DIM = 256
DROPOUT = 0.2

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1

LR = 1e-4
EPOCHS = 20

# ============================================================
# Early stopping
# ============================================================
PATIENCE = 5
MIN_DELTA = 1e-4

# ============================================================
# Paths
# ============================================================
print("➡ Tokenizer loading")
TOKENIZER_DIR = "tokenizer"
TRAIN_FILE = "data_clean/train.txt"
VAL_FILE = "data_clean/eval.txt"
CHECKPOINT_DIR = "model/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Dataset & loaders
# ============================================================
print("➡ Loading Dataset")
train_ds = StoryDataset(TOKENIZER_DIR, TRAIN_FILE, BLOCK_SIZE)
val_ds   = StoryDataset(TOKENIZER_DIR, VAL_FILE,   BLOCK_SIZE)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True
)

# ============================================================
# Model
# ============================================================
print("➡ Model Mounted")
model = GPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    ff_dim=FF_DIM,
    context_length=BLOCK_SIZE
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# ============================================================
# Training loop
# ============================================================
best_val_loss = float("inf")
epochs_without_improvement = 0

print("\n" + "=" * 60)
print("➡ Starting training")
print("=" * 60 + "\n")

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

    # --------------------
    # Train
    # --------------------
    model.train()
    optimizer.zero_grad()
    total_train_loss = 0.0

    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for step, (x, y) in enumerate(train_bar):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1)
            )
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_train_loss += loss.item() * GRAD_ACCUM_STEPS
        train_bar.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS)

    avg_train_loss = total_train_loss / len(train_loader)

    # --------------------
    # Validation
    # --------------------
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        for x, y in val_bar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(
                    logits.view(-1, VOCAB_SIZE),
                    y.view(-1)
                )

            total_val_loss += loss.item()
            val_bar.set_postfix(loss=loss.item())

    avg_val_loss = total_val_loss / len(val_loader)

    # --------------------
    # Logging
    # --------------------
    print(
        f"Epoch {epoch + 1} summary | "
        f"train loss: {avg_train_loss:.4f} | "
        f"val loss: {avg_val_loss:.4f}"
    )

    # --------------------
    # Save epoch checkpoint
    # --------------------
    epoch_ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        f"gpt_epoch_{epoch + 1}.pt"
    )
    torch.save(model.state_dict(), epoch_ckpt_path)
    print(f"💾 Saved epoch checkpoint → {epoch_ckpt_path}")

    # --------------------
    # Best model + early stopping
    # --------------------
    if avg_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

        best_ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            "gpt_best.pt"
        )
        torch.save(model.state_dict(), best_ckpt_path)
        print("🏆 New best model saved")

    else:
        epochs_without_improvement += 1
        print(
            f"⚠ No validation improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    if epochs_without_improvement >= PATIENCE:
        print(
            "\n🛑 Early stopping triggered "
            f"(no improvement for {PATIENCE} epochs)"
        )
        break

print("\n✅ Training complete.")
