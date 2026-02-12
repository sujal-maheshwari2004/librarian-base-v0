from datasets import load_dataset
import os
import random
import time
import re
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
TRAIN_FRACTION = 0.02      # use 2% (safe for 4M param model)
TEST_RATIO = 0.05          # 5%
EVAL_RATIO = 0.05          # 5%
RANDOM_SEED = 42
MIN_DOC_LEN = 200          # minimum characters
MAX_DOC_LEN = 5000         # truncate very long books (optional)

random.seed(RANDOM_SEED)

# -------------------------
# Logging helpers
# -------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def timed(stage):
    log(f"▶ {stage} started")
    return time.time()

def done(stage, t0):
    log(f"✔ {stage} finished in {time.time() - t0:.1f}s")

# -------------------------
# Paths
# -------------------------
data_dir = "../data_clean"
os.makedirs(data_dir, exist_ok=True)

train_file = os.path.join(data_dir, "train.txt")
eval_file  = os.path.join(data_dir, "eval.txt")
test_file  = os.path.join(data_dir, "test.txt")

# -------------------------
# Basic Cleaner
# -------------------------
MULTI_SPACE_RE = re.compile(r"\s{2,}")

def clean_text(text):
    text = text.strip()
    if len(text) < MIN_DOC_LEN:
        return ""

    text = MULTI_SPACE_RE.sub(" ", text)

    # Optional truncation to avoid extremely long docs
    if len(text) > MAX_DOC_LEN:
        text = text[:MAX_DOC_LEN]

    return text.strip()

# -------------------------
# Load BookCorpus
# -------------------------
t0 = timed("Loading TinyStories dataset")
dataset = load_dataset("roneneldan/TinyStories")
done("Loading TinyStories dataset", t0)

texts = dataset["train"]["text"]
total_docs = len(texts)

# -------------------------
# Subsample
# -------------------------
t1 = timed("Subsampling dataset")

num_keep = int(total_docs * TRAIN_FRACTION)
indices = random.sample(range(total_docs), num_keep)

log(f"✂ Keeping {num_keep:,} / {total_docs:,} documents ({TRAIN_FRACTION*100:.2f}%)")
done("Subsampling dataset", t1)

# -------------------------
# Clean Documents
# -------------------------
t2 = timed("Cleaning documents")

cleaned_docs = []

for i in tqdm(indices, desc="Cleaning"):
    cleaned = clean_text(texts[i])
    if cleaned:
        cleaned_docs.append(cleaned)

log(f"🧹 Kept {len(cleaned_docs):,} clean documents")
done("Cleaning documents", t2)

# -------------------------
# Shuffle and Split
# -------------------------
t3 = timed("Splitting dataset")

random.shuffle(cleaned_docs)

total = len(cleaned_docs)
test_size = int(total * TEST_RATIO)
eval_size = int(total * EVAL_RATIO)
train_size = total - test_size - eval_size

train_split = cleaned_docs[:train_size]
eval_split  = cleaned_docs[train_size:train_size + eval_size]
test_split  = cleaned_docs[train_size + eval_size:]

log(f"📊 Train: {len(train_split):,}")
log(f"📊 Eval : {len(eval_split):,}")
log(f"📊 Test : {len(test_split):,}")

done("Splitting dataset", t3)

# -------------------------
# Write Files
# -------------------------
def write_split(name, data, path):
    t = timed(f"Writing {name}")

    with open(path, "w", encoding="utf-8") as f:
        for doc in tqdm(data, desc=f"Writing {name}"):
            f.write(doc + "\n")

    done(f"Writing {name}", t)

write_split("train", train_split, train_file)
write_split("eval", eval_split, eval_file)
write_split("test", test_split, test_file)

# -------------------------
# Stats
# -------------------------
def file_stats(path):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    words = 0
    lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words += len(line.split())
            lines += 1
    return size_mb, words, lines

print("\n📊 Dataset summary:\n")

for name, path in [
    ("Training", train_file),
    ("Evaluation", eval_file),
    ("Testing", test_file),
]:
    size_mb, words, lines = file_stats(path)
    print(f"{name:<11}: {size_mb:8.2f} MB | {words:,} words | {lines:,} documents")

log("🚀 BookCorpus dataset ready")
