"""
Q5 – Language Modeling
WikiText-2: Trigram LM (Kneser-Ney smoothing) vs LSTM LM (2-layer, weight tying)
"""

import os
import math
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from utils import set_seed, SEED, device
set_seed(SEED)
os.makedirs("outputs/q5", exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_FREQ   = 2          # vocab threshold
BPTT       = 35         # backprop-through-time sequence length
BATCH_SIZE = 32
EMBED_DIM  = 512
HIDDEN_SIZE= 512
N_LAYERS   = 2
DROPOUT    = 0.5
N_EPOCHS   = 20
LR         = 1e-3       # AdamW learning rate
CLIP       = 1.0        # standard gradient clipping for LM
LR_FACTOR  = 2.0        # kept for reference, scheduler handles decay
N_SAMPLES  = 5          # generated text samples per model
SAMPLE_LEN = 50         # words per sample

# ---------------------------------------------------------------------------
# 5.1 — Load WikiText-2
# ---------------------------------------------------------------------------
print("=" * 60)
print("5.1  Loading WikiText-2 dataset…")
raw = load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"  train lines: {len(raw['train'])}")
print(f"  val   lines: {len(raw['validation'])}")
print(f"  test  lines: {len(raw['test'])}")

# ---------------------------------------------------------------------------
# 5.2 — Tokenization: word-level, <unk>, <eos>
# ---------------------------------------------------------------------------
print("\n5.2  Tokenization (word-level)…")

UNK, EOS, PAD = "<unk>", "<eos>", "<pad>"

def tokenize(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    return text.lower().split() + [EOS]


def corpus_tokens(split_data) -> list:
    tokens = []
    for row in split_data:
        toks = tokenize(row["text"])
        tokens.extend(toks)
    return tokens


train_tokens = corpus_tokens(raw["train"])
val_tokens   = corpus_tokens(raw["validation"])
test_tokens  = corpus_tokens(raw["test"])

print(f"  Train tokens: {len(train_tokens):,}")
print(f"  Val   tokens: {len(val_tokens):,}")
print(f"  Test  tokens: {len(test_tokens):,}")

# ---------------------------------------------------------------------------
# 5.3 — Vocabulary (min_freq=2)
# ---------------------------------------------------------------------------
print(f"\n5.3  Building vocabulary (min_freq={MIN_FREQ})…")

freq = Counter(train_tokens)
vocab_words = [w for w, c in freq.items() if c >= MIN_FREQ]
vocab = [PAD, UNK, EOS] + sorted(vocab_words)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

PAD_IDX = word2idx[PAD]
UNK_IDX = word2idx[UNK]
EOS_IDX = word2idx[EOS]

print(f"  Vocab size: {len(vocab):,} (raw unique: {len(freq):,})")


def encode(tokens: list) -> list:
    return [word2idx.get(t, UNK_IDX) for t in tokens]


train_ids = encode(train_tokens)
val_ids   = encode(val_tokens)
test_ids  = encode(test_tokens)

# Save vocab stats
pd.DataFrame({
    "split": ["train", "val", "test"],
    "tokens": [len(train_tokens), len(val_tokens), len(test_tokens)],
}).to_csv("outputs/q5/q5_vocab_stats.csv", index=False)

# ---------------------------------------------------------------------------
# 5.4 — Model 1: Trigram Language Model (Kneser-Ney smoothing)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5.4  Trigram Language Model (Kneser-Ney smoothing)…")


class TrigramLM:
    """Interpolated Kneser-Ney trigram language model.

    P_KN(w3|w1,w2) = max(c(w1,w2,w3) - d, 0) / c(w1,w2)
                   + lambda(w1,w2) * P_KN(w3|w2)

    P_KN(w3|w2)    = max(c_cont(w2,w3) - d, 0) / c_cont(w2)
                   + lambda(w2) * P_uni(w3)

    P_uni(w3)      = |{w2 : c(w2,w3)>0}| / total_continuation_count
    """

    DISCOUNT = 0.75   # standard fixed discount for KN

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        # trigram counts: (w1,w2) → Counter{w3: count}
        self.tri_counts  = defaultdict(Counter)
        # bigram counts: w2 → Counter{w3: count}
        self.bi_counts   = defaultdict(Counter)
        # continuation counts for unigram: w3 → |{w2 : c(w2,w3)>0}|
        self.uni_cont    = Counter()
        # number of unique (w2,w3) pairs — denominator for P_uni
        self.total_bi_types = 0

    def train(self, ids: list):
        d = self.DISCOUNT
        for i in range(len(ids) - 2):
            w1, w2, w3 = ids[i], ids[i + 1], ids[i + 2]
            self.tri_counts[(w1, w2)][w3] += 1
            self.bi_counts[w2][w3]        += 1

        # Continuation unigram: how many unique left-contexts does w3 appear in
        for w2, counter in self.bi_counts.items():
            for w3, cnt in counter.items():
                self.uni_cont[w3] += 1   # each unique (w2,w3) pair
        self.total_bi_types = sum(self.uni_cont.values())

        print(f"  Trigram contexts: {len(self.tri_counts):,}")
        print(f"  Bigram  contexts: {len(self.bi_counts):,}")
        print(f"  Discount d={d}")

    def _p_uni(self, w3: int) -> float:
        """Kneser-Ney unigram: continuation probability."""
        return self.uni_cont.get(w3, 0) / max(self.total_bi_types, 1)

    def _p_bi(self, w2: int, w3: int) -> float:
        """Kneser-Ney bigram with backoff to KN unigram."""
        d = self.DISCOUNT
        counter  = self.bi_counts.get(w2)
        if not counter:
            return self._p_uni(w3)
        c_w2w3   = counter.get(w3, 0)
        c_w2     = sum(counter.values())
        n_w2     = len(counter)           # |{w3 : c(w2,w3)>0}|
        lam_w2   = (d * n_w2) / max(c_w2, 1)
        norm     = max(c_w2w3 - d, 0) / c_w2
        return norm + lam_w2 * self._p_uni(w3)

    def log_prob(self, w1: int, w2: int, w3: int) -> float:
        d = self.DISCOUNT
        counter  = self.tri_counts.get((w1, w2))
        if not counter:
            return math.log(max(self._p_bi(w2, w3), 1e-10))
        c_tri    = counter.get(w3, 0)
        c_bi     = sum(counter.values())
        n_bi     = len(counter)           # |{w3 : c(w1,w2,w3)>0}|
        lam_bi   = (d * n_bi) / max(c_bi, 1)
        norm     = max(c_tri - d, 0) / c_bi
        prob     = norm + lam_bi * self._p_bi(w2, w3)
        return math.log(max(prob, 1e-10))

    def perplexity(self, ids: list) -> float:
        log_prob = 0.0
        n = 0
        for i in range(len(ids) - 2):
            log_prob += self.log_prob(ids[i], ids[i + 1], ids[i + 2])
            n += 1
        return math.exp(-log_prob / n)

    def generate(self, seed_ids: list, length: int = SAMPLE_LEN,
                 temperature: float = 1.0) -> list:
        ids = list(seed_ids[-2:])
        for _ in range(length):
            w1, w2 = ids[-2], ids[-1]
            # Sample from trigram distribution (fall back to bigram for unseen ctx)
            counter = self.tri_counts.get((w1, w2)) or self.bi_counts.get(w2)
            if counter:
                words = list(counter.keys())
                probs = np.array([counter[w] for w in words], dtype=float)
                if temperature != 1.0:
                    probs = np.power(probs, 1.0 / temperature)
                probs /= probs.sum()
                next_w = np.random.choice(words, p=probs)
            else:
                next_w = random.randint(3, self.vocab_size - 1)
            ids.append(next_w)
            if next_w == EOS_IDX:
                break
        return ids[2:]


trigram = TrigramLM(len(vocab))
trigram.train(train_ids)

trigram_val_ppl  = trigram.perplexity(val_ids)
trigram_test_ppl = trigram.perplexity(test_ids)
print(f"  Val  Perplexity: {trigram_val_ppl:.2f}")
print(f"  Test Perplexity: {trigram_test_ppl:.2f}")

# Generate 5 samples
print(f"\n  Trigram samples (temperature=0.8):")
trigram_samples = []
for i in range(N_SAMPLES):
    seed = random.sample(range(3, len(vocab)), 2)
    gen  = trigram.generate(seed, length=SAMPLE_LEN, temperature=0.8)
    text = " ".join(idx2word.get(t, UNK) for t in gen if t != EOS_IDX)
    trigram_samples.append(text)
    print(f"  [{i+1}] {text[:120]}")

# ---------------------------------------------------------------------------
# 5.5 — Model 2: LSTM Language Model (2-layer, weight tying)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5.5  LSTM Language Model (2-layer, weight tying)…")


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 n_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop      = nn.Dropout(dropout)
        self.rnn       = nn.LSTM(embed_dim, hidden_size, n_layers,
                                  batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc        = nn.Linear(hidden_size, vocab_size)
        # Weight tying: embedding matrix = output projection matrix
        if embed_dim == hidden_size:
            self.fc.weight = self.embedding.weight
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

    def forward(self, x, hidden=None):
        emb = self.drop(self.embedding(x))
        out, hidden = self.rnn(emb, hidden)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int):
        w = next(self.parameters())
        return (w.new_zeros(self.n_layers, batch_size, self.hidden_size),
                w.new_zeros(self.n_layers, batch_size, self.hidden_size))

    @torch.no_grad()
    def generate(self, seed_ids: list, length: int = SAMPLE_LEN,
                 temperature: float = 0.8) -> list:
        self.eval()
        ids = list(seed_ids)
        hidden = self.init_hidden(1)
        # Feed seed
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        _, hidden = self(src, hidden)
        cur = torch.tensor([ids[-1]], dtype=torch.long, device=device).unsqueeze(0)
        result = list(ids)
        for _ in range(length):
            logits, hidden = self(cur, hidden)
            logits = logits[:, -1, :] / temperature
            probs  = torch.softmax(logits, dim=-1).squeeze()
            next_w = torch.multinomial(probs, 1).item()
            result.append(next_w)
            cur = torch.tensor([[next_w]], dtype=torch.long, device=device)
            if next_w == EOS_IDX:
                break
        return result[len(seed_ids):]


# BPTT Dataset
class LMDataset(Dataset):
    def __init__(self, ids: list, bptt: int):
        self.ids  = torch.tensor(ids, dtype=torch.long)
        self.bptt = bptt
        self.n    = (len(ids) - 1) // bptt

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.bptt
        x = self.ids[start : start + self.bptt]
        y = self.ids[start + 1 : start + self.bptt + 1]
        return x, y


train_lm_ds = LMDataset(train_ids, BPTT)
val_lm_ds   = LMDataset(val_ids,   BPTT)

train_lm_loader = DataLoader(train_lm_ds, batch_size=BATCH_SIZE, shuffle=True)
val_lm_loader   = DataLoader(val_lm_ds,   batch_size=BATCH_SIZE, shuffle=False)

lstm_model = LSTMLanguageModel(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    pad_idx=PAD_IDX,
).to(device)

n_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"  LSTM LM parameters: {n_params:,}")
print(f"  embed={EMBED_DIM}, hidden={HIDDEN_SIZE}, layers={N_LAYERS}, dropout={DROPOUT}")
print(f"  Weight tying: {'YES' if EMBED_DIM == HIDDEN_SIZE else 'NO (embed≠hidden)'}")
print(f"  Training on {device}, epochs={N_EPOCHS}, bptt={BPTT}, batch={BATCH_SIZE}")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

history = []
best_val_ppl = float("inf")
best_path    = "outputs/q5/lstm_best.pt"
lr           = LR


def evaluate_lm(model, loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(-1, V), y.reshape(-1))
            non_pad = (y != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 20))


print(f"\n  Epoch logs:")
for epoch in range(1, N_EPOCHS + 1):
    lstm_model.train()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
    for x, y in tqdm(train_lm_loader, desc=f"  Epoch {epoch:2d}/{N_EPOCHS}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = lstm_model(x)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(lstm_model.parameters(), CLIP)
        optimizer.step()
        non_pad = (y != PAD_IDX).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    train_ppl = math.exp(min(total_loss / total_tokens, 20))
    val_ppl   = evaluate_lm(lstm_model, val_lm_loader)
    elapsed   = time.time() - t0

    scheduler.step(val_ppl)
    lr = optimizer.param_groups[0]["lr"]
    history.append((epoch, train_ppl, val_ppl, lr))

    marker = ""
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save(lstm_model.state_dict(), best_path)
        marker = " ✓"

    print(f"  Epoch {epoch:2d}/{N_EPOCHS} | train_ppl={train_ppl:.2f}"
          f" | val_ppl={val_ppl:.2f} | lr={lr:.6f} | {elapsed:.0f}s{marker}")

# Save training history
hist_df = pd.DataFrame(history, columns=["epoch", "train_ppl", "val_ppl", "lr"])
hist_df.to_csv("outputs/q5/q5_lstm_training.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist_df["epoch"], hist_df["train_ppl"], label="train")
axes[0].plot(hist_df["epoch"], hist_df["val_ppl"],   label="val")
axes[0].set_title("LSTM LM Perplexity"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Perplexity"); axes[0].legend()
axes[1].plot(hist_df["epoch"], hist_df["lr"])
axes[1].set_title("Learning Rate Schedule"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("LR")
plt.tight_layout()
plt.savefig("outputs/q5/q5_lstm_curve.png", dpi=150)
plt.close()

# Load best checkpoint & compute test perplexity
lstm_model.load_state_dict(torch.load(best_path, map_location=device))
test_lm_ds     = LMDataset(test_ids, BPTT)
test_lm_loader = DataLoader(test_lm_ds, batch_size=BATCH_SIZE, shuffle=False)
lstm_val_ppl   = best_val_ppl
lstm_test_ppl  = evaluate_lm(lstm_model, test_lm_loader)
print(f"\n  Best val_ppl = {lstm_val_ppl:.2f}")
print(f"  Test ppl     = {lstm_test_ppl:.2f}")

# Generate 5 samples
print(f"\n  LSTM samples (temperature=0.8):")
lstm_samples = []
for i in range(N_SAMPLES):
    seed = random.sample(range(3, len(vocab)), 2)
    gen  = lstm_model.generate(seed, length=SAMPLE_LEN, temperature=0.8)
    text = " ".join(idx2word.get(t, UNK) for t in gen if t != EOS_IDX)
    lstm_samples.append(text)
    print(f"  [{i+1}] {text[:120]}")

# ---------------------------------------------------------------------------
# 5.6 — Results table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5.6  Results summary…")

results = pd.DataFrame([
    {"model": "Trigram (Kneser-Ney)", "val_ppl": round(trigram_val_ppl, 2),
     "test_ppl": round(trigram_test_ppl, 2)},
    {"model": "LSTM LM (2-layer)", "val_ppl": round(lstm_val_ppl, 2),
     "test_ppl": round(lstm_test_ppl, 2)},
])
print(results.to_string(index=False))
results.to_csv("outputs/q5/q5_results.csv", index=False)
print("  Results → outputs/q5/q5_results.csv")

# Bar chart
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
vals_val  = [trigram_val_ppl,  lstm_val_ppl]
vals_test = [trigram_test_ppl, lstm_test_ppl]
ax.bar(x - 0.2, vals_val,  0.4, label="Val PPL")
ax.bar(x + 0.2, vals_test, 0.4, label="Test PPL")
ax.set_xticks(x)
ax.set_xticklabels(["Trigram (Kneser-Ney)", "LSTM LM (2-layer)"])
ax.set_ylabel("Perplexity (lower = better)")
ax.set_title("Q5 Language Model Perplexity Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/q5/q5_perplexity_comparison.png", dpi=150)
plt.close()
print("  Chart → outputs/q5/q5_perplexity_comparison.png")

# ---------------------------------------------------------------------------
# 5.7-5.8 — Qualitative samples to CSV
# ---------------------------------------------------------------------------
print("\n5.7  Saving generated text samples…")
sample_rows = []
for i, (tg, ls) in enumerate(zip(trigram_samples, lstm_samples)):
    sample_rows.append({"sample_id": i + 1, "trigram": tg, "lstm": ls})
samples_df = pd.DataFrame(sample_rows)
samples_df.to_csv("outputs/q5/q5_samples.csv", index=False)
print("  Samples → outputs/q5/q5_samples.csv")

# ---------------------------------------------------------------------------
# TTR (Type-Token Ratio) analysis
# ---------------------------------------------------------------------------
print("\n5.8  TTR (Type-Token Ratio) analysis…")

def compute_ttr(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

ttr_rows = []
for i, (tg, ls) in enumerate(zip(trigram_samples, lstm_samples)):
    ttr_rows.append({
        "sample_id":   i + 1,
        "trigram_ttr": round(compute_ttr(tg), 4),
        "lstm_ttr":    round(compute_ttr(ls), 4),
        "trigram_len": len(tg.split()),
        "lstm_len":    len(ls.split()),
    })

ttr_df = pd.DataFrame(ttr_rows)
ttr_df.loc["avg"] = ttr_df.mean(numeric_only=True)
ttr_df.to_csv("outputs/q5/q5_ttr_analysis.csv")
print(ttr_df.to_string())
print("  TTR → outputs/q5/q5_ttr_analysis.csv")

avg_ttr_trigram = ttr_df.loc["avg", "trigram_ttr"]
avg_ttr_lstm    = ttr_df.loc["avg", "lstm_ttr"]
print(f"\n  Avg TTR — Trigram: {avg_ttr_trigram:.4f} | LSTM: {avg_ttr_lstm:.4f}")
print("  Higher TTR → more diverse vocabulary (not necessarily more coherent)")

print("\n5.9  Fluency & coherence notes:")
print("  Trigram: local bigram context only, long-range coherence poor")
print("  LSTM:    hidden state carries long-range dependencies, more fluent")

print("\nPhase 5 complete. Outputs in outputs/q5/")
