"""
Q1 – Text Classification (Representation Learning)
IMDb sentiment classification: TF-IDF+LR, TF-IDF+SVM, BiLSTM, DistilBERT-base-uncased
"""

import os
import re
import csv
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from utils import device, set_seed, SEED

set_seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_SIZE = 45000
VAL_SIZE   = 2500
TEST_SIZE  = 2500
BATCH_SIZE_BERT = 8
BATCH_SIZE_LSTM = 64
LSTM_EPOCHS = 10
BERT_EPOCHS = 3
GLOVE_DIM   = 100
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
LR_LSTM     = 1e-3
LR_BERT     = 3e-5
MAX_LEN_BERT = 192
os.makedirs("outputs/q1", exist_ok=True)

# ---------------------------------------------------------------------------
# 1.1 — Load IMDb
# ---------------------------------------------------------------------------
print("=" * 60)
print("1.1  Loading IMDb dataset...")
raw = load_dataset("imdb")

# Combine train (25K) + unsupervised(0 labels, skip) → use only labelled
all_train = raw["train"].shuffle(seed=SEED)   # 25 000 labelled
all_test  = raw["test"]                        # 25 000 labelled

# ---------------------------------------------------------------------------
# 1.2 — Train / Val / Test split
# We take 45K train from the full labelled pool (train+test), 2.5K val, 2.5K test
# ---------------------------------------------------------------------------
print("1.2  Creating train/val/test splits…")
from datasets import concatenate_datasets
full = concatenate_datasets([raw["train"], raw["test"]]).shuffle(seed=SEED)

train_ds = full.select(range(TRAIN_SIZE))
val_ds   = full.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))
test_ds  = full.select(range(TRAIN_SIZE + VAL_SIZE, TRAIN_SIZE + VAL_SIZE + TEST_SIZE))

print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# ---------------------------------------------------------------------------
# 1.3 — Preprocessing pipeline
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)          # punctuation / special chars
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in STOP_WORDS)
    return text

print("1.3  Preprocessing texts…")
train_texts_clean = [clean_text(t) for t in tqdm(train_ds["text"], desc="  train")]
val_texts_clean   = [clean_text(t) for t in tqdm(val_ds["text"],   desc="  val")]
test_texts_clean  = [clean_text(t) for t in tqdm(test_ds["text"],  desc="  test")]

train_labels = list(train_ds["label"])
val_labels   = list(val_ds["label"])
test_labels  = list(test_ds["label"])

# ---------------------------------------------------------------------------
# 1.4 — Tokenization strategy comparison
# Strategy A: word-level split  |  Strategy B: BERT WordPiece
# ---------------------------------------------------------------------------
print("1.4  Tokenization strategy comparison…")
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

sample = train_texts_clean[0][:200]
tokens_a = sample.split()
tokens_b = bert_tokenizer.tokenize(sample)

tok_table = pd.DataFrame({
    "Strategy": ["Word-level (A)", "WordPiece BERT (B)"],
    "Example tokens (first 15)": [
        str(tokens_a[:15]),
        str(tokens_b[:15]),
    ],
    "Token count (sample)": [len(tokens_a), len(tokens_b)],
})
print(tok_table.to_string(index=False))
tok_table.to_csv("outputs/q1/q1_tokenization_comparison.csv", index=False)
print("  Saved: outputs/q1_tokenization_comparison.csv")

# ---------------------------------------------------------------------------
# 1.5 — Preprocessing kararlarının etkisini tablo halinde dokümante et
# ---------------------------------------------------------------------------
print("1.5  Preprocessing impact statistics…")

# Raw (uncleaned) texts
train_texts_raw = list(train_ds["text"])

def text_stats(texts_raw, texts_clean, label):
    raw_lens   = [len(t.split()) for t in texts_raw]
    clean_lens = [len(t.split()) for t in texts_clean]
    html_count = sum(1 for t in texts_raw if re.search(r"<[^>]+>", t))
    return {
        "Split":                     label,
        "Avg tokens (raw)":          round(np.mean(raw_lens), 1),
        "Avg tokens (clean)":        round(np.mean(clean_lens), 1),
        "Token reduction %":         round((1 - np.mean(clean_lens) / np.mean(raw_lens)) * 100, 1),
        "Samples with HTML tags":    html_count,
        "Vocab size (raw)":          len(set(w for t in texts_raw   for w in t.lower().split())),
        "Vocab size (clean)":        len(set(w for t in texts_clean for w in t.split())),
    }

prep_stats = pd.DataFrame([
    text_stats(list(train_ds["text"]), train_texts_clean, "Train"),
    text_stats(list(val_ds["text"]),   val_texts_clean,   "Val"),
    text_stats(list(test_ds["text"]),  test_texts_clean,  "Test"),
])
print(prep_stats.to_string(index=False))
prep_stats.to_csv("outputs/q1/q1_preprocessing_stats.csv", index=False)
print("  Saved: outputs/q1_preprocessing_stats.csv")

# ---------------------------------------------------------------------------
# 1.6 — Model 1: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("1.6  TF-IDF + Logistic Regression…")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_texts_clean)
X_val_tfidf   = tfidf.transform(val_texts_clean)
X_test_tfidf  = tfidf.transform(test_texts_clean)

lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
lr_model.fit(X_train_tfidf, train_labels)

lr_val_preds  = lr_model.predict(X_val_tfidf)
lr_test_preds = lr_model.predict(X_test_tfidf)

lr_val_acc  = accuracy_score(val_labels,  lr_val_preds)
lr_val_f1   = f1_score(val_labels,  lr_val_preds, average="macro")
lr_test_acc = accuracy_score(test_labels, lr_test_preds)
lr_test_f1  = f1_score(test_labels, lr_test_preds, average="macro")
print(f"  Val  — Acc: {lr_val_acc:.4f} | Macro-F1: {lr_val_f1:.4f}")
print(f"  Test — Acc: {lr_test_acc:.4f} | Macro-F1: {lr_test_f1:.4f}")

# ---------------------------------------------------------------------------
# 1.7 — Model 2: TF-IDF + SVM
# ---------------------------------------------------------------------------
print("\n1.7  TF-IDF + SVM…")
svm_model = LinearSVC(C=1.0, random_state=SEED, max_iter=2000)
svm_model.fit(X_train_tfidf, train_labels)

svm_val_preds  = svm_model.predict(X_val_tfidf)
svm_test_preds = svm_model.predict(X_test_tfidf)

svm_val_acc  = accuracy_score(val_labels,  svm_val_preds)
svm_val_f1   = f1_score(val_labels,  svm_val_preds, average="macro")
svm_test_acc = accuracy_score(test_labels, svm_test_preds)
svm_test_f1  = f1_score(test_labels, svm_test_preds, average="macro")
print(f"  Val  — Acc: {svm_val_acc:.4f} | Macro-F1: {svm_val_f1:.4f}")
print(f"  Test — Acc: {svm_test_acc:.4f} | Macro-F1: {svm_test_f1:.4f}")

# ---------------------------------------------------------------------------
# 1.8 — Model 3: BiLSTM + GloVe
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("1.8  BiLSTM + GloVe embeddings…")

# --- Vocabulary ---
from collections import Counter

def build_vocab(texts, max_vocab=30000):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(train_texts_clean)
print(f"  Vocab size: {len(vocab)}")

# --- Load GloVe ---
GLOVE_PATH = os.path.expanduser("~/.cache/glove/glove.6B.100d.txt")

def load_glove(path, vocab, dim=100):
    embed = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    embed[0] = 0.0  # <pad>
    if not os.path.exists(path):
        print(f"  GloVe not found at {path}, using random init.")
        return embed
    found = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                embed[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"  GloVe: {found}/{len(vocab)} words matched.")
    return embed

glove_matrix = load_glove(GLOVE_PATH, vocab, dim=GLOVE_DIM)

# --- Dataset ---
MAX_SEQ_LEN = 256

def encode(texts, vocab, max_len=MAX_SEQ_LEN):
    out = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in t.split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        out.append(ids)
    return np.array(out, dtype=np.int64)

X_train_lstm = encode(train_texts_clean, vocab)
X_val_lstm   = encode(val_texts_clean,   vocab)
X_test_lstm  = encode(test_texts_clean,  vocab)

class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader_lstm = DataLoader(SentimentDataset(X_train_lstm, train_labels),
                               batch_size=BATCH_SIZE_LSTM, shuffle=True)
val_loader_lstm   = DataLoader(SentimentDataset(X_val_lstm,   val_labels),
                               batch_size=BATCH_SIZE_LSTM)
test_loader_lstm  = DataLoader(SentimentDataset(X_test_lstm,  test_labels),
                               batch_size=BATCH_SIZE_LSTM)

# --- Model ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
                 num_classes=2, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.drop(self.embedding(x))
        out, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)   # last fwd + bwd hidden
        return self.fc(self.drop(h))

lstm_model = BiLSTMClassifier(
    vocab_size=len(vocab), embed_dim=GLOVE_DIM,
    hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
    pretrained=glove_matrix
).to(device)

optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=LR_LSTM)
criterion = nn.CrossEntropyLoss()

def evaluate_loader(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(yb.cpu().tolist())
    return accuracy_score(trues, preds), f1_score(trues, preds, average="macro"), preds, trues

lstm_train_losses, lstm_val_accs = [], []
LSTM_CKPT      = "outputs/q1/bilstm_best.pt"
EARLY_STOP     = 3          # stop if no improvement for 3 consecutive epochs
best_val_acc_lstm = 0.0
no_improve     = 0

print("  Training BiLSTM…")
for epoch in range(1, LSTM_EPOCHS + 1):
    lstm_model.train()
    total_loss = 0.0
    for xb, yb in tqdm(train_loader_lstm, desc=f"  Epoch {epoch}/{LSTM_EPOCHS}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer_lstm.zero_grad()
        loss = criterion(lstm_model(xb), yb)
        loss.backward()
        optimizer_lstm.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader_lstm)
    val_acc, val_f1, _, _ = evaluate_loader(lstm_model, val_loader_lstm)
    lstm_train_losses.append(avg_loss)
    lstm_val_accs.append(val_acc)

    marker = ""
    if val_acc > best_val_acc_lstm:
        best_val_acc_lstm = val_acc
        torch.save(lstm_model.state_dict(), LSTM_CKPT)
        no_improve = 0
        marker = " ✓"
    else:
        no_improve += 1

    print(f"  Epoch {epoch:02d} — loss: {avg_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}{marker}")

    if no_improve >= EARLY_STOP:
        print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)")
        break

# Load best checkpoint for evaluation
lstm_model.load_state_dict(torch.load(LSTM_CKPT, map_location=device))
lstm_val_acc, lstm_val_f1, _, _                      = evaluate_loader(lstm_model, val_loader_lstm)
lstm_test_acc, lstm_test_f1, lstm_test_preds, _ = evaluate_loader(lstm_model, test_loader_lstm)
print(f"  Val  — Acc: {lstm_val_acc:.4f} | Macro-F1: {lstm_val_f1:.4f}")
print(f"  Test — Acc: {lstm_test_acc:.4f} | Macro-F1: {lstm_test_f1:.4f}")

# Training curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(lstm_train_losses, marker="o"); ax1.set_title("BiLSTM Train Loss"); ax1.set_xlabel("Epoch")
ax2.plot(lstm_val_accs,     marker="o"); ax2.set_title("BiLSTM Val Accuracy"); ax2.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("outputs/q1/q1_bilstm_training_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 1.9 — Model 4: BERT-base-uncased fine-tuning
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("1.9  DistilBERT-base-uncased fine-tuning…")

BERT_CKPT = "outputs/q1/distilbert_best.pt"

class IMDbBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[i], dtype=torch.long),
        }

# Use raw (uncleaned) texts for BERT — it handles its own tokenisation
train_bert_ds = IMDbBertDataset(list(train_ds["text"]), train_labels, bert_tokenizer, MAX_LEN_BERT)
val_bert_ds   = IMDbBertDataset(list(val_ds["text"]),   val_labels,   bert_tokenizer, MAX_LEN_BERT)
test_bert_ds  = IMDbBertDataset(list(test_ds["text"]),  test_labels,  bert_tokenizer, MAX_LEN_BERT)

train_loader_bert = DataLoader(train_bert_ds, batch_size=BATCH_SIZE_BERT, shuffle=True,  num_workers=0)
val_loader_bert   = DataLoader(val_bert_ds,   batch_size=BATCH_SIZE_BERT, shuffle=False, num_workers=0)
test_loader_bert  = DataLoader(test_bert_ds,  batch_size=BATCH_SIZE_BERT, shuffle=False, num_workers=0)

bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)

optimizer_bert = AdamW(bert_model.parameters(), lr=LR_BERT)
total_steps    = len(train_loader_bert) * BERT_EPOCHS
scheduler_bert = get_linear_schedule_with_warmup(
    optimizer_bert,
    num_warmup_steps=int(0.06 * total_steps),
    num_training_steps=total_steps,
)

def evaluate_bert(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            preds.extend(out.logits.argmax(1).cpu().tolist())
            trues.extend(lbl.cpu().tolist())
    return accuracy_score(trues, preds), f1_score(trues, preds, average="macro"), preds, trues

bert_train_losses, bert_val_accs = [], []
best_val_acc = 0.0

print("  Training DistilBERT…")
for epoch in range(1, BERT_EPOCHS + 1):
    bert_model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader_bert, desc=f"  Epoch {epoch}/{BERT_EPOCHS}", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["label"].to(device)
        optimizer_bert.zero_grad()
        out  = bert_model(input_ids=ids, attention_mask=mask, labels=lbl)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer_bert.step()
        scheduler_bert.step()
        total_loss += out.loss.item()
    avg_loss = total_loss / len(train_loader_bert)
    val_acc, val_f1, _, _ = evaluate_bert(bert_model, val_loader_bert)
    bert_train_losses.append(avg_loss)
    bert_val_accs.append(val_acc)
    print(f"  Epoch {epoch:02d} — loss: {avg_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(bert_model.state_dict(), BERT_CKPT)
        print(f"    Checkpoint saved (val_acc={val_acc:.4f})")

# Load best checkpoint for test evaluation
bert_model.load_state_dict(torch.load(BERT_CKPT, map_location=device))
bert_val_acc, bert_val_f1, _, _              = evaluate_bert(bert_model, val_loader_bert)
bert_test_acc, bert_test_f1, bert_test_preds, _ = evaluate_bert(bert_model, test_loader_bert)
print(f"  Val  — Acc: {bert_val_acc:.4f} | Macro-F1: {bert_val_f1:.4f}")
print(f"  Test — Acc: {bert_test_acc:.4f} | Macro-F1: {bert_test_f1:.4f}")

# Training curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(bert_train_losses, marker="o"); ax1.set_title("BERT Train Loss");     ax1.set_xlabel("Epoch")
ax2.plot(bert_val_accs,     marker="o"); ax2.set_title("BERT Val Accuracy");   ax2.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("outputs/q1/q1_bert_training_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 1.10-1.11 — Results table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("1.10-1.11  Results table…")

results = pd.DataFrame([
    {"Model": "TF-IDF + LR",  "Val Acc": lr_val_acc,   "Val F1": lr_val_f1,
     "Test Acc": lr_test_acc, "Test F1": lr_test_f1},
    {"Model": "TF-IDF + SVM", "Val Acc": svm_val_acc,  "Val F1": svm_val_f1,
     "Test Acc": svm_test_acc,"Test F1": svm_test_f1},
    {"Model": "BiLSTM+GloVe", "Val Acc": lstm_val_acc, "Val F1": lstm_val_f1,
     "Test Acc": lstm_test_acc,"Test F1": lstm_test_f1},
    {"Model": "DistilBERT",   "Val Acc": bert_val_acc, "Val F1": bert_val_f1,
     "Test Acc": bert_test_acc,"Test F1": bert_test_f1},
])
results = results.round(4)
print(results.to_string(index=False))
results.to_csv("outputs/q1/q1_results.csv", index=False)
print("  Saved: outputs/q1_results.csv")

# ---------------------------------------------------------------------------
# 1.12 — Confusion matrices
# ---------------------------------------------------------------------------
print("1.12  Confusion matrices…")
CLASS_NAMES = ["Negative", "Positive"]

def plot_cm(preds, trues, title, fname):
    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()

plot_cm(lr_test_preds,   test_labels, "TF-IDF + LR",  "outputs/q1/q1_cm_lr.png")
plot_cm(svm_test_preds,  test_labels, "TF-IDF + SVM", "outputs/q1/q1_cm_svm.png")
plot_cm(lstm_test_preds, test_labels, "BiLSTM+GloVe", "outputs/q1/q1_cm_bilstm.png")
plot_cm(bert_test_preds, test_labels, "DistilBERT",   "outputs/q1/q1_cm_bert.png")
print("  Saved confusion matrices to outputs/")

# ---------------------------------------------------------------------------
# 1.13 — Misclassified examples analysis (at least 5)
# ---------------------------------------------------------------------------
print("1.13  Misclassified examples analysis…")

def get_misclassified(preds, trues, texts, model_name, n=10):
    errors = []
    for i, (p, t) in enumerate(zip(preds, trues)):
        if p != t:
            errors.append({
                "model":     model_name,
                "text":      texts[i][:300],
                "true":      CLASS_NAMES[t],
                "predicted": CLASS_NAMES[p],
            })
        if len(errors) >= n:
            break
    return errors

test_texts_raw = list(test_ds["text"])
misclassified = []
misclassified += get_misclassified(lr_test_preds,   test_labels, test_texts_raw, "TF-IDF+LR",    n=5)
misclassified += get_misclassified(svm_test_preds,  test_labels, test_texts_raw, "TF-IDF+SVM",   n=5)
misclassified += get_misclassified(lstm_test_preds, test_labels, test_texts_raw, "BiLSTM+GloVe", n=5)
misclassified += get_misclassified(bert_test_preds, test_labels, test_texts_raw, "DistilBERT",   n=5)

misc_df = pd.DataFrame(misclassified)
misc_df.to_csv("outputs/q1/q1_misclassified.csv", index=False)
print(f"  Saved {len(misc_df)} misclassified examples → outputs/q1_misclassified.csv")

# Print first 5 for inspection
print("\n  Sample misclassified (first 5):")
for _, row in misc_df.head(5).iterrows():
    print(f"  [{row['model']}] True={row['true']}, Pred={row['predicted']}")
    print(f"    Text: {row['text'][:120]}…\n")

# ---------------------------------------------------------------------------
# 1.13b — Common error pattern analysis
# Patterns: short review, long review, negation keywords, irony markers
# ---------------------------------------------------------------------------
print("1.13b  Common error pattern analysis…")

NEGATION_WORDS  = {"not", "no", "never", "neither", "nor", "nothing", "nobody",
                   "nowhere", "hardly", "barely", "scarcely", "doesn't", "isn't",
                   "wasn't", "weren't", "haven't", "hadn't", "won't", "wouldn't",
                   "don't", "didn't", "can't", "couldn't", "shouldn't"}
IRONY_MARKERS   = {"but", "however", "although", "though", "despite", "yet",
                   "nevertheless", "supposedly", "apparently", "pretend", "claim"}

def detect_patterns(text: str) -> dict:
    words      = text.lower().split()
    word_count = len(words)
    word_set   = set(words)
    return {
        "length_bucket":    "short (<50)"  if word_count < 50
                            else "medium (50-200)" if word_count <= 200
                            else "long (>200)",
        "has_negation":     bool(word_set & NEGATION_WORDS),
        "has_irony_marker": bool(word_set & IRONY_MARKERS),
        "word_count":       word_count,
    }

pattern_rows = []
for _, row in misc_df.iterrows():
    p = detect_patterns(row["text"])
    pattern_rows.append({
        "model":          row["model"],
        "true":           row["true"],
        "predicted":      row["predicted"],
        "word_count":     p["word_count"],
        "length_bucket":  p["length_bucket"],
        "has_negation":   p["has_negation"],
        "has_irony_marker": p["has_irony_marker"],
    })

pattern_df = pd.DataFrame(pattern_rows)
pattern_df.to_csv("outputs/q1/q1_error_analysis.csv", index=False)

# Aggregate pattern summary per model
summary_rows = []
for model_name, grp in pattern_df.groupby("model"):
    summary_rows.append({
        "model":              model_name,
        "total_errors":       len(grp),
        "short_reviews_%":    round(100 * (grp["length_bucket"] == "short (<50)").mean(), 1),
        "long_reviews_%":     round(100 * (grp["length_bucket"] == "long (>200)").mean(), 1),
        "negation_%":         round(100 * grp["has_negation"].mean(), 1),
        "irony_marker_%":     round(100 * grp["has_irony_marker"].mean(), 1),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/q1/q1_error_pattern_summary.csv", index=False)
print(summary_df.to_string(index=False))
print("  Saved: outputs/q1/q1_error_analysis.csv")
print("  Saved: outputs/q1/q1_error_pattern_summary.csv")

print("\nPhase 1 complete. Results saved to outputs/")
