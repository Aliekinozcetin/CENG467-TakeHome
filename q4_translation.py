"""
Q4 – Machine Translation (EN→DE)
Multi30k: Seq2Seq+Bahdanau Attention (from scratch) vs Helsinki-NLP/opus-mt-en-de
"""

import os
import re
import csv
import math
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

import sacrebleu
from bert_score import score as bert_score_fn
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt",   quiet=True)

import spacy

from utils import set_seed, SEED
set_seed(SEED)
# LSTM ops are faster on CPU than MPS on Apple Silicon
device = torch.device("cpu")
os.makedirs("outputs/q4", exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_SIZE       = 29000     # full Multi30k train
MAX_LEN          = 50        # max tokens per sentence
MIN_FREQ         = 2         # vocab threshold
EMBED_DIM        = 256
HIDDEN_SIZE      = 256   # reduced from 512 for MPS efficiency
N_LAYERS         = 1     # reduced from 2 for MPS efficiency
DROPOUT          = 0.3
BATCH_SIZE       = 256   # larger batch for throughput
N_EPOCHS         = 10
LR               = 1e-3
CLIP             = 1.0
TEACHER_FORCING  = 0.5
HELSINKI_MODEL   = "Helsinki-NLP/opus-mt-en-de"
HELSINKI_BEAMS   = 5
BERT_SCORE_MODEL = "bert-base-multilingual-cased"

# ---------------------------------------------------------------------------
# 4.1 — Load Multi30k dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("4.1  Loading Multi30k dataset (EN→DE)…")
raw = load_dataset("bentrevett/multi30k")
print(f"  train={len(raw['train'])}, val={len(raw['validation'])}, test={len(raw['test'])}")
print(f"  Sample EN: {raw['train'][0]['en']}")
print(f"  Sample DE: {raw['train'][0]['de']}")

# ---------------------------------------------------------------------------
# 4.2 — Preprocessing: spaCy tokenization (lowercase)
# ---------------------------------------------------------------------------
print("\n4.2  Preprocessing: spaCy tokenization…")

# Load spaCy models (en_core_web_sm / de_core_news_sm)
# Falls back to whitespace tokenizer if models not installed
def _load_spacy(model_name: str):
    try:
        return spacy.load(model_name, disable=["ner", "parser", "tagger"])
    except OSError:
        print(f"  [WARN] spaCy model '{model_name}' not found — downloading…")
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name, disable=["ner", "parser", "tagger"])

nlp_en = _load_spacy("en_core_web_sm")
nlp_de = _load_spacy("de_core_news_sm")


def tokenize_en(text: str) -> list:
    return [tok.text.lower() for tok in nlp_en.tokenizer(text.strip())]


def tokenize_de(text: str) -> list:
    return [tok.text.lower() for tok in nlp_de.tokenizer(text.strip())]


def preprocess_split(split, src_tokenizer, tgt_tokenizer, label=""):
    src_tokens, tgt_tokens = [], []
    for ex in tqdm(split, desc=f"  Tokenizing {label}", leave=False):
        src_tokens.append(src_tokenizer(ex["en"]))
        tgt_tokens.append(tgt_tokenizer(ex["de"]))
    return src_tokens, tgt_tokens


train_en_tok, train_de_tok = preprocess_split(raw["train"],      tokenize_en, tokenize_de, "train")
val_en_tok,   val_de_tok   = preprocess_split(raw["validation"], tokenize_en, tokenize_de, "val")
test_en_tok,  test_de_tok  = preprocess_split(raw["test"],       tokenize_en, tokenize_de, "test")

src_lens = [len(s) for s in train_en_tok]
tgt_lens = [len(t) for t in train_de_tok]
print(f"  EN avg tokens: {np.mean(src_lens):.1f}, max: {max(src_lens)}")
print(f"  DE avg tokens: {np.mean(tgt_lens):.1f}, max: {max(tgt_lens)}")

# ---------------------------------------------------------------------------
# 4.3 — Vocabulary (BPE-style min_freq threshold, word-level)
# ---------------------------------------------------------------------------
print("\n4.3  Building vocabularies (min_freq={})…".format(MIN_FREQ))


class Vocabulary:
    PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"

    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = min_freq
        self.word2idx: dict = {}
        self.idx2word: dict = {}
        self._freq: Counter = Counter()
        for tok in [self.PAD, self.UNK, self.SOS, self.EOS]:
            self._add(tok, force=True)

    def _add(self, word: str, force: bool = False):
        if word not in self.word2idx and (force or self._freq[word] >= self.min_freq):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build(self, sentences: list):
        for tokens in sentences:
            self._freq.update(tokens)
        for word, cnt in self._freq.items():
            if cnt >= self.min_freq:
                self._add(word)

    def encode(self, tokens: list) -> list:
        unk = self.word2idx[self.UNK]
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self):
        return len(self.word2idx)

    @property
    def pad_idx(self): return self.word2idx[self.PAD]
    @property
    def sos_idx(self): return self.word2idx[self.SOS]
    @property
    def eos_idx(self): return self.word2idx[self.EOS]


src_vocab = Vocabulary()
tgt_vocab = Vocabulary()
src_vocab.build(train_en_tok)
tgt_vocab.build(train_de_tok)

print(f"  EN vocab size: {len(src_vocab)}")
print(f"  DE vocab size: {len(tgt_vocab)}")

vocab_stats = pd.DataFrame({
    "language": ["en", "de"],
    "vocab_size": [len(src_vocab), len(tgt_vocab)],
    "min_freq": [MIN_FREQ, MIN_FREQ],
})
vocab_stats.to_csv("outputs/q4/q4_vocab_stats.csv", index=False)

# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab, max_len=MAX_LEN):
        self.data = []
        for src, tgt in zip(src_sents, tgt_sents):
            s = [src_vocab.sos_idx] + src_vocab.encode(src[:max_len]) + [src_vocab.eos_idx]
            t = [tgt_vocab.sos_idx] + tgt_vocab.encode(tgt[:max_len]) + [tgt_vocab.eos_idx]
            self.data.append((s, t))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def collate_fn(batch, src_pad, tgt_pad):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    max_src, max_tgt = max(src_lens), max(tgt_lens)
    src_t = torch.tensor([s + [src_pad] * (max_src - len(s)) for s in src_batch], dtype=torch.long)
    tgt_t = torch.tensor([t + [tgt_pad] * (max_tgt - len(t)) for t in tgt_batch], dtype=torch.long)
    return src_t, tgt_t, torch.tensor(src_lens), torch.tensor(tgt_lens)


collate = lambda b: collate_fn(b, src_vocab.pad_idx, tgt_vocab.pad_idx)

train_ds = TranslationDataset(train_en_tok, train_de_tok, src_vocab, tgt_vocab)
val_ds   = TranslationDataset(val_en_tok,   val_de_tok,   src_vocab, tgt_vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# ---------------------------------------------------------------------------
# 4.4 — Model 1: Seq2Seq + Bahdanau Attention
# ---------------------------------------------------------------------------
print("\n4.4  Building Seq2Seq + Bahdanau Attention model…")


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.attn = nn.Linear(enc_hidden * 2 + dec_hidden, dec_hidden)
        self.v    = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        src_len   = encoder_outputs.size(1)
        hidden_e  = hidden.unsqueeze(1).expand(-1, src_len, -1)
        energy    = torch.tanh(self.attn(torch.cat([hidden_e, encoder_outputs], dim=2)))
        attn      = self.v(energy).squeeze(2)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        return torch.softmax(attn, dim=1)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_size, n_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        emb = self.drop(self.embedding(src))
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        h = torch.tanh(self.fc_h(torch.cat([h[-2], h[-1]], dim=1)))
        c = torch.tanh(self.fc_c(torch.cat([c[-2], c[-1]], dim=1)))
        return out, h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden, dec_hidden, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(enc_hidden, dec_hidden)
        self.rnn       = nn.LSTM(embed_dim + enc_hidden * 2, dec_hidden, batch_first=True)
        self.fc_out    = nn.Linear(dec_hidden + enc_hidden * 2 + embed_dim, vocab_size)
        self.drop      = nn.Dropout(dropout)

    def forward(self, tok, hidden, cell, enc_out, mask=None):
        emb     = self.drop(self.embedding(tok.unsqueeze(1)))        # [B,1,E]
        attn_w  = self.attention(hidden, enc_out, mask)              # [B,S]
        context = torch.bmm(attn_w.unsqueeze(1), enc_out)           # [B,1,H*2]
        rnn_in  = torch.cat([emb, context], dim=2)
        out, (h, c) = self.rnn(rnn_in, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        pred = self.fc_out(torch.cat([out.squeeze(1), context.squeeze(1), emb.squeeze(1)], dim=1))
        return pred, h.squeeze(0), c.squeeze(0), attn_w


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, sos_idx, eos_idx):
        super().__init__()
        self.encoder     = encoder
        self.decoder     = decoder
        self.src_pad_idx = src_pad_idx
        self.sos_idx     = sos_idx
        self.eos_idx     = eos_idx

    def _mask(self, src):
        return (src != self.src_pad_idx)

    def forward(self, src, tgt, src_lens, tf_ratio=TEACHER_FORCING):
        B, T   = src.size(0), tgt.size(1)
        V      = self.decoder.fc_out.out_features
        outputs = torch.zeros(B, T, V, device=src.device)
        enc_out, h, c = self.encoder(src, src_lens)
        mask   = self._mask(src)
        dec_in = tgt[:, 0]
        for t in range(1, T):
            pred, h, c, _ = self.decoder(dec_in, h, c, enc_out, mask)
            outputs[:, t] = pred
            dec_in = tgt[:, t] if random.random() < tf_ratio else pred.argmax(1)
        return outputs

    @torch.no_grad()
    def translate(self, src, src_lens, max_len=MAX_LEN):
        self.eval()
        enc_out, h, c = self.encoder(src, src_lens)
        mask  = self._mask(src)
        dec_in = torch.full((src.size(0),), self.sos_idx, dtype=torch.long, device=src.device)
        seqs   = [[] for _ in range(src.size(0))]
        for _ in range(max_len):
            pred, h, c, _ = self.decoder(dec_in, h, c, enc_out, mask)
            dec_in = pred.argmax(1)
            for i, tok in enumerate(dec_in.tolist()):
                if tok != self.eos_idx:
                    seqs[i].append(tok)
        return seqs


encoder = Encoder(len(src_vocab), EMBED_DIM, HIDDEN_SIZE, N_LAYERS, DROPOUT)
decoder = Decoder(len(tgt_vocab), EMBED_DIM, HIDDEN_SIZE, HIDDEN_SIZE, DROPOUT)
model   = Seq2Seq(encoder, decoder, src_vocab.pad_idx, tgt_vocab.sos_idx, tgt_vocab.eos_idx).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Seq2Seq parameters: {n_params:,}")
print(f"  Encoder: 1-layer BiLSTM hidden={HIDDEN_SIZE} (CPU — faster than MPS for LSTM)")
print(f"  Decoder: 1-layer LSTM + Bahdanau Attention")
print(f"  embed_dim={EMBED_DIM}, dropout={DROPOUT}, teacher_forcing={TEACHER_FORCING}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)
best_val  = float("inf")
best_path = "outputs/q4/seq2seq_best.pt"
history   = []

print(f"\n  Training on {device} — {N_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, clip={CLIP}")

for epoch in range(1, N_EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    t0 = time.time()
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch:2d}/{N_EPOCHS} [train]", leave=False)
    for src, tgt, sl, _ in pbar:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        out = model(src, tgt, sl)
        loss = criterion(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src, tgt, sl, _ in tqdm(val_loader, desc=f"  Epoch {epoch:2d}/{N_EPOCHS} [val]  ", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, sl, tf_ratio=0.0)
            val_loss += criterion(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1)).item()
    val_loss /= len(val_loader)

    train_ppl = math.exp(min(train_loss, 20))
    val_ppl   = math.exp(min(val_loss, 20))
    elapsed   = time.time() - t0
    history.append((epoch, train_loss, val_loss, train_ppl, val_ppl))

    marker = ""
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), best_path)
        marker = " ✓"

    print(f"  Epoch {epoch:2d}/{N_EPOCHS} | train_loss={train_loss:.4f} ppl={train_ppl:.1f}"
          f" | val_loss={val_loss:.4f} ppl={val_ppl:.1f} | {elapsed:.0f}s{marker}")

# Save training curve CSV
pd.DataFrame(history, columns=["epoch","train_loss","val_loss","train_ppl","val_ppl"]) \
  .to_csv("outputs/q4/q4_seq2seq_training.csv", index=False)

# Plot training curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = [h[0] for h in history]
axes[0].plot(epochs, [h[1] for h in history], label="train")
axes[0].plot(epochs, [h[2] for h in history], label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(epochs, [h[3] for h in history], label="train")
axes[1].plot(epochs, [h[4] for h in history], label="val")
axes[1].set_title("Perplexity"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout()
plt.savefig("outputs/q4/q4_seq2seq_curve.png", dpi=150)
plt.close()
print(f"  Best val_loss={best_val:.4f}  ppl={math.exp(min(best_val,20)):.1f}")
print("  Training curve → outputs/q4/q4_seq2seq_curve.png")

# Load best checkpoint
model.load_state_dict(torch.load(best_path, map_location=device))

# ---------------------------------------------------------------------------
# 4.5 — Model 2: Helsinki-NLP/opus-mt-en-de (pretrained inference)
# ---------------------------------------------------------------------------
print("\n4.5  Helsinki-NLP/opus-mt-en-de — pretrained Transformer inference…")
test_en_raw = [x["en"] for x in raw["test"]]
test_de_raw = [x["de"] for x in raw["test"]]

hel_tokenizer = MarianTokenizer.from_pretrained(HELSINKI_MODEL)
hel_model     = MarianMTModel.from_pretrained(HELSINKI_MODEL).to(device)
hel_model.eval()

helsinki_hyps = []
hel_batch = 32
for i in tqdm(range(0, len(test_en_raw), hel_batch), desc="  Helsinki inference"):
    batch   = test_en_raw[i: i + hel_batch]
    encoded = hel_tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = hel_model.generate(**encoded, num_beams=HELSINKI_BEAMS,
                                  max_length=128, early_stopping=True)
    helsinki_hyps.extend(hel_tokenizer.batch_decode(out, skip_special_tokens=True))

print(f"  Helsinki inference done — {len(helsinki_hyps)} translations")

# ---------------------------------------------------------------------------
# Seq2Seq inference on test set
# ---------------------------------------------------------------------------
print("\n  Seq2Seq inference on test set…")

def _infer_collate(batch):
    src_batch, _ = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    max_s = max(src_lens)
    padded = torch.tensor([s + [src_vocab.pad_idx] * (max_s - len(s)) for s in src_batch], dtype=torch.long)
    return padded, torch.tensor(src_lens)

test_ds_inf = TranslationDataset(test_en_tok, [["dummy"]] * len(test_en_tok), src_vocab, tgt_vocab)
inf_loader  = DataLoader(test_ds_inf, batch_size=64, shuffle=False, collate_fn=_infer_collate)

seq2seq_hyps = []
model.eval()
with torch.no_grad():
    for src, sl in tqdm(inf_loader, desc="  Seq2Seq inference"):
        src = src.to(device)
        generated = model.translate(src, sl)
        for tok_ids in generated:
            words = [tgt_vocab.idx2word.get(i, "<unk>") for i in tok_ids
                     if i not in (tgt_vocab.sos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx)]
            seq2seq_hyps.append(" ".join(words))

print(f"  Seq2Seq inference done — {len(seq2seq_hyps)} translations")

# ---------------------------------------------------------------------------
# 4.6-4.9 — Evaluation: BLEU, METEOR, ChrF, BERTScore
# ---------------------------------------------------------------------------
print("\n4.6-4.9  Computing metrics (BLEU, METEOR, ChrF, BERTScore)…")
from nltk.translate.meteor_score import meteor_score as nltk_meteor


def compute_all_metrics(hyps: list, refs: list, label: str) -> dict:
    bleu  = sacrebleu.corpus_bleu(hyps, [refs])
    chrf  = sacrebleu.corpus_chrf(hyps, [refs])
    meteor_vals = [nltk_meteor([r.split()], h.split()) for h, r in zip(hyps, refs)]
    meteor = float(np.mean(meteor_vals))
    _, _, F1 = bert_score_fn(hyps, refs, lang="de",
                              model_type=BERT_SCORE_MODEL,
                              verbose=False, device=str(device))
    bs = float(F1.mean())
    print(f"\n  [{label}]")
    print(f"    BLEU:      {bleu.score:.4f}")
    print(f"    ChrF:      {chrf.score:.4f}")
    print(f"    METEOR:    {meteor:.4f}")
    print(f"    BERTScore: {bs:.4f}")
    return {"model": label, "bleu": round(bleu.score, 4),
            "chrf": round(chrf.score, 4), "meteor": round(meteor, 4),
            "bertscore_f1": round(bs, 4)}


results = []
results.append(compute_all_metrics(seq2seq_hyps,  test_de_raw, "Seq2Seq+Bahdanau"))
results.append(compute_all_metrics(helsinki_hyps, test_de_raw, "Helsinki-NLP/opus-mt-en-de"))

# ---------------------------------------------------------------------------
# 4.10 — Save results table
# ---------------------------------------------------------------------------
print("\n4.10  Saving results table…")
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/q4/q4_results.csv", index=False)
print("  Results → outputs/q4/q4_results.csv")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
metrics  = ["bleu", "chrf", "meteor", "bertscore_f1"]
x        = np.arange(len(metrics))
width    = 0.35
for i, row in results_df.iterrows():
    vals = [row[m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=row["model"])
ax.set_xticks(x + width / 2)
ax.set_xticklabels(["BLEU", "ChrF", "METEOR", "BERTScore-F1"])
ax.set_title("Q4 Translation Metrics Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/q4/q4_metrics_comparison.png", dpi=150)
plt.close()
print("  Metrics chart → outputs/q4/q4_metrics_comparison.png")

# ---------------------------------------------------------------------------
# 4.11 — Qualitative examples
# ---------------------------------------------------------------------------
print("\n4.11  Qualitative examples…")
qual_indices = [0, 100, 200, 400, 700]
qual_rows = []
for idx in qual_indices:
    qual_rows.append({
        "source_en":    test_en_raw[idx],
        "reference_de": test_de_raw[idx],
        "seq2seq":      seq2seq_hyps[idx],
        "helsinki":     helsinki_hyps[idx],
    })

pd.DataFrame(qual_rows).to_csv("outputs/q4/q4_qualitative.csv", index=False)
print("  Qualitative examples → outputs/q4/q4_qualitative.csv")
for r in qual_rows[:3]:
    print(f"\n    SRC:  {r['source_en']}")
    print(f"    REF:  {r['reference_de']}")
    print(f"    S2S:  {r['seq2seq']}")
    print(f"    HEL:  {r['helsinki']}")

# ---------------------------------------------------------------------------
# 4.12-4.13 — Rare word & long-range analysis + summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4.12  Rare word handling & long-range dependency analysis…")

# Collect OOV rates on test set
oov_rates = []
for toks in test_en_tok:
    oov = sum(1 for t in toks if t not in src_vocab.word2idx or src_vocab.word2idx[t] == src_vocab.word2idx["<unk>"])
    oov_rates.append(oov / max(len(toks), 1))
oov_mean = float(np.mean(oov_rates))

# Sentence-level BLEU by length bucket for Seq2Seq
from sacrebleu.metrics import BLEU as SacreBLEU
_bleu = SacreBLEU(effective_order=True)

def sentence_bleu(hyp, ref):
    try:
        return _bleu.sentence_score(hyp, [ref]).score
    except Exception:
        return 0.0

buckets = {"short (≤10)": [], "medium (11-25)": [], "long (>25)": []}
for hyp, ref, src_t in zip(seq2seq_hyps, test_de_raw, test_en_tok):
    ln = len(src_t)
    s  = sentence_bleu(hyp, ref)
    if ln <= 10:   buckets["short (≤10)"].append(s)
    elif ln <= 25: buckets["medium (11-25)"].append(s)
    else:          buckets["long (>25)"].append(s)

analysis_rows = []
for bucket, scores in buckets.items():
    analysis_rows.append({
        "bucket":        bucket,
        "count":         len(scores),
        "seq2seq_bleu":  round(float(np.mean(scores)) if scores else 0.0, 4),
    })
# Also add Helsinki bucket BLEU for comparison
buckets_hel = {"short (≤10)": [], "medium (11-25)": [], "long (>25)": []}
for hyp, ref, src_t in zip(helsinki_hyps, test_de_raw, test_en_tok):
    ln = len(src_t)
    s  = sentence_bleu(hyp, ref)
    if ln <= 10:   buckets_hel["short (≤10)"].append(s)
    elif ln <= 25: buckets_hel["medium (11-25)"].append(s)
    else:          buckets_hel["long (>25)"].append(s)
for row, (bucket, scores) in zip(analysis_rows, buckets_hel.items()):
    row["helsinki_bleu"] = round(float(np.mean(scores)) if scores else 0.0, 4)

analysis_df = pd.DataFrame(analysis_rows)
analysis_df["test_oov_rate"] = round(oov_mean, 4)
analysis_df.to_csv("outputs/q4/q4_length_analysis.csv", index=False)
print(f"  Test OOV rate: {oov_mean:.2%}")
print(analysis_df.to_string(index=False))
print("  Length analysis → outputs/q4/q4_length_analysis.csv")

print("\n4.13  Metric interpretation:")
print("  BLEU  — n-gram precision (surface overlap, brevity penalty)")
print("  ChrF  — character n-gram F-score (robust to morphology, good for DE)")
print("  METEOR— unigram match with synonyms + word order (recall-oriented)")
print("  BERTScore — contextual embedding cosine similarity (semantic similarity)")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<35} {'BLEU':>7} {'ChrF':>7} {'METEOR':>8} {'BERT-F1':>9}")
print("-" * 60)
for r in results:
    print(f"{r['model']:<35} {r['bleu']:>7.4f} {r['chrf']:>7.4f}"
          f" {r['meteor']:>8.4f} {r['bertscore_f1']:>9.4f}")
print("=" * 60)
print("\nQ4 complete. Outputs in outputs/q4/")
