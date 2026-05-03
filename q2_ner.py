"""
Q2 – Named Entity Recognition
CoNLL-2003: BiLSTM-CRF vs DistilBERT-base-cased token classification
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_dataset, load_from_disk
from transformers import (
    DistilBertTokenizerFast, DistilBertForTokenClassification,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from utils import device, set_seed, SEED

set_seed(SEED)
os.makedirs("outputs/q2", exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LSTM_HIDDEN   = 256
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.3
LSTM_LR       = 1e-3
LSTM_EPOCHS   = 15
LSTM_BATCH    = 32
CHAR_EMBED    = 30
CHAR_HIDDEN   = 50
WORD_EMBED    = 100

BERT_LR       = 3e-5
BERT_EPOCHS   = 5
BERT_BATCH    = 16
BERT_MAX_LEN  = 128

# ---------------------------------------------------------------------------
# 2.1 — Load CoNLL-2003
# ---------------------------------------------------------------------------
print("=" * 60)
print("2.1  Loading CoNLL-2003 dataset…")
CONLL_CACHE = "outputs/q2/conll2003_cached"
if os.path.exists(CONLL_CACHE):
    raw = load_from_disk(CONLL_CACHE)
    print("  Loaded from local cache.")
else:
    raw = load_dataset("conll2003", trust_remote_code=True)
    raw.save_to_disk(CONLL_CACHE)
    print("  Downloaded and cached to disk.")
print(f"  train={len(raw['train'])}, val={len(raw['validation'])}, test={len(raw['test'])}")

# ---------------------------------------------------------------------------
# 2.2 — BIO tagging structure
# ---------------------------------------------------------------------------
print("\n2.2  BIO tagging structure…")
label_list = raw["train"].features["ner_tags"].feature.names  # ClassLabel.names
id2label   = {i: l for i, l in enumerate(label_list)}
label2id   = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

print(f"  Labels ({num_labels}): {label_list}")
print("  Entity types: PER, ORG, LOC, MISC")
print("  Scheme: BIO — B-=begin, I-=inside, O=outside")

sample = raw["train"][0]
print("\n  Sample sentence:")
for token, tag_id in zip(sample["tokens"], sample["ner_tags"]):
    print(f"    {token:15s} → {id2label[tag_id]}")

tag_counts = Counter()
for ex in raw["train"]:
    for t in ex["ner_tags"]:
        tag_counts[id2label[t]] += 1
tag_df = pd.DataFrame(tag_counts.most_common(), columns=["Tag", "Count"])
tag_df.to_csv("outputs/q2/q2_tag_distribution.csv", index=False)
print(f"\n  Tag distribution:")
print(tag_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 2.3 — Token-label alignment strategy
# ---------------------------------------------------------------------------
print("\n2.3  Token-label alignment strategy:")
print("  - First subword token of each word → real label")
print("  - Subsequent subword tokens        → -100 (ignored in loss)")
print("  - Special tokens [CLS], [SEP]      → -100")

# ===========================================================================
# 2.4 — Model 1: BiLSTM-CRF
# ===========================================================================
print("\n" + "=" * 60)
print("2.4  BiLSTM-CRF…")

def build_word_vocab(dataset):
    counter = Counter()
    for ex in dataset:
        counter.update([w.lower() for w in ex["tokens"]])
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in counter:
        vocab[w] = len(vocab)
    return vocab

def build_char_vocab(dataset):
    chars = set()
    for ex in dataset:
        for w in ex["tokens"]:
            chars.update(w)
    vocab = {"<pad>": 0, "<unk>": 1}
    for c in sorted(chars):
        vocab[c] = len(vocab)
    return vocab

word_vocab = build_word_vocab(raw["train"])
char_vocab = build_char_vocab(raw["train"])
print(f"  Word vocab: {len(word_vocab)}, Char vocab: {len(char_vocab)}")

GLOVE_PATH = os.path.expanduser("~/.cache/glove/glove.6B.100d.txt")

def load_glove(path, vocab, dim=100):
    embed = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    embed[0] = 0.0
    if not os.path.exists(path):
        print(f"  GloVe not found, using random init.")
        return embed
    found = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in vocab:
                embed[vocab[parts[0]]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"  GloVe: {found}/{len(vocab)} words matched.")
    return embed

glove_matrix = load_glove(GLOVE_PATH, word_vocab)

MAX_SEQ_LEN  = 128
MAX_WORD_LEN = 20

def encode_sentence(tokens, labels):
    word_ids  = [word_vocab.get(w.lower(), 1) for w in tokens[:MAX_SEQ_LEN]]
    label_ids = [labels[i] for i in range(min(len(labels), MAX_SEQ_LEN))]
    char_ids  = []
    for w in tokens[:MAX_SEQ_LEN]:
        cids = [char_vocab.get(c, 1) for c in w[:MAX_WORD_LEN]]
        cids += [0] * (MAX_WORD_LEN - len(cids))
        char_ids.append(cids)
    pad = MAX_SEQ_LEN - len(word_ids)
    word_ids  += [0] * pad
    label_ids += [-100] * pad
    char_ids  += [[0] * MAX_WORD_LEN] * pad
    length = min(len(tokens), MAX_SEQ_LEN)
    return word_ids, char_ids, label_ids, length

class NERDataset(Dataset):
    def __init__(self, split):
        self.data = []
        for ex in raw[split]:
            w, c, l, length = encode_sentence(ex["tokens"], ex["ner_tags"])
            self.data.append((w, c, l, length))

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        w, c, l, length = self.data[i]
        return (torch.tensor(w, dtype=torch.long),
                torch.tensor(c, dtype=torch.long),
                torch.tensor(l, dtype=torch.long),
                length)

train_ner = NERDataset("train")
val_ner   = NERDataset("validation")
test_ner  = NERDataset("test")

train_loader = DataLoader(train_ner, batch_size=LSTM_BATCH, shuffle=True)
val_loader   = DataLoader(val_ner,   batch_size=LSTM_BATCH)
test_loader  = DataLoader(test_ner,  batch_size=LSTM_BATCH)

class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags    = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_trans = nn.Parameter(torch.randn(num_tags))
        self.end_trans   = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        return -self._score(emissions, tags, mask) + self._normalizer(emissions, mask)

    def decode(self, emissions, mask):
        return self._viterbi(emissions, mask)

    def _score(self, emissions, tags, mask):
        batch, seq, _ = emissions.shape
        score = self.start_trans[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for i in range(1, seq):
            m = mask[:, i]
            score += (self.transitions[tags[:, i-1], tags[:, i]] +
                      emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1)) * m
        last = mask.long().sum(1) - 1
        last_tags = tags.gather(1, last.unsqueeze(1)).squeeze(1)
        score += self.end_trans[last_tags]
        return score

    def _normalizer(self, emissions, mask):
        _, seq, _ = emissions.shape
        score = self.start_trans + emissions[:, 0]
        for i in range(1, seq):
            s = score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)
            s = torch.logsumexp(s, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), s, score)
        return torch.logsumexp(score + self.end_trans, dim=1)

    def _viterbi(self, emissions, mask):
        batch, seq, _ = emissions.shape
        score   = self.start_trans + emissions[:, 0]
        history = []
        for i in range(1, seq):
            s = score.unsqueeze(2) + self.transitions
            best_score, best_tag = s.max(dim=1)
            s = best_score + emissions[:, i]
            score = torch.where(mask[:, i].unsqueeze(1), s, score)
            history.append(best_tag)
        _, best_last = score.max(dim=1)
        paths = []
        for b in range(batch):
            tag  = best_last[b].item()
            path = [tag]
            length = mask[b].long().sum().item()
            for h in reversed(history[:length - 1]):
                tag = h[b, tag].item()
                path.append(tag)
            path.reverse()
            paths.append(path)
        return paths

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv  = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        b, seq, char = x.shape
        x   = x.view(b * seq, char)
        emb = self.embed(x).permute(0, 2, 1)
        out = torch.relu(self.conv(emb)).max(dim=2)[0]
        return out.view(b, seq, -1)

class BiLSTMCRF(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, num_tags, pretrained=None):
        super().__init__()
        self.word_embed = nn.Embedding(word_vocab_size, WORD_EMBED, padding_idx=0)
        if pretrained is not None:
            self.word_embed.weight.data.copy_(torch.from_numpy(pretrained))
        self.char_cnn = CharCNN(char_vocab_size, CHAR_EMBED, CHAR_HIDDEN)
        self.drop     = nn.Dropout(LSTM_DROPOUT)
        self.lstm     = nn.LSTM(WORD_EMBED + CHAR_HIDDEN, LSTM_HIDDEN // 2,
                                num_layers=LSTM_LAYERS, batch_first=True,
                                bidirectional=True,
                                dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0)
        self.fc  = nn.Linear(LSTM_HIDDEN, num_tags)
        self.crf = CRF(num_tags)

    def _emit(self, words, chars):
        w = self.drop(self.word_embed(words))
        c = self.drop(self.char_cnn(chars))
        out, _ = self.lstm(torch.cat([w, c], dim=-1))
        return self.fc(self.drop(out))

    def forward(self, words, chars, tags, mask):
        emissions = self._emit(words, chars)
        return self.crf(emissions, tags.clamp(min=0), mask).mean()

    def predict(self, words, chars, mask):
        return self.crf.decode(self._emit(words, chars), mask)

bilstm_crf     = BiLSTMCRF(len(word_vocab), len(char_vocab), num_labels, pretrained=glove_matrix).to(device)
optimizer_lstm = torch.optim.Adam(bilstm_crf.parameters(), lr=LSTM_LR)

def evaluate_bilstm(model, loader):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for words, chars, tags, lengths in loader:
            words, chars, tags = words.to(device), chars.to(device), tags.to(device)
            mask  = (tags != -100)
            preds = model.predict(words, chars, mask)
            for i, (pred, length) in enumerate(zip(preds, lengths)):
                true_seq = tags.clamp(min=0)[i, :length].cpu().tolist()
                all_trues.append([id2label[t] for t in true_seq])
                all_preds.append([id2label[p] for p in pred[:length]])
    return (precision_score(all_trues, all_preds),
            recall_score(all_trues, all_preds),
            f1_score(all_trues, all_preds),
            all_preds, all_trues)

bilstm_losses, bilstm_val_f1s = [], []
BILSTM_CKPT      = "outputs/q2/bilstm_crf_best.pt"
EARLY_STOP_NER   = 3
best_val_f1_lstm = 0.0
no_improve_lstm  = 0

print("  Training BiLSTM-CRF…")
for epoch in range(1, LSTM_EPOCHS + 1):
    bilstm_crf.train()
    total_loss = 0.0
    for words, chars, tags, _ in tqdm(train_loader, desc=f"  Epoch {epoch:02d}/{LSTM_EPOCHS}", leave=False):
        words, chars, tags = words.to(device), chars.to(device), tags.to(device)
        mask = (tags != -100)
        optimizer_lstm.zero_grad()
        loss = bilstm_crf(words, chars, tags, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(bilstm_crf.parameters(), 5.0)
        optimizer_lstm.step()
        total_loss += loss.item()
    avg = total_loss / len(train_loader)
    _, _, vf1, _, _ = evaluate_bilstm(bilstm_crf, val_loader)
    bilstm_losses.append(avg)
    bilstm_val_f1s.append(vf1)

    marker = ""
    if vf1 > best_val_f1_lstm:
        best_val_f1_lstm = vf1
        torch.save(bilstm_crf.state_dict(), BILSTM_CKPT)
        no_improve_lstm = 0
        marker = " ✓"
    else:
        no_improve_lstm += 1

    print(f"  Epoch {epoch:02d} — loss: {avg:.4f} | val_F1: {vf1:.4f}{marker}")

    if no_improve_lstm >= EARLY_STOP_NER:
        print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_NER} epochs)")
        break

bilstm_crf.load_state_dict(torch.load(BILSTM_CKPT, map_location=device))
lstm_vp, lstm_vr, lstm_vf1, _, _                  = evaluate_bilstm(bilstm_crf, val_loader)
lstm_tp, lstm_tr, lstm_tf1, lstm_preds, lstm_trues = evaluate_bilstm(bilstm_crf, test_loader)
print(f"  Val  — P:{lstm_vp:.4f} R:{lstm_vr:.4f} F1:{lstm_vf1:.4f}")
print(f"  Test — P:{lstm_tp:.4f} R:{lstm_tr:.4f} F1:{lstm_tf1:.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(bilstm_losses,   marker="o"); a1.set_title("BiLSTM-CRF Loss"); a1.set_xlabel("Epoch")
a2.plot(bilstm_val_f1s, marker="o"); a2.set_title("BiLSTM-CRF Val F1"); a2.set_xlabel("Epoch")
plt.tight_layout(); plt.savefig("outputs/q2/q2_bilstm_crf_curve.png", dpi=150); plt.close()

# ===========================================================================
# 2.5 — Model 2: DistilBERT-base-cased
# ===========================================================================
print("\n" + "=" * 60)
print("2.5  DistilBERT-base-cased token classification…")

bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

class NERBertDataset(Dataset):
    def __init__(self, split):
        self.examples = []
        for ex in raw[split]:
            enc = bert_tokenizer(
                ex["tokens"], is_split_into_words=True,
                max_length=BERT_MAX_LEN, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            word_ids  = enc.word_ids()
            ner_tags  = ex["ner_tags"]
            label_ids, prev = [], None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    label_ids.append(ner_tags[wid] if wid < len(ner_tags) else -100)
                else:
                    label_ids.append(-100)
                prev = wid
            self.examples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(label_ids, dtype=torch.long),
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

print("  Building BERT datasets…")
train_bert = NERBertDataset("train")
val_bert   = NERBertDataset("validation")
test_bert  = NERBertDataset("test")

train_loader_bert = DataLoader(train_bert, batch_size=BERT_BATCH, shuffle=True,  num_workers=0)
val_loader_bert   = DataLoader(val_bert,   batch_size=BERT_BATCH, shuffle=False, num_workers=0)
test_loader_bert  = DataLoader(test_bert,  batch_size=BERT_BATCH, shuffle=False, num_workers=0)

bert_model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-cased", num_labels=num_labels,
    id2label=id2label, label2id=label2id,
).to(device)

optimizer_bert = AdamW(bert_model.parameters(), lr=BERT_LR)
total_steps    = len(train_loader_bert) * BERT_EPOCHS
scheduler      = get_linear_schedule_with_warmup(
    optimizer_bert,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

BERT_CKPT = "outputs/q2/distilbert_ner_best.pt"

def evaluate_bert_ner(model, loader):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds  = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
            for i in range(labels.shape[0]):
                ps, ts = [], []
                for p, t in zip(preds[i].cpu().tolist(), labels[i].cpu().tolist()):
                    if t == -100: continue
                    ps.append(id2label[p]); ts.append(id2label[t])
                all_preds.append(ps); all_trues.append(ts)
    return (precision_score(all_trues, all_preds),
            recall_score(all_trues, all_preds),
            f1_score(all_trues, all_preds),
            all_preds, all_trues)

bert_losses, bert_val_f1s = [], []
best_val_f1 = 0.0
print("  Training DistilBERT-base-cased…")
for epoch in range(1, BERT_EPOCHS + 1):
    bert_model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader_bert, desc=f"  Epoch {epoch:02d}/{BERT_EPOCHS}", leave=False):
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer_bert.zero_grad()
        loss = bert_model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer_bert.step(); scheduler.step()
        total_loss += loss.item()
    avg = total_loss / len(train_loader_bert)
    _, _, vf1, _, _ = evaluate_bert_ner(bert_model, val_loader_bert)
    bert_losses.append(avg); bert_val_f1s.append(vf1)
    print(f"  Epoch {epoch:02d} — loss: {avg:.4f} | val_F1: {vf1:.4f}")
    if vf1 > best_val_f1:
        best_val_f1 = vf1
        torch.save(bert_model.state_dict(), BERT_CKPT)
        print(f"    Checkpoint saved (val_F1={vf1:.4f})")

bert_model.load_state_dict(torch.load(BERT_CKPT, map_location=device))
bert_vp, bert_vr, bert_vf1, _, _                       = evaluate_bert_ner(bert_model, val_loader_bert)
bert_tp, bert_tr, bert_tf1, bert_preds, bert_trues = evaluate_bert_ner(bert_model, test_loader_bert)
print(f"  Val  — P:{bert_vp:.4f} R:{bert_vr:.4f} F1:{bert_vf1:.4f}")
print(f"  Test — P:{bert_tp:.4f} R:{bert_tr:.4f} F1:{bert_tf1:.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(bert_losses,   marker="o"); a1.set_title("BERT NER Loss");   a1.set_xlabel("Epoch")
a2.plot(bert_val_f1s, marker="o"); a2.set_title("BERT NER Val F1"); a2.set_xlabel("Epoch")
plt.tight_layout(); plt.savefig("outputs/q2/q2_bert_curve.png", dpi=150); plt.close()

# ===========================================================================
# 2.6-2.7 — Results
# ===========================================================================
print("\n" + "=" * 60)
print("2.6-2.7  Results…")

results = pd.DataFrame([
    {"Model": "BiLSTM-CRF",
     "Val P": lstm_vp, "Val R": lstm_vr, "Val F1": lstm_vf1,
     "Test P": lstm_tp, "Test R": lstm_tr, "Test F1": lstm_tf1},
    {"Model": "DistilBERT-base-cased",
     "Val P": bert_vp, "Val R": bert_vr, "Val F1": bert_vf1,
     "Test P": bert_tp, "Test R": bert_tr, "Test F1": bert_tf1},
]).round(4)
print(results.to_string(index=False))
results.to_csv("outputs/q2/q2_results.csv", index=False)

print("\n  BiLSTM-CRF — per entity:")
print(classification_report(lstm_trues, lstm_preds))
print("\n  DistilBERT-base-cased — per entity:")
print(classification_report(bert_trues, bert_preds))

def per_entity_rows(trues, preds, model_name):
    report = classification_report(trues, preds, output_dict=True)
    rows = []
    for ent in ["PER", "ORG", "LOC", "MISC"]:
        if ent in report:
            r = report[ent]
            rows.append({"Model": model_name, "Entity": ent,
                         "Precision": round(r["precision"], 4),
                         "Recall":    round(r["recall"], 4),
                         "F1":        round(r["f1-score"], 4),
                         "Support":   int(r["support"])})
    return rows

entity_df = pd.DataFrame(
    per_entity_rows(lstm_trues, lstm_preds, "BiLSTM-CRF") +
    per_entity_rows(bert_trues, bert_preds, "DistilBERT-base-cased")
)
entity_df.to_csv("outputs/q2/q2_per_entity_results.csv", index=False)
print(entity_df.to_string(index=False))

# ===========================================================================
# 2.8 — Error analysis
# ===========================================================================
print("\n" + "=" * 60)
print("2.8  Error analysis…")

def error_analysis(trues, preds, model_name, split, n=20):
    boundary, type_conf = [], []
    total_boundary = total_type_conf = 0
    dataset = raw[split]
    for i, (ts, ps) in enumerate(zip(trues, preds)):
        tokens = dataset[i]["tokens"][:len(ts)]
        for j, (t, p) in enumerate(zip(ts, ps)):
            if t == p: continue
            tok    = tokens[j] if j < len(tokens) else "?"
            t_type = t.split("-")[-1] if "-" in t else t
            p_type = p.split("-")[-1] if "-" in p else p
            if t_type == p_type:
                total_boundary += 1
                if len(boundary) < n:
                    boundary.append({"model": model_name, "token": tok, "true": t, "pred": p, "error": "boundary"})
            elif t != "O" and p != "O":
                total_type_conf += 1
                if len(type_conf) < n:
                    type_conf.append({"model": model_name, "token": tok, "true": t, "pred": p, "error": "type_confusion"})
    return boundary, type_conf, total_boundary, total_type_conf

lb, lt, lb_total, lt_total = error_analysis(lstm_trues, lstm_preds, "BiLSTM-CRF",           "test")
bb, bt, bb_total, bt_total = error_analysis(bert_trues, bert_preds, "DistilBERT-base-cased", "test")

errors_df = pd.DataFrame(lb + lt + bb + bt)
errors_df.to_csv("outputs/q2/q2_error_analysis.csv", index=False)

# Error count summary
error_summary = pd.DataFrame([
    {"model": "BiLSTM-CRF",           "boundary_errors": lb_total, "type_confusion_errors": lt_total,
     "total_errors": lb_total + lt_total},
    {"model": "DistilBERT-base-cased", "boundary_errors": bb_total, "type_confusion_errors": bt_total,
     "total_errors": bb_total + bt_total},
])
error_summary.to_csv("outputs/q2/q2_error_summary.csv", index=False)
print(f"  Saved → outputs/q2/q2_error_analysis.csv")
print(f"  Saved → outputs/q2/q2_error_summary.csv")
print(error_summary.to_string(index=False))

print("\n  Sample boundary errors:")
for e in (lb + bb)[:5]:
    print(f"    [{e['model']}] '{e['token']}' | true={e['true']} → pred={e['pred']}")

print("\n  Sample type confusion errors:")
for e in (lt + bt)[:5]:
    print(f"    [{e['model']}] '{e['token']}' | true={e['true']} → pred={e['pred']}")

print("\nPhase 2 complete. Results saved to outputs/")
