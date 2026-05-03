"""
Q3 – Text Summarization
CNN/DailyMail subset: LexRank (extractive) vs BART-large-cnn (abstractive)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from datasets import load_dataset

# Extractive — sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Abstractive — BART
from transformers import pipeline

# Metrics
from rouge_score import rouge_scorer
import evaluate
import bert_score as bert_score_lib

import nltk
nltk.download("punkt",    quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from utils import device, set_seed, SEED

set_seed(SEED)
os.makedirs("outputs/q3", exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_SIZE  = 1000
LEXRANK_SENTENCES = 3
BART_MODEL = "facebook/bart-large-cnn"   # fine-tuned on CNN/DM — direct inference
BART_MAX_INPUT  = 1024
BART_MAX_TARGET = 128
BART_MIN_TARGET = 30
BART_BATCH = 8

# ---------------------------------------------------------------------------
# 3.1 — Load CNN/DailyMail subset (test only — no training needed)
# ---------------------------------------------------------------------------
print("=" * 60)
print("3.1  Loading CNN/DailyMail dataset (test subset only)…")
raw = load_dataset("cnn_dailymail", "3.0.0")

# LexRank: unsupervised, BART: pretrained — no training, evaluation only
test_sub = raw["test"].shuffle(seed=SEED).select(range(TEST_SIZE))
print(f"  test subset: {len(test_sub)}")

# ---------------------------------------------------------------------------
# 3.2 — Dataset structure
# ---------------------------------------------------------------------------
print("\n3.2  Dataset structure…")
sample = test_sub[0]
print(f"  Fields: {list(sample.keys())}")
print(f"  Article length (words): {len(sample['article'].split())}")
print(f"  Highlights length (words): {len(sample['highlights'].split())}")
print(f"\n  Sample article (first 200 chars):\n  {sample['article'][:200]}…")
print(f"\n  Sample highlights:\n  {sample['highlights'][:300]}")

articles   = list(test_sub["article"])
references = list(test_sub["highlights"])

# ---------------------------------------------------------------------------
# 3.3 — Model 1: LexRank (Extractive)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3.3  LexRank summarization…")

LANGUAGE = "english"
stemmer  = Stemmer(LANGUAGE)
lexrank  = LexRankSummarizer(stemmer)
lexrank.stop_words = get_stop_words(LANGUAGE)

def lexrank_summarize(text, n_sentences=LEXRANK_SENTENCES):
    try:
        parser  = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        summary = lexrank(parser.document, n_sentences)
        return " ".join(str(s) for s in summary)
    except Exception:
        return text[:512]

lexrank_summaries = []
for article in tqdm(articles, desc="  LexRank"):
    lexrank_summaries.append(lexrank_summarize(article))

print(f"  Done. Sample output:\n  {lexrank_summaries[0][:300]}…")

# ---------------------------------------------------------------------------
# 3.4 — Model 2: BART-large-cnn (Abstractive — direct inference)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"3.4  BART abstractive summarization ({BART_MODEL})…")

device_str = "mps" if torch.backends.mps.is_available() else "cpu"
bart_pipe = pipeline(
    "summarization",
    model=BART_MODEL,
    device=device_str,
    dtype=torch.float32,
    model_kwargs={"max_position_embeddings": BART_MAX_INPUT},
)

def bart_summarize_batch(texts, batch_size=BART_BATCH):
    summaries = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  BART"):
        batch = texts[i : i + batch_size]
        # Character-level truncation to stay within BART's 1024 token limit
        batch_trunc = [t[:3500] for t in batch]
        outs = bart_pipe(
            batch_trunc,
            max_length=BART_MAX_TARGET,
            min_length=BART_MIN_TARGET,
            truncation=True,
            no_repeat_ngram_size=3,
        )
        summaries.extend([o["summary_text"] for o in outs])
    return summaries

bart_summaries = bart_summarize_batch(articles)
print(f"  Done. Sample output:\n  {bart_summaries[0][:300]}…")

# ---------------------------------------------------------------------------
# 3.5-3.8 — Evaluation: ROUGE, BLEU, METEOR, BERTScore
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3.5-3.8  Computing metrics…")

scorer_rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)
meteor_metric = evaluate.load("meteor")
bleu_metric   = evaluate.load("sacrebleu")

def compute_all_metrics(summaries, refs, model_name):
    r1s, r2s, rls, bleus, meteors = [], [], [], [], []
    for hyp, ref in zip(summaries, refs):
        # ROUGE
        scores = scorer_rouge.score(ref, hyp)
        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)
        rls.append(scores["rougeL"].fmeasure)
        # BLEU (sacrebleu expects list of refs)
        bleu = bleu_metric.compute(
            predictions=[hyp], references=[[ref]]
        )["score"]
        bleus.append(bleu)
        # METEOR
        meteor = meteor_metric.compute(
            predictions=[hyp], references=[ref]
        )["meteor"]
        meteors.append(meteor)

    # BERTScore (batch)
    print(f"  Computing BERTScore for {model_name}…")
    P, R, F1 = bert_score_lib.score(
        summaries, refs,
        lang="en",
        device=device_str,
        verbose=False,
        batch_size=16,
    )

    return {
        "Model":      model_name,
        "ROUGE-1":    round(np.mean(r1s), 4),
        "ROUGE-2":    round(np.mean(r2s), 4),
        "ROUGE-L":    round(np.mean(rls), 4),
        "BLEU":       round(np.mean(bleus), 4),
        "METEOR":     round(np.mean(meteors), 4),
        "BERTScore-P": round(P.mean().item(), 4),
        "BERTScore-R": round(R.mean().item(), 4),
        "BERTScore-F1":round(F1.mean().item(), 4),
    }

lexrank_metrics = compute_all_metrics(lexrank_summaries, references, "LexRank")
bart_metrics    = compute_all_metrics(bart_summaries,    references, "BART-large-cnn")

# ---------------------------------------------------------------------------
# 3.9 — Save results
# ---------------------------------------------------------------------------
print("\n3.9  Results table…")
results = pd.DataFrame([lexrank_metrics, bart_metrics])
print(results.to_string(index=False))
results.to_csv("outputs/q3/q3_results.csv", index=False)
print("  Saved: outputs/q3/q3_results.csv")

# Bar chart comparison
metrics_to_plot = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR", "BERTScore-F1"]
x = np.arange(len(metrics_to_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, [lexrank_metrics[m] for m in metrics_to_plot], width, label="LexRank")
ax.bar(x + width/2, [bart_metrics[m]    for m in metrics_to_plot], width, label="BART-large-cnn")
ax.set_xticks(x); ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel("Score"); ax.set_title("Q3 Summarization Metrics Comparison")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/q3/q3_metrics_comparison.png", dpi=150)
plt.close()
print("  Saved: outputs/q3/q3_metrics_comparison.png")

# ---------------------------------------------------------------------------
# 3.10 — Qualitative examples (at least 3)
# ---------------------------------------------------------------------------
print("\n3.10  Qualitative examples…")

def quick_scores(hyp, ref):
    r = scorer_rouge.score(ref, hyp)
    _, _, f = bert_score_lib.score([hyp], [ref], lang="en", device=device_str, verbose=False)
    return round(r["rouge1"].fmeasure, 4), round(r["rougeL"].fmeasure, 4), round(f.mean().item(), 4)

qual_rows = []
for i in range(3):
    lr_r1, lr_rl, lr_bs = quick_scores(lexrank_summaries[i], references[i])
    ba_r1, ba_rl, ba_bs = quick_scores(bart_summaries[i],    references[i])
    qual_rows.append({
        "idx":               i,
        "article":           articles[i][:500],
        "reference":         references[i],
        "lexrank":           lexrank_summaries[i],
        "bart":              bart_summaries[i],
        "lexrank_rouge1":    lr_r1,
        "lexrank_rougeL":    lr_rl,
        "lexrank_bertscore": lr_bs,
        "bart_rouge1":       ba_r1,
        "bart_rougeL":       ba_rl,
        "bart_bertscore":    ba_bs,
    })

qual_df = pd.DataFrame(qual_rows)
qual_df.to_csv("outputs/q3/q3_qualitative_examples.csv", index=False)
print("  Saved: outputs/q3/q3_qualitative_examples.csv")

for i, row in qual_df.iterrows():
    print(f"\n  --- Example {i+1} ---")
    print(f"  REFERENCE : {row['reference'][:200]}")
    print(f"  LEXRANK   : {row['lexrank'][:200]}")
    print(f"             R1={row['lexrank_rouge1']} | RL={row['lexrank_rougeL']} | BS={row['lexrank_bertscore']}")
    print(f"  BART      : {row['bart'][:200]}")
    print(f"             R1={row['bart_rouge1']} | RL={row['bart_rougeL']} | BS={row['bart_bertscore']}")

# ---------------------------------------------------------------------------
# Per-example ROUGE to visualise distribution
# ---------------------------------------------------------------------------
rouge1_lr, rouge1_bart = [], []
for hyp_lr, hyp_bart, ref in zip(lexrank_summaries, bart_summaries, references):
    rouge1_lr.append(scorer_rouge.score(ref, hyp_lr)["rouge1"].fmeasure)
    rouge1_bart.append(scorer_rouge.score(ref, hyp_bart)["rouge1"].fmeasure)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(rouge1_lr,   bins=30, alpha=0.6, label="LexRank")
ax.hist(rouge1_bart, bins=30, alpha=0.6, label="BART-large-cnn")
ax.set_xlabel("ROUGE-1 F1"); ax.set_ylabel("Count")
ax.set_title("ROUGE-1 Distribution (500 test examples)")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/q3/q3_rouge1_distribution.png", dpi=150)
plt.close()
print("  Saved: outputs/q3/q3_rouge1_distribution.png")

print("\nPhase 3 complete. Results saved to outputs/q3/")
