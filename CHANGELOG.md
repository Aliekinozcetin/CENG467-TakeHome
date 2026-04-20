# CHANGELOG — CENG 467 Take-Home Midterm

All notable changes and progress will be documented here.
Format: `[YYYY-MM-DD] | Phase | Description`

---

## [2026-04-19] — Düzeltme: Q3 Dataset Kararı
### Changed
- Q3 CNN/DailyMail subset: "5K train + 500 test" → "1000 test only"
- Gerekçe: LexRank unsupervised, BART (bart-large-cnn) pretrained — eğitim yapılmıyor
- `q3_summarization.py`: train_data referansları kaldırıldı, sadece test_data kullanılıyor

---

## Unreleased

### Planned
- Q1: TF-IDF, BiLSTM, BERT-base on IMDb
- Q2: BiLSTM-CRF, BERT-base-cased on CoNLL-2003
- Q3: LexRank, BART-base on CNN/DailyMail subset
- Q4: Seq2Seq+Attention, Helsinki-NLP/opus-mt-en-de on Multi30k
- Q5: Trigram LM, LSTM LM on WikiText-2

---

## [2026-04-18] — Phase 0: Planning & Decisions

### Added
- `TODO.md` oluşturuldu — tüm görevler ve faz planı
- `CHANGELOG.md` oluşturuldu — ilerleme takibi için

### Decisions Made
| Karar | Seçim | Gerekçe |
|-------|-------|---------|
| Platform | MacBook M4, Python venv | Lokal çalışma, MPS GPU desteği |
| Q1 Dataset | IMDb (full, 50K) | Geniş, iyi bilinen benchmark |
| Q1 Models | TF-IDF+LR, TF-IDF+SVM, BiLSTM, BERT-base-uncased | Sparse→Dense→Contextual spektrumu |
| Q1 Subset | Full dataset | Raporlama kalitesi için |
| Q2 Dataset | CoNLL-2003 | Zorunlu |
| Q2 Models | BiLSTM-CRF, BERT-base-cased | Cased önemli NER için |
| Q3 Dataset | CNN/DailyMail (5K subset) | Bellek kısıtı |
| Q3 Extractive | LexRank (sumy) | TextRank'e göre daha sağlam istatistiksel temel |
| Q3 Abstractive | BART-base (facebook/bart-large-cnn inference) | Dengeli model boyutu |
| Q4 Dataset | Multi30k (EN→DE) | Zorunlu |
| Q4 Models | Seq2Seq+Bahdanau, Helsinki-NLP/opus-mt-en-de | Sıfırdan vs pretrained karşılaştırması |
| Q5 Dataset | WikiText-2 | Penn Treebank'a göre daha temiz |
| Q5 Models | Trigram+Laplace, LSTM (2-layer) | Classical vs Neural karşılaştırması |
| Transformer size | BERT-base / BART-base | Dengeli (hız vs performans) |
| Device | MPS (Apple Silicon M4) | ~3-5x CPU'dan hızlı |
| Random Seed | 42 | Tüm deneylerde sabit |
| Code Style | Her soru ayrı `.py` dosyası | Sade ve reproducible |

---

<!-- Buradan itibaren ilerleme kaydederken doldur -->

## [2026-04-18] — Phase 0: Project Setup
### Added
- [x] GitHub repo (klasör yapısı kuruldu, 0.1 skip)
- [x] venv kurulumu: `python3 -m venv .venv` (Python 3.9.6)
- [x] requirements.txt oluşturuldu ve kuruldu
- [x] Klasör yapısı: `report/`, `outputs/`, placeholder `.py` dosyaları
- [x] `utils.py` — MPS device helper (`device = mps`)
- [x] `utils.py` — Global seed: `set_seed(42)` (torch, numpy, random, transformers)

### Package Versions
| Package | Version |
|---------|---------|
| torch | 2.8.0 |
| transformers | 4.57.6 |
| datasets | 4.5.0 |
| scikit-learn | 1.6.1 |
| nltk | 3.9.2 |
| sumy | 0.12.0 |
| rouge-score | 0.1.2 |
| bert-score | 0.3.13 |
| sacrebleu | 2.6.0 |
| evaluate | 0.4.6 |
| sacremoses | 0.1.1 |
| seqeval | 1.2.2 |
| networkx | 3.2.1 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| tqdm | 4.67.3 |
| spacy | 3.7.5 |

---

## [2026-04-18] — Phase 1: Q1 Text Classification
### Added
- [x] `q1_text_classification.py` implement edildi (tüm modeller tek dosyada)
- [x] IMDb dataset yükleme: 45K train / 2.5K val / 2.5K test split
- [x] Preprocessing pipeline: lowercase, HTML temizlik, noktalama, stopword removal
- [x] Tokenization karşılaştırması: Word-level vs WordPiece (BERT) → `outputs/q1_tokenization_comparison.csv`
- [x] TF-IDF + Logistic Regression (max_features=50000, ngram=(1,2), C=1.0)
- [x] TF-IDF + SVM (LinearSVC, C=1.0)
- [x] BiLSTM + GloVe (embed=100, hidden=256, 2-layer, dropout=0.3, epoch=10, batch=64)
- [x] BERT-base-uncased fine-tuning (max_len=128, lr=2e-5, batch=16, epoch=3, MPS)
- [x] Confusion matrix (4 model) → `outputs/q1_cm_*.png`
- [x] Training curves → `outputs/q1_bilstm_training_curve.png`, `outputs/q1_bert_training_curve.png`
- [x] Misclassified examples → `outputs/q1_misclassified.csv`
- [x] Results table → `outputs/q1_results.csv`

### Notes
- BERT max_length 512→128 (MPS memory optimizasyonu)
- GloVe bulunamazsa random init kullanılır (script bunu handle ediyor)
- BiLSTM vocab: 30K kelime, max_seq_len=256

### Results
| Model | Val Acc | Val F1 | Test Acc | Test F1 |
|-------|---------|--------|----------|---------|
| TF-IDF + LR  | 0.9120 | 0.9120 | 0.9132 | 0.9130 |
| TF-IDF + SVM | 0.9068 | 0.9068 | 0.9140 | 0.9139 |
| BiLSTM+GloVe | 0.8980 | 0.8980 | 0.8988 | 0.8985 |
| BERT-base    | 0.8968 | 0.8968 | 0.8964 | 0.8961 |

### Observations
- TF-IDF modelleri BERT ve BiLSTM'i geride bıraktı — max_length=128 BERT'i kısıtladı (IMDb reviews çok uzun)
- BiLSTM epoch 4'te peak (0.9116), sonra overfitting başladı
- GloVe: 29552/30000 (%98.5) kelime eşleşti
- Preprocessing %49 token azaltması sağladı (231→118 avg token)
- HTML tag içeren örnek oranı: Train %58, Val/Test %58

---

## [2026-04-19] — Phase 2: Q2 Named Entity Recognition
### Added
- [x] `q2_ner.py` implement edildi
- [x] CoNLL-2003 disk cache üzerinden yüklendi (`/tmp/conll2003_cached`) — datasets>=4.0 script desteği kaldırdığı için
- [x] BIO tagging yapısı dokümante edildi (9 label: O + 4×BIO) → `outputs/q2_tag_distribution.csv`
- [x] Token-label alignment: ilk subword→gerçek label, sonraki subword→-100
- [x] BiLSTM-CRF: CharCNN + GloVe 100d (18415/21011 eşleşti), hidden=256, CRF layer (manuel)
- [x] BERT-base-cased token classification, AdamW lr=3e-5, batch=16, epoch=5, MPS
- [x] Per-entity results → `outputs/q2_per_entity_results.csv`
- [x] Error analysis → `outputs/q2_error_analysis.csv`
- [x] Training curves → `outputs/q2_bilstm_crf_curve.png`, `outputs/q2_bert_curve.png`

### Bug Fixed
- `trust_remote_code` datasets>=4.0'da kaldırıldı → datasets==2.18.0 ile cache'e alındı, sonra load_from_disk ile yüklendi

### Results
| Model | Val P | Val R | Val F1 | Test P | Test R | Test F1 |
|-------|-------|-------|--------|--------|--------|---------|
| BiLSTM-CRF      | 0.9279 | 0.9165 | 0.9222 | 0.8732 | 0.8507 | 0.8618 |
| BERT-base-cased | TBD    | TBD    | TBD    | TBD    | TBD    | TBD    |

### Observations
- BiLSTM-CRF: val_F1 steady artış (0.8169→0.9222), overfitting yok — CRF regularization etkisi
- BiLSTM-CRF: val/test gap büyük (0.9222 vs 0.8618) — test seti OOV kelimeleri daha zor
- BERT epoch 2'de en iyi checkpoint (val_F1=0.9398) — contextual embeddings NER'i belirgin güçlendiriyor
- Tag dağılımı çok dengesiz: O=169K vs I-MISC=1155 — nadir entity tipleri zorlu

---

## [2026-04-19] — Phase 3: Q3 Text Summarization
### Added
- [x] `q3_summarization.py` implement edildi
- [x] CNN/DailyMail 1000 test örneği (evaluation only, eğitim yok)
- [x] LexRank — sumy, 3 cümle, TF-IDF cosine similarity, ~17sn/1000 örnek
- [x] BART-large-cnn — facebook/bart-large-cnn direkt inference, MPS, ~70dk/1000 örnek
- [x] ROUGE, BLEU, METEOR, BERTScore (roberta-large) hesaplandı
- [x] 3 qualitative örnek → `outputs/q3/q3_qualitative_examples.csv`
- [x] Metrik karşılaştırma grafiği → `outputs/q3/q3_metrics_comparison.png`
- [x] ROUGE-1 dağılım histogramı → `outputs/q3/q3_rouge1_distribution.png`

### Results
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | BERTScore-F1 |
|-------|---------|---------|---------|------|--------|--------------|
| LexRank       | 0.3539 | 0.1319 | 0.2219 | 7.8562  | 0.3201 | 0.8629 |
| BART-large-cnn| 0.4405 | 0.2127 | 0.3103 | 14.7468 | 0.3912 | 0.8806 |

### Observations
- BART tüm metriklerde LexRank'i belirgin geride bıraktı
- ROUGE-1 gap: 0.44 vs 0.35 — BART %25 daha iyi unigram örtüşmesi
- BLEU gap en büyük: 14.75 vs 7.86 — BART referansa yakın n-gram kullanıyor
- BERTScore farkı küçük (0.8806 vs 0.8629) — semantic içerik her ikisinde de yakalanıyor
- LexRank hız avantajı: 17sn vs 70dk — 240x daha hızlı
- Qualitative: BART daha akıcı ve öz, LexRank orijinal cümleleri koruyor (faithfulness yüksek)

---

## [2026-04-20] — Phase 4: Q4 Machine Translation
### Added
- [x] Multi30k yükleme (bentrevett/multi30k) — train=29000, val=1014, test=1000
- [x] spaCy tokenization (en_core_web_sm + de_core_news_sm), lowercase
- [x] Word-level vocabulary (min_freq=2): EN ~5K, DE ~7K token
- [x] Seq2Seq + Bahdanau Attention — 1-layer BiLSTM encoder, 1-layer LSTM decoder
  - CPU eğitim (LSTM MPS'den hızlı), epoch=10, batch=256, Adam lr=1e-3, clip=1.0
  - Best checkpoint → outputs/q4/seq2seq_best.pt (gitignore'a eklendi)
- [x] Helsinki-NLP/opus-mt-en-de — direkt inference, beam_search num_beams=5
- [x] BLEU, ChrF, METEOR, BERTScore (bert-base-multilingual-cased, lang=de)
- [x] 5 qualitative örnek → outputs/q4/q4_qualitative.csv
- [x] Length-bucket BLEU + OOV analizi → outputs/q4/q4_length_analysis.csv
- [x] Training curve → outputs/q4/q4_seq2seq_curve.png
- [x] Metrics chart → outputs/q4/q4_metrics_comparison.png

### Hyperparameters (actual vs TODO)
| Param | TODO | Kod | Gerekçe |
|-------|------|-----|---------|
| N_EPOCHS | 20 | 10 | CPU'da makul süre, best-ckpt ile yeterli |
| HIDDEN_SIZE | 512 | 256 | Bellek/hız optimizasyonu |
| N_LAYERS | 2 | 1 | CPU performansı |
| Device | MPS | CPU | LSTM CPU'da daha hızlı |

### Results
| Model | BLEU | METEOR | ChrF | BERTScore-F1 |
|-------|------|--------|------|-------------|
| Seq2Seq+Bahdanau | 3.5128 | 0.4569 | 38.9939 | 0.7415 |
| Helsinki-NLP/opus-mt-en-de | 36.3710 | 0.5924 | 64.2252 | 0.8972 |

### Observations
- Helsinki tüm metriklerde Seq2Seq'i belirgin geride bıraktı (BLEU: 36.4 vs 3.5)
- Seq2Seq repetition sorunu: greedy decode + no_repeat_ngram_size eksikliği ("etwas etwas etwas…")
- Seq2Seq ChrF=38.99 ve METEOR=0.457: morfoloji öğrenilmiş, sıra/tekrar hatası var
- Best checkpoint epoch 7 (val_ppl=20.8), epoch 8+ hafif overfitting
- Test OOV rate: %1.71 — düşük, vocab yeterli
- Length-bucket: kısa cümlede Seq2Seq BLEU=9.47, uzunda 5.04 — uzun bağımlılık zayıf
- Helsinki BERTScore=0.8972: semantik içerik neredeyse referans kalitesinde

---

## [2026-04-20] — Phase 5: Q5 Language Modeling
### Added
- [x] WikiText-2 yükleme (wikitext-2-raw-v1), word-level tokenization
- [x] Vocab: min_freq=2, <pad>/<unk>/<eos> özel tokenler
- [x] Trigram LM: defaultdict(Counter) bigram context, Laplace smoothing
- [x] LSTM LM: 2-layer, hidden=512, embed=256, dropout=0.5, SGD+LR-decay, bptt=35
- [x] Val/Test perplexity → outputs/q5/q5_results.csv
- [x] 5 text sample her model → outputs/q5/q5_samples.csv (temperature=0.8)
- [x] Training curve + LR schedule → outputs/q5/q5_lstm_curve.png
- [x] Perplexity bar chart → outputs/q5/q5_perplexity_comparison.png
- [x] outputs/q5/lstm_best.pt gitignore'a eklendi

### Results
| Model | Val PPL | Test PPL |
|-------|---------|---------|
| Trigram (Laplace) | 15728.80 | 15295.68 |
| LSTM LM (2-layer) | 341.26 | 326.52 |

### Observations
- Trigram yüksek PPL beklenen: Laplace smoothing + 39K vocab = agresif ceza (sparsity problemi)
- LSTM 30 epoch boyunca sürekli iyileşti, LR decay hiç tetiklenmedi — daha fazla epoch ile düşer
- LSTM qualitative: anlamlı İngilizce yapılar üretiyor; Trigram tamamen incoherent
- Literatür hedefi: WikiText-2 2-layer LSTM ~130 PPL; bizim 326.52 epoch sınırlılığından
- Weight tying aktif (embed=hidden=512): parameter verimliliği sağlandı

---

## [YYYY-MM-DD] — Phase 6: LaTeX Report
### Added
- [ ] main.tex oluşturuldu
- [ ] Tüm bölümler yazıldı
- [ ] Tablolar eklendi
- [ ] Grafikler eklendi
- [ ] PDF derlendi

---

## [YYYY-MM-DD] — Phase 7: Submission
### Done
- [ ] GitHub push
- [ ] ZIP oluşturuldu
- [ ] Email gönderildi
- [ ] Teams'e yüklendi
