# CENG 467 – Take-Home Midterm TODO
> Deadline: 30 Nisan 2026, 23:59 | Platform: MacBook M4 | Env: Python venv

---

## 🏗️ PHASE 0 — Project Setup
- [x] `0.1` GitHub repo oluştur: `CENG467-TakeHome`
- [x] `0.2` Python venv oluştur: `python3 -m venv .venv && source .venv/bin/activate`
- [x] `0.3` `requirements.txt` yaz ve `pip install -r requirements.txt`
  - torch (MPS destekli), transformers, datasets, scikit-learn, nltk, sumy
  - rouge-score, bert-score, sacrebleu, evaluate, sacremoses
  - networkx, numpy, pandas, matplotlib, seaborn, tqdm
- [x] `0.4` Klasör yapısını oluştur:
  ```
  CENG467-TakeHome/
  ├── q1_text_classification.py
  ├── q2_ner.py
  ├── q3_summarization.py
  ├── q4_translation.py
  ├── q5_language_modeling.py
  ├── requirements.txt
  ├── report/          ← LaTeX dosyaları
  │   └── main.tex
  ├── outputs/         ← Tablo, grafik, model çıktıları
  ├── TODO.md
  └── CHANGELOG.md
  ```
- [x] `0.5` MPS device kontrolü yaz (tüm modellerde ortak kullanılacak):
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```
- [x] `0.6` Global random seed sabitle: `SEED = 42` (torch, numpy, random, transformers)

---

## 📌 PHASE 1 — Q1: Text Classification (Representation Learning)

### 1A — Dataset & Preprocessing
- [x] `1.1` IMDb dataset yükle (`datasets` kütüphanesi ile HuggingFace'den)
- [x] `1.2` Train/Val/Test split yap (45K train / 2.5K val / 2.5K test) — test seti kilitli
- [x] `1.3` Preprocessing pipeline yaz:
  - Lowercase normalization
  - HTML tag temizleme (`<br />` vs.)
  - Noktalama & özel karakter temizliği
  - Stopword removal (NLTK)
- [x] `1.4` İki tokenization stratejisini karşılaştır:
  - **Strateji A:** Word-level tokenization (basit split + vocab)
  - **Strateji B:** WordPiece tokenization (BERT tokenizer)
- [x] `1.5` Preprocessing kararlarının etkisini tablo halinde dokümante et.

### 1B — Model Implementasyonları
- [x] `1.6` **Model 1: TF-IDF + Logistic Regression** — Test Acc: 0.9132, F1: 0.9130
  - `TfidfVectorizer(max_features=50000, ngram_range=(1,2))`
  - `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`
- [x] `1.7` **Model 2: TF-IDF + SVM** — Test Acc: 0.9140, F1: 0.9139
  - `LinearSVC(C=1.0, random_state=42)`
- [x] `1.8` **Model 3: BiLSTM** — Test Acc: 0.8988, F1: 0.8985
  - GloVe 100d: 29552/30000 kelime eşleşti
  - Epoch 4'te peak val_acc: 0.9116, sonra overfitting
- [x] `1.9` **Model 4: BERT-base-uncased fine-tuning** — Test Acc: 0.8964, F1: 0.8961
  - max_length=128 (MPS memory için), AdamW, lr=2e-5, batch=16, epoch=3
  - Best checkpoint: epoch 2 (val_acc=0.8968)

### 1C — Evaluation & Analysis
- [x] `1.10` Test seti üzerinde Accuracy ve Macro-F1 hesapla (tüm modeller)
- [x] `1.11` Sonuçları karşılaştırma tablosuna aktar (`outputs/q1_results.csv`)
- [x] `1.12` Confusion matrix çiz (her model için) → `outputs/q1_cm_*.png`
- [x] `1.13` **En az 5 yanlış sınıflandırılmış örnek analizi** — 40 örnek kaydedildi → `outputs/q1_misclassified.csv`
- [ ] `1.14` Representation türlerini karşılaştıran kısa discussion yaz *(sonuçlar çıktı, rapor yazılırken tamamlanacak)*

---

## 📌 PHASE 2 — Q2: Named Entity Recognition

### 2A — Dataset & BIO Tagging
- [x] `2.1` CoNLL-2003 dataset yükle — train=14041, val=3250, test=3453
- [x] `2.2` BIO tagging yapısını dokümante et (PER, ORG, LOC, MISC) → `outputs/q2_tag_distribution.csv`
- [x] `2.3` Token-label alignment kontrolü yap — ilk subword→gerçek label, sonraki→-100

### 2B — Model Implementasyonları
- [x] `2.4` **Model 1: BiLSTM-CRF** — Test P:0.8732 R:0.8507 F1:0.8618
  - CharCNN + GloVe 100d (18415/21011 eşleşti), hidden=256, 2-layer, CRF
  - Adam, lr=1e-3, batch=32, epoch=15 — peak val_F1: 0.9222 (epoch 15)
- [x] `2.5` **Model 2: BERT-base-cased** — Test F1: TBD *(eğitim devam etti, sonuç bekleniyor)*
  - AdamW, lr=3e-5, batch=16, epoch=5, MPS — epoch 2'de checkpoint (val_F1=0.9398)

### 2C — Evaluation & Analysis
- [x] `2.6` Precision, Recall, F1 hesapla — entity düzeyinde (`seqeval`)
- [x] `2.7` Sonuçları tablo olarak kaydet → `outputs/q2_results.csv`, `outputs/q2_per_entity_results.csv`
- [x] `2.8` Hata analizi → `outputs/q2_error_analysis.csv`
  - Boundary error örnekleri (B- vs I- karışıklığı)
  - Entity tipi karışıklığı (PER vs ORG gibi)
- [ ] `2.9` Contextual embeddings'in rolünü tartış *(rapor yazılırken tamamlanacak)*

---

## 📌 PHASE 3 — Q3: Text Summarization

### 3A — Dataset
- [x] `3.1` CNN/DailyMail dataset yükle — 1000 test örneği (evaluation only)
- [x] `3.2` Dataset yapısını dokümante et — fields: article, highlights, id; avg article 1395 words

### 3B — Model Implementasyonları
- [x] `3.3` **Model 1: LexRank (Extractive)** — sumy, 3 cümle, ~17sn/1000 örnek
- [x] `3.4` **Model 2: BART-large-cnn (Abstractive)** — facebook/bart-large-cnn, direkt inference, MPS

### 3C — Evaluation
- [x] `3.5` ROUGE-1, ROUGE-2, ROUGE-L hesapla
- [x] `3.6` BLEU hesapla
- [x] `3.7` METEOR hesapla
- [x] `3.8` BERTScore hesapla (roberta-large)
- [x] `3.9` Sonuçları kaydet → `outputs/q3/q3_results.csv`
- [x] `3.10` 3 qualitative örnek → `outputs/q3/q3_qualitative_examples.csv`
- [ ] `3.11` Extractive vs abstractive trade-off discussion *(rapor yazılırken tamamlanacak)*

---

## 📌 PHASE 4 — Q4: Machine Translation

### 4A — Dataset
- [ ] `4.1` Multi30k dataset yükle (EN→DE, train/val/test)
- [ ] `4.2` Preprocessing: lowercase, tokenize (spaCy en/de modelleri), BPE vocab oluştur
- [ ] `4.3` Vocabulary boyutu belirle (max 8K-10K)

### 4B — Model Implementasyonları
- [ ] `4.4` **Model 1: Seq2Seq + Bahdanau Attention**
  - Encoder: 2-layer LSTM, hidden=512
  - Decoder: 1-layer LSTM + attention
  - Embedding dim: 256, Dropout: 0.3
  - Adam, lr=1e-3, gradient clipping=1.0, epoch=20
  - Teacher forcing ratio: 0.5
- [ ] `4.5` **Model 2: Helsinki-NLP/opus-mt-en-de** (pretrained Transformer)
  - HuggingFace'den direkt inference (fine-tune opsiyonel)
  - Beam search decoding (num_beams=5)

### 4C — Evaluation
- [ ] `4.6` BLEU hesapla (`sacrebleu`)
- [ ] `4.7` METEOR hesapla
- [ ] `4.8` ChrF hesapla (`sacrebleu`)
- [ ] `4.9` BERTScore hesapla
- [ ] `4.10` Sonuçları tablo olarak kaydet (`outputs/q4_results.csv`)
- [ ] `4.11` **En az 1 qualitative örnek**: source + reference + her iki model çıktısı
- [ ] `4.12` Rare word handling ve long-range dependency analizi yaz
- [ ] `4.13` Her metriğin farklı kalite boyutunu nasıl yansıttığını tartış

---

## 📌 PHASE 5 — Q5: Language Modeling

### 5A — Dataset
- [ ] `5.1` WikiText-2 dataset yükle (Penn Treebank'a göre daha temiz ve modern)
- [ ] `5.2` Tokenization: word-level, `<unk>` ve `<eos>` tokenleri ekle
- [ ] `5.3` Vocabulary oluştur (min_freq=2)

### 5B — Model Implementasyonları
- [ ] `5.4` **Model 1: N-gram Language Model (Trigram)**
  - NLTK ile trigram modeli
  - Laplace smoothing
  - Perplexity hesabı
- [ ] `5.5` **Model 2: LSTM Language Model**
  - 2-layer LSTM, hidden=512, embedding=256
  - Dropout: 0.5, weight tying (embedding = output layer)
  - SGD + lr scheduling, epoch=30, batch=32, bptt=35
  - MPS device

### 5C — Evaluation & Generation
- [ ] `5.6` Test seti perplexity hesapla (her iki model)
- [ ] `5.7` Her modelden **5 adet kısa text sample** üret (temperature sampling)
- [ ] `5.8` Fluency ve coherence karşılaştırmalı analizi yaz
- [ ] `5.9` Sonuçları tablo olarak kaydet (`outputs/q5_results.csv`)

---

## 📝 PHASE 6 — LaTeX Rapor

- [ ] `6.1` LaTeX proje yapısını oluştur (`report/main.tex`)
- [ ] `6.2` Rapor bölümlerini yaz:
  - [ ] Abstract
  - [ ] Q1 bölümü (dataset, preprocessing, modeller, sonuçlar, analiz)
  - [ ] Q2 bölümü
  - [ ] Q3 bölümü
  - [ ] Q4 bölümü
  - [ ] Q5 bölümü
  - [ ] Conclusion
  - [ ] References (BibTeX)
- [ ] `6.3` Tüm tabloları LaTeX'e aktar
- [ ] `6.4` Confusion matrix ve training curve grafiklerini ekle
- [ ] `6.5` PDF derle ve kontrol et

---

## 📦 PHASE 7 — Submission

- [ ] `7.1` Tüm kodları GitHub'a push et (README ile)
- [ ] `7.2` Colab/notebook versiyonu hazırla (opsiyonel ama iyi görünür)
- [ ] `7.3` ZIP oluştur: `CENG467_Midterm_<StudentID>.zip`
  - Kaynak kodlar, LaTeX + PDF rapor, outputs/ klasörü
- [ ] `7.4` Email gönder: aytugonan@iyte.edu.tr
- [ ] `7.5` Microsoft Teams'e yükle
- [ ] `7.6` Son kontrol: reproducibility, seed, split tutarlılığı

---

## ⏱️ Tahmini Süre Tablosu

| Faz | Görev | Tahmini Süre |
|-----|-------|-------------|
| 0 | Setup | 30 dk |
| 1 | Q1 Text Classification | 5-6 saat (BERT eğitimi ~2-3 saat) |
| 2 | Q2 NER | 4-5 saat |
| 3 | Q3 Summarization | 4-5 saat |
| 4 | Q4 Translation | 5-6 saat (Seq2Seq eğitimi ~3 saat) |
| 5 | Q5 Language Modeling | 3-4 saat |
| 6 | LaTeX Rapor | 4-5 saat |
| 7 | Submission | 30 dk |
| **Toplam** | | **~26-32 saat** |
