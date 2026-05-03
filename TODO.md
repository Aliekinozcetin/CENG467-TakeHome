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
- [x] `1.14` Representation türlerini karşılaştıran kısa discussion yaz

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
- [x] `2.9` Contextual embeddings'in rolünü tartış

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
- [x] `3.11` Extractive vs abstractive trade-off discussion

---

## 📌 PHASE 4 — Q4: Machine Translation

### 4A — Dataset
- [x] `4.1` Multi30k dataset yükle (EN→DE) — train=29000, val=1014, test=1000
- [x] `4.2` Preprocessing: spaCy tokenization (en_core_web_sm / de_core_news_sm), lowercase
- [x] `4.3` Word-level vocabulary (min_freq=2) — EN ~5K, DE ~7K tokens

### 4B — Model Implementasyonları
- [x] `4.4` **Model 1: Seq2Seq + Bahdanau Attention**
  - Encoder: 1-layer BiLSTM, hidden=256 (CPU — LSTM MPS'den hızlı)
  - Decoder: 1-layer LSTM + Bahdanau Attention
  - Embedding dim: 256, Dropout: 0.3
  - Adam, lr=1e-3, gradient clipping=1.0, epoch=10
  - Teacher forcing ratio: 0.5
- [x] `4.5` **Model 2: Helsinki-NLP/opus-mt-en-de** (pretrained Transformer)
  - HuggingFace direkt inference, beam search (num_beams=5), max_length=128

### 4C — Evaluation
- [x] `4.6` BLEU hesapla (`sacrebleu`)
- [x] `4.7` METEOR hesapla (`nltk.translate.meteor_score`)
- [x] `4.8` ChrF hesapla (`sacrebleu`)
- [x] `4.9` BERTScore hesapla (`bert-base-multilingual-cased`, lang=de)
- [x] `4.10` Sonuçları tablo olarak kaydet → `outputs/q4/q4_results.csv`
- [x] `4.11` 5 qualitative örnek → `outputs/q4/q4_qualitative.csv` (idx: 0,100,200,400,700)
- [x] `4.12` Rare word (OOV rate) + length-bucket BLEU analizi → `outputs/q4/q4_length_analysis.csv`
- [x] `4.13` Her metriğin kalite boyutu: BLEU (precision), ChrF (morphology), METEOR (recall), BERTScore (semantic)
- [x] `4.14` Sonuçları analiz et ve raporda tartış

---

## 📌 PHASE 5 — Q5: Language Modeling

### 5A — Dataset
- [x] `5.1` WikiText-2 dataset yükle (wikitext-2-raw-v1)
- [x] `5.2` Tokenization: word-level, lowercase, `<unk>`, `<eos>`, `<pad>` tokenleri
- [x] `5.3` Vocabulary oluştur (min_freq=2)

### 5B — Model Implementasyonları
- [x] `5.4` **Model 1: Trigram LM (Kneser-Ney smoothing)**
  - defaultdict(Counter) bigram context tablosu
  - Interpolated Kneser-Ney smoothing (discount=0.75)
  - Val + Test perplexity, temperature sampling (0.8)
- [x] `5.5` **Model 2: LSTM Language Model**
  - 2-layer LSTM, hidden=512, embedding=512 (weight tying aktif)
  - Dropout=0.5, AdamW (lr=1e-3) + ReduceLROnPlateau, epoch=20, batch=32, bptt=35
  - device=mps/cpu (utils.device)

### 5C — Evaluation & Generation
- [x] `5.6` Val + Test perplexity hesapla → `outputs/q5/q5_results.csv`
- [x] `5.7` Her modelden 5 sample → `outputs/q5/q5_samples.csv` (temperature=0.8)
- [x] `5.8` Fluency/coherence karşılaştırması (print + rapor)
- [x] `5.9` Sonuçları analiz et ve raporda tartış

---

## 📝 PHASE 6 — LaTeX Rapor

- [x] `6.1` LaTeX proje yapısını oluştur (`report/main.tex`)
- [x] `6.2` Rapor bölümlerini yaz:
  - [x] Abstract
  - [x] Q1 bölümü (dataset, preprocessing, modeller, sonuçlar, analiz)
  - [x] Q2 bölümü
  - [x] Q3 bölümü
  - [x] Q4 bölümü
  - [x] Q5 bölümü
  - [x] Conclusion
- [x] `6.3` Tüm tabloları LaTeX'e aktar
- [x] `6.4` Confusion matrix ve training curve grafiklerini ekle
- [ ] `6.5` PDF derle ve kontrol et (Overleaf'te yapılacak)

---

## 📦 PHASE 7 — Submission

- [x] `7.1` Tüm kodları GitHub'a push et
- [ ] `7.2` Colab/notebook versiyonu hazırla (opsiyonel)
- [ ] `7.3` ZIP oluştur: `CENG467_Midterm_290201047.zip`
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
