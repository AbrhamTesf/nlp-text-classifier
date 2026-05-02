# 🧠 NLP Text Classifier — Multi-Label News Classification

> Fine-tuned DistilBERT vs. TF-IDF baseline for multi-label text classification, served via FastAPI and tracked with Weights & Biases.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

---

## 📌 Project Overview

This project builds an end-to-end multi-label text classification pipeline that assigns one or more topic labels (e.g., *Politics*, *Sports*, *Technology*) to a given piece of text.

The goal is to demonstrate:
- A full ML pipeline from raw data to deployed API
- The performance gap between a classical baseline (TF-IDF + Logistic Regression) and a fine-tuned transformer (DistilBERT)
- Clean, reproducible, production-style code

**Dataset:** [AG News](https://huggingface.co/datasets/ag_news) — 120,000 news articles across 4 categories.

---

## 🏗️ Project Structure

```
nlp-text-classifier/
├── data/                   # Raw and processed datasets
├── notebooks/              # Exploratory data analysis & experiments
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_transformer_finetuning.ipynb
├── src/                    # Source code (reusable modules)
│   ├── __init__.py
│   ├── preprocess.py       # Text cleaning & tokenization
│   ├── train_baseline.py   # TF-IDF + Logistic Regression
│   ├── train_transformer.py# DistilBERT fine-tuning
│   └── evaluate.py         # Metrics & evaluation utilities
├── api/                    # FastAPI serving layer
│   └── main.py
├── models/                 # Saved model checkpoints (gitignored)
├── requirements.txt
└── README.md
```

---

## 🔬 Approach

### Step 1 — Baseline: TF-IDF + Logistic Regression
A fast classical baseline using bag-of-words features. Serves as the benchmark to beat.

### Step 2 — Transformer: Fine-tuned DistilBERT
Fine-tuning `distilbert-base-uncased` on the AG News dataset using HuggingFace `transformers`. DistilBERT is 40% smaller and 60% faster than BERT while retaining ~97% of its performance.

### Step 3 — Serving
The best model is wrapped in a **FastAPI** REST endpoint that accepts raw text and returns predicted labels + confidence scores.

### Step 4 — Experiment Tracking
All training runs are logged to **Weights & Biases** — including loss curves, accuracy, and hyperparameters — for full reproducibility.

---

## 📊 Results

| Model | Accuracy | F1 Score (Macro) | Training Time |
|---|---|---|---|
| TF-IDF + Logistic Regression | — | — | — |
| DistilBERT (fine-tuned) | — | — | — |

> Results will be updated as experiments are completed.

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/AbrhamTesf/nlp-text-classifier.git
cd nlp-text-classifier
```

### 2. Set up environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the baseline
```bash
python src/train_baseline.py
```

### 4. Fine-tune DistilBERT
```bash
python src/train_transformer.py
```

### 5. Start the API
```bash
uvicorn api.main:app --reload
```
Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 🌐 API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "The president signed a new trade agreement with the EU today."
}
```

**Response:**
```json
{
  "label": "Politics",
  "confidence": 0.94,
  "all_scores": {
    "Politics": 0.94,
    "Business": 0.04,
    "Sports": 0.01,
    "Technology": 0.01
  }
}
```

---

## 📈 Experiment Tracking

Training metrics are tracked with [Weights & Biases](https://wandb.ai).

To run with tracking:
```bash
wandb login
python src/train_transformer.py --track
```

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch + HuggingFace Transformers |
| Classical ML | scikit-learn |
| Data | HuggingFace Datasets |
| API | FastAPI + Uvicorn |
| Experiment Tracking | Weights & Biases |
| Notebooks | Jupyter |

---

## 📚 Key Learnings

> This section will be filled in as the project progresses.

- Why transformer fine-tuning outperforms bag-of-words approaches
- Trade-offs between model size, speed, and accuracy
- How to structure an NLP project for production

---

## 🗺️ Roadmap

- [x] Project setup & structure
- [ ] Exploratory data analysis
- [ ] Baseline model (TF-IDF)
- [ ] DistilBERT fine-tuning
- [ ] FastAPI serving
- [ ] Weights & Biases integration
- [ ] Final results & write-up

---

## 👤 Author

**Abrham T. Tadesse**
[GitHub](https://github.com/AbrhamTesf) 


---

## 📄 License

MIT License — feel free to use this as a reference for your own projects.
