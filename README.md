<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Framework-Transformers-FFD21E?logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/AI%20Detection-v1.0-brightgreen">
</p>

# AI vs Human Text Classifier

A binary classification system that distinguishes **AI-generated text** from **human-written text** using a combination of traditional machine learning models and deep learning approaches. The project includes a comprehensive Jupyter Notebook for model training/evaluation and an interactive **Streamlit web app** for real-time inference and PDF report generation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Notebook (Training & Evaluation)](#notebook-training--evaluation)
- [Streamlit App](#streamlit-app)
- [Results & Findings](#results--findings)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [References & Citations](#references--citations)

---

## Project Overview

This project tackles the problem of detecting AI-generated text — a growing concern in academic integrity, content moderation, and digital forensics. We train and evaluate **10 model variants** (5 model architectures × 2 feature types each, plus LSTM and fine-tuned DistilBERT) on a labeled dataset of AI vs. human-written samples.

**Key objectives:**
1. Engineer meaningful features from raw text (TF-IDF and BERT embeddings)
2. Train and tune traditional ML models (SVM, Decision Tree, AdaBoost, MLP)
3. Train deep learning models (LSTM, fine-tuned DistilBERT)
4. Compare performance across all models using accuracy, precision, recall, F1, and ROC-AUC
5. Deploy the best models via an interactive Streamlit web interface

---

## Dataset

The dataset consists of **8,176 text samples** (4,088 human-written, 4,088 AI-generated), sourced from the course Canvas page. Each sample has:

| Column | Description |
|--------|-------------|
| `text` | The raw text content |
| `label` | 0 = Human-written, 1 = AI-generated |

The dataset is **perfectly balanced**, with an 80/20 stratified train-test split used throughout all experiments.

---

## Feature Engineering

Two complementary feature extraction approaches were used:

### 1. TF-IDF (Term Frequency–Inverse Document Frequency)

- **Library:** `sklearn.feature_extraction.text.TfidfVectorizer`
- **Configuration:** `max_features=10000`, `ngram_range=(1, 2)`, `stop_words="english"`
- **How it works:** Each document is represented as a sparse vector where each dimension corresponds to a word or two-word phrase, weighted by its importance across the corpus. Common words like "the" are down-weighted; distinctive words get higher scores.
- **Advantages:** Fast, interpretable, works well with linear models.

### 2. BERT Embeddings

- **Model:** `bert-base-uncased` (12-layer, 768-hidden, 110M parameters) — [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
- **Method:** Mean pooling of the last hidden states, producing a **768-dimensional dense vector** per document.
- **How it works:** Unlike TF-IDF, BERT captures **contextual meaning** — the same word ("bank") gets different representations depending on context ("river bank" vs. "bank account").
- **Advantages:** Captures semantics, handles synonyms, superior for deep learning models.

> Both feature sets were saved: TF-IDF vectorizer → `models/tfidf_vectorizer.pkl`, BERT embeddings computed on-the-fly during training and cached during inference.

---

## Models

### Traditional Machine Learning Models (trained on both TF-IDF & BERT features)

| Model | Description | Hyperparameter Tuning |
|-------|-------------|----------------------|
| **SVM (LinearSVC)** | Finds the optimal hyperplane separating the two classes. Good for high-dimensional text data. | `C`: [0.1, 1, 10, 100], `tol`: [1e-3, 1e-4, 1e-5] via `GridSearchCV` |
| **Decision Tree** | Learns a series of yes/no questions. Highly interpretable — can visualize the tree structure. | `max_depth=20`, `min_samples_split=5` |
| **AdaBoost** | Ensemble of weak stumps (decision trees of depth 1) that iteratively focuses on hard-to-classify samples. | `n_estimators=100`, base estimator: `DecisionTreeClassifier(max_depth=1)` |
| **MLP (Feedforward Neural Network)** | Simple multi-layer perceptron with 1–2 hidden layers. A bridge from ML to DL. | `hidden_layer_sizes`: [(128,), (128, 64)], `alpha`: [1e-4, 1e-3], `learning_rate_init`: [1e-3, 1e-4] via `GridSearchCV` |

### Deep Learning Models

| Model | Description | Architecture |
|-------|-------------|-------------|
| **LSTM** | Long Short-Term Memory network that reads text sequentially, capturing word order and long-range dependencies. [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) | Embedding(20K, 128) → LSTM(64) → Dropout(0.2) → Linear(1). Trained with BCEWithLogitsLoss, early stopping (patience=4). |
| **DistilBERT** | A distilled, smaller, faster version of BERT that retains 97% of BERT's language understanding while being 40% smaller and 60% faster. [Sanh et al., 2019](https://arxiv.org/abs/1910.01108) | Fine-tuned `distilbert-base-uncased` for binary classification. `learning_rate=2e-5`, `batch_size=8`, `epochs=2`, AdamW optimizer. |

---

## Notebook (Training & Evaluation)

The Jupyter Notebook (`notebooks/project1_notebook.ipynb`) covers the complete ML workflow:

1. **Data Loading & Investigation** — Loads the Excel dataset, checks for missing values, examines class balance.
2. **Train/Test Split** — 80/20 stratified split (6,540 train, 1,636 test).
3. **Feature Engineering**
   - TF-IDF vectorization with bigrams
   - BERT embedding extraction with mean pooling
4. **Model Training**
   - 4 ML models × 2 feature sets = 8 trained models
   - LSTM with PyTorch (custom vocab, embedding layer, early stopping)
   - DistilBERT fine-tuning using Hugging Face `Trainer` API
5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
   - ROC Curve & AUC for every model
   - Combined ROC comparison plot
6. **Model Saving** — All models saved as `.pkl` or `.h5` files in `models/`

### Results Summary

```
=== Model Performance Summary ===
SVM (TF-IDF):         AUC = 0.9852
SVM (BERT):           AUC = 0.9480
Decision Tree (TF):   AUC = 0.8247
Decision Tree (BERT): AUC = 0.8351
AdaBoost (TF):        AUC = 0.9733
AdaBoost (BERT):      AUC = 0.9866
MLP (TF-IDF):         AUC = 0.9907
MLP (BERT):           AUC = 0.9826
LSTM:                 AUC = 0.9490
DistilBERT:           AUC = 0.9940

Best Model: DistilBERT Pipeline
```

DistilBERT achieves the highest AUC (0.994), followed closely by MLP with TF-IDF (0.991).

---

## Streamlit App

The interactive web app (`app.py`) provides a user-friendly interface for:

1. **Text Input** — Paste text directly or upload `.docx` / `.pdf` files (auto-extracts text, word limit: 1,000 words)
2. **Model Selection** — Choose from any individual model or run **all models for comparison**
3. **Text Statistics** — Word count, sentence length distribution, per-sentence breakdown
4. **Real-time Inference** — Prediction label (AI/Human) + confidence percentage
5. **PDF Report Generation** — Download a formatted PDF with text statistics and prediction results

### App Architecture

```
app.py ───→ inference.py ───→ models/*.pkl, models/*.h5
    │              │
    │              ├── TF-IDF Vectorizer
    │              ├── BERT Embeddings (via transformers)
    │              ├── LSTM Model (PyTorch)
    │              └── DistilBERT (via transformers)
    │
    └──→ download_report.py ───→ PDF output (via fpdf2)
```

### Running the App

```bash
streamlit run app.py
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Intro2LLM-Project1.git
cd Intro2LLM-Project1

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### To re-run the notebook training:

```bash
jupyter notebook notebooks/project1_notebook.ipynb
```

---

## Project Structure

```
├── app.py                          # Streamlit web application
├── inference.py                    # Model inference engine (all models)
├── download_report.py              # PDF report generator (fpdf2)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── train_data_with_labels.xlsx # Training dataset
│   └── Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow...pdf
├── models/
│   ├── tfidf_vectorizer.pkl        # Trained TF-IDF vectorizer
│   ├── svm_model_tfidf.pkl         # SVM trained on TF-IDF
│   ├── svm_model_bert.pkl          # SVM trained on BERT embeddings
│   ├── decision_tree_model_tfidf.pkl
│   ├── decision_tree_model_bert.pkl
│   ├── adaboost_model_tfidf.pkl
│   ├── adaboost_model_bert.pkl
│   ├── nn_model_tfidf.pkl          # MLP trained on TF-IDF
│   ├── nn_model_bert.pkl           # MLP trained on BERT embeddings
│   ├── lstm_model.pkl              # PyTorch LSTM model
│   └── distilbert_model.h5         # Fine-tuned DistilBERT state dict
└── notebooks/
    ├── project1_notebook.ipynb     # Main training notebook
    └── project1_notebook copy.ipynb # Backup copy
```

---

## References & Citations

### Papers

1. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
   - Used for BERT embedding extraction (feature engineering).

2. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).** *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS 2019 EMC<sup>2</sup> Workshop. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
   - Used for the fine-tuned DistilBERT classifier (best performing model).

3. **Hochreiter, S., & Schmidhuber, J. (1997).** *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780. [PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)
   - Used for the LSTM deep learning model.

4. **Cortes, C., & Vapnik, V. (1995).** *Support-vector networks.* Machine Learning, 20(3), 273–297.
   - Used for the SVM (LinearSVC) models.

5. **Freund, Y., & Schapire, R. E. (1997).** *A decision-theoretic generalization of on-line learning and an application to boosting.* Journal of Computer and System Sciences, 55(1), 119–139.
   - Used for the AdaBoost ensemble model.

### Libraries & Frameworks

- **Hugging Face Transformers** — [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
  - Used for BERT, DistilBERT tokenizers and models. [Wolf et al., 2020](https://aclanthology.org/2020.emnlp-demos.6/)

- **PyTorch** — [https://pytorch.org](https://pytorch.org)
  - Deep learning framework for LSTM and DistilBERT. [Paszke et al., 2019](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library)

- **scikit-learn** — [https://scikit-learn.org](https://scikit-learn.org)
  - ML models, TF-IDF vectorizer, evaluation metrics. [Pedregosa et al., 2011](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)

- **Streamlit** — [https://streamlit.io](https://streamlit.io)
  - Web application framework for the interactive demo.

- **fpdf2** — [https://pyfpdf.github.io/fpdf2/](https://pyfpdf.github.io/fpdf2/)
  - PDF report generation library.

### Related Work on AI Text Detection

- **GPTZero** — [https://gptzero.me](https://gptzero.me)
- **OpenAI AI Text Classifier** — [https://platform.openai.com/ai-text-classifier](https://platform.openai.com/ai-text-classifier)
- **GLTR (Giant Language Model Test Room)** — [https://gltr.io](https://gltr.io) — [Gehrmann et al., 2019](https://aclanthology.org/D19-5531/)
- **Mitchell et al. (2023)** — *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature.* [arXiv:2301.11305](https://arxiv.org/abs/2301.11305)

---

## License

This project was developed as part of the **Introduction to LLMs** course at [University Name]. Educational use only.