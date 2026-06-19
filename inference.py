import os
import re
import sys
import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# ── Device ──────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Constants (must match training) ─────────────────────────────────────
MAX_WORDS = 1000
MAX_LEN = 200          # LSTM max sequence length
MAX_WORDS_VOCAB = 20000  # LSTM vocab size
BATCH_SIZE = 16
BERT_MAX_LENGTH = 128

# ── LSTM model definition (must match notebook) ─────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        return self.fc(x).squeeze(1)



# ── LSTM tokenizer (must match notebook) ────────────────────────────────
def simple_tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)


# ── Global caches ───────────────────────────────────────────────────────
_tfidf_vectorizer = None
_bert_tokenizer = None
_bert_model = None
_lstm_vocab = None
_distilbert_tokenizer = None
_distilbert_model_arch = None


def get_tfidf_vectorizer():
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        path = os.path.join("models", "tfidf_vectorizer.pkl")
        _tfidf_vectorizer = joblib.load(path)
    return _tfidf_vectorizer


def get_bert_embeddings(texts):
    """Convert a list of texts into BERT embeddings using mean pooling."""
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if _bert_model is None:
        _bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        _bert_model.eval()

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            encoded = _bert_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            outputs = _bert_model(**encoded)
            last_hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)

            mean_pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(embeddings)


def get_lstm_vocab():
    """Build LSTM vocab from the saved model's internal state if possible,
    otherwise rebuild from scratch (not ideal but works for inference)."""
    global _lstm_vocab
    if _lstm_vocab is not None:
        return _lstm_vocab

    # Load the LSTM model to extract vocab_size
    lstm_path = os.path.join("models", "lstm_model.pkl")
    # Patch __main__ so joblib can find LSTMClassifier in the pickle
    sys.modules['__main__'].LSTMClassifier = LSTMClassifier
    lstm_model = joblib.load(lstm_path)
    vocab_size = lstm_model.embedding.num_embeddings

    # Build a minimal vocab with the right size
    _lstm_vocab = {"<PAD>": 0, "<UNK>": 1}
    # Fill remaining slots with placeholder tokens
    for i in range(2, vocab_size):
        _lstm_vocab[f"<TOKEN_{i}>"] = i

    return _lstm_vocab


def encode_lstm_text(text, vocab):
    """Tokenize and encode text for LSTM input."""
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in simple_tokenize(text)]
    ids = ids[:MAX_LEN]
    ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long).to(device)


def get_distilbert_model():
    """Load the fine-tuned DistilBERT model."""
    global _distilbert_tokenizer, _distilbert_model_arch
    if _distilbert_tokenizer is None:
        _distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if _distilbert_model_arch is None:
        _distilbert_model_arch = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        state_dict_path = os.path.join("models", "distilbert_model.h5")
        _distilbert_model_arch.load_state_dict(torch.load(state_dict_path, map_location=device))
        _distilbert_model_arch.to(device)
        _distilbert_model_arch.eval()
    return _distilbert_tokenizer, _distilbert_model_arch


def load_model(model_name):
    """Load a model from the models/ directory by its display name (without extension)."""
    # Try .pkl first, then .h5
    pkl_path = os.path.join("models", f"{model_name}.pkl")
    h5_path = os.path.join("models", f"{model_name}.h5")

    if os.path.exists(pkl_path):
        # Patch __main__ so joblib can find LSTMClassifier in the pickle
        sys.modules['__main__'].LSTMClassifier = LSTMClassifier
        return joblib.load(pkl_path)
    elif os.path.exists(h5_path):
        return h5_path  # Return path for special handling
    else:
        raise FileNotFoundError(f"Model not found: {model_name}")


def predict(text, model_name):
    """
    Run inference on a single text string using the specified model.
    Returns (prediction_label, probability).
    """
    if not text.strip():
        return None, None

    # ── Determine model type from name ──────────────────────────────────
    is_tfidf = model_name.endswith("_tfidf")
    is_bert_emb = model_name.endswith("_bert")
    is_lstm = model_name == "lstm_model"
    is_distilbert = model_name == "distilbert_model"

    # ── TF-IDF based models ─────────────────────────────────────────────
    if is_tfidf:
        vectorizer = get_tfidf_vectorizer()
        model = load_model(model_name)
        X = vectorizer.transform([text])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X)[0]
            proba = 1 / (1 + np.exp(-score))  # sigmoid
        else:
            proba = float(model.predict(X)[0])
        pred = int(proba > 0.5)
        return pred, proba

    # ── BERT embedding based models ─────────────────────────────────────
    elif is_bert_emb:
        model = load_model(model_name)
        emb = get_bert_embeddings([text])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(emb)[0, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(emb)[0]
            proba = 1 / (1 + np.exp(-score))
        else:
            proba = float(model.predict(emb)[0])
        pred = int(proba > 0.5)
        return pred, proba

    # ── LSTM model ──────────────────────────────────────────────────────
    elif is_lstm:
        model = load_model(model_name)
        model.to(device)
        model.eval()
        vocab = get_lstm_vocab()
        input_tensor = encode_lstm_text(text, vocab)
        with torch.no_grad():
            logits = model(input_tensor)
            proba = torch.sigmoid(logits).item()
        pred = int(proba > 0.5)
        return pred, proba

    # ── DistilBERT model ────────────────────────────────────────────────
    elif is_distilbert:
        tokenizer, model = get_distilbert_model()
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            proba = probs[0, 1].item()
        pred = int(proba > 0.5)
        return pred, proba

    else:
        raise ValueError(f"Unknown model type: {model_name}")