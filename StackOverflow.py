# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter

# NLP
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# HF / Torch
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    ProgressCallback,
    EarlyStoppingCallback
)
from torch.optim import AdamW
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

# ===============================
# CONFIGURATION & GLOBALS
# ===============================

CSV_PATH = "stackoverflow_posts.csv"
MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "./model_finetuned_bert"
OUTPUT_DIR = "./results"

MAX_LENGTH = 64
TRAIN_RATIO = 0.8
RANDOM_STATE = 42
EPOCHS = 1
LR = 2e-5
BATCH_SIZE = 16

TOP_N_TAGS = 10
CONTINUE_TRAINING = True

HIDDEN_DIMS = [512, 256]
DROPOUT = 0.3
NUM_LAYERS_TO_FREEZE = 6

# Définition du device par défaut (peut être surchargé dans les fonctions)
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialisation NLTK silencieuse
try:
    for pkg in ["punkt", "wordnet"]:
        nltk.download(pkg, quiet=True)
    _LEMMATIZER = WordNetLemmatizer()
except Exception as e:
    print(f"Warning NLTK: {e}")


# ===============================
# MODEL DEFINITION (Shared)
# ===============================

class BertWithExtraLayers(nn.Module):
    """
    Enveloppe BERT avec des couches supplémentaires avant le classificateur final.
    Utilisée dans StackOverflow.py, api.py et app.py.
    """

    def __init__(self, model_name, num_labels, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        layers = []
        input_dim = self.bert.config.hidden_size

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim

        self.extra_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_dim, num_labels)
        self.config = self.bert.config
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            return_dict=True)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.extra_layers(pooled_output)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


# ===============================
# DATA PROCESSING
# ===============================

def load_dataset(csv_path, n_rows: int | None = None):
    df = pd.read_csv(csv_path, nrows=n_rows)
    df = df.drop_duplicates().dropna()
    return df


def keep_primary_tag(df):
    df = df[df["tag_position"] == 0].copy()
    df = df.sort_values("post_id").drop_duplicates(subset=["post_id"], keep="first")
    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    if '_LEMMATIZER' in globals():
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens if len(t) > 1]
    return " ".join(tokens)


def apply_cleaning(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(clean_text)
    return df


def split_dataset(df, train_ratio=0.8):
    df_train, df_temp = train_test_split(df, test_size=(1 - train_ratio), random_state=RANDOM_STATE)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=RANDOM_STATE)
    return df_train, df_val, df_test


def encode_labels(df_train, df_val, df_test):
    encoder = LabelEncoder()
    encoder.fit(df_train["tag_id"])
    known = set(encoder.classes_)
    df_val = df_val[df_val["tag_id"].isin(known)].copy()
    df_test = df_test[df_test["tag_id"].isin(known)].copy()
    df_train["labels"] = encoder.transform(df_train["tag_id"])
    df_val["labels"] = encoder.transform(df_val["tag_id"])
    df_test["labels"] = encoder.transform(df_test["tag_id"])
    return df_train, df_val, df_test, encoder


# ===============================
# TRAINING UTILS
# ===============================

def build_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function_builder(tokenizer):
    def _tok(batch):
        return tokenizer(batch["title"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    return _tok


def to_hf_datasets(df_train, df_val, df_test, tokenize_fn):
    cols = ["title", "labels"]
    train_ds = Dataset.from_pandas(df_train[cols], preserve_index=False)
    val_ds = Dataset.from_pandas(df_val[cols], preserve_index=False)
    test_ds = Dataset.from_pandas(df_test[cols], preserve_index=False)
    return (
        train_ds.map(tokenize_fn, batched=True),
        val_ds.map(tokenize_fn, batched=True),
        test_ds.map(tokenize_fn, batched=True)
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


def freeze_layers_custom(model, num_layers_to_freeze=6):
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i in range(num_layers_to_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Gelé : embeddings + {num_layers_to_freeze} premières couches")
    print(f"✓ Paramètres entraînables : {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")


def get_optimizer_custom(model):
    optimizer_params = []
    low_layers_params = [p for p in model.bert.encoder.layer[:6].parameters() if p.requires_grad]
    if low_layers_params:
        optimizer_params.append({'params': low_layers_params, 'lr': 1e-5})
    high_layers_params = [p for p in model.bert.encoder.layer[6:].parameters() if p.requires_grad]
    if high_layers_params:
        optimizer_params.append({'params': high_layers_params, 'lr': 3e-5})
    optimizer_params.append({'params': model.extra_layers.parameters(), 'lr': 5e-5})
    optimizer_params.append({'params': model.classifier.parameters(), 'lr': 5e-5})
    return AdamW(optimizer_params, weight_decay=0.01)


def load_or_init_model_custom(num_labels, classes_list):
    model = BertWithExtraLayers(
        model_name=MODEL_NAME,
        num_labels=num_labels,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    ).to(default_device)

    freeze_layers_custom(model, num_layers_to_freeze=NUM_LAYERS_TO_FREEZE)

    model_path = os.path.join(SAVE_DIR, "pytorch_model.bin")
    if CONTINUE_TRAINING and os.path.exists(model_path):
        print(f"⚠️ Chargement des poids existants depuis {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=default_device)
            model.load_state_dict(state_dict, strict=False)
            print("✓ Poids chargés avec succès")
        except Exception as e:
            print(f"⚠️ Impossible de charger les poids : {e}")
            print("→ Démarrage avec un nouveau modèle")

    id2label = {i: str(tag) for i, tag in enumerate(classes_list)}
    return model, id2label


def train_eval_save(train_ds, val_ds, test_ds, tokenizer, encoder):
    model, id2label = load_or_init_model_custom(len(encoder.classes_), encoder.classes_)
    optimizer = get_optimizer_custom(model)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        optimizers=(optimizer, None)
    )

    print("DÉBUT DE L'ENTRAÎNEMENT")
    trainer.train()
    print("ÉVALUATION SUR LE TEST SET")
    metrics = trainer.evaluate(test_ds)
    pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(SAVE_DIR)

    id2label_path = os.path.join(SAVE_DIR, "id2label.json")
    with open(id2label_path, "w") as f:
        json.dump(id2label, f, indent=2)

    model_config = {
        "model_name": MODEL_NAME,
        "num_labels": len(encoder.classes_),
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "max_length": MAX_LENGTH,
        "num_layers_frozen": NUM_LAYERS_TO_FREEZE,
        "architecture": "BertWithExtraLayers"
    }
    config_path = os.path.join(SAVE_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    label_mapping = {"classes": encoder.classes_.tolist()}
    label_mapping_path = os.path.join(SAVE_DIR, "label_encoder.json")
    with open(label_mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)

    print(f"✅ MODÈLE COMPLET SAUVEGARDÉ DANS : {SAVE_DIR}")
    return model, id2label


# ===============================
# INFERENCE / PREDICTION UTILS
# ===============================

def predict_proba_titles(model, tokenizer, titles, device=default_device, max_length=MAX_LENGTH):
    """
    Prédit les probabilités pour une liste de titres.
    Réutilisable dans api.py et app.py.
    """
    model.eval()
    batch = tokenizer(titles, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits

    return F.softmax(logits, dim=1).cpu()


def topk_predictions(model, tokenizer, titles, id2label, k=3, device=default_device):
    probs = predict_proba_titles(model, tokenizer, titles, device=device)
    top_probs, top_idx = torch.topk(probs, k, dim=1)

    results = []
    for i, title in enumerate(titles):
        pairs = []
        for prob, idx in zip(top_probs[i], top_idx[i]):
            idx = int(idx)
            # Gestion robuste si id2label est int ou str
            tag_id = int(id2label[idx]) if idx in id2label else int(id2label[str(idx)])
            pairs.append({"tag_id": tag_id, "proba": float(prob)})
        results.append({"title": title, "topk": pairs})
    return results


def threshold_predictions(model, tokenizer, titles, id2label, threshold=0.35, device=default_device):
    probs = predict_proba_titles(model, tokenizer, titles, device=device)
    results = []

    for i, title in enumerate(titles):
        above = []
        for idx, p in enumerate(probs[i]):
            if p >= threshold:
                idx = int(idx)
                tag_id = int(id2label[idx]) if idx in id2label else int(id2label[str(idx)])
                above.append({"tag_id": tag_id, "proba": float(p)})

        above.sort(key=lambda x: x["proba"], reverse=True)
        results.append({"title": title, "selected": above})

    return results


# ===============================
# MAIN ENTRY POINT
# ===============================

def main():
    print("STACKOVERFLOW TAG PREDICTOR - ENTRAÎNEMENT")

    # 1. Load
    df = load_dataset(CSV_PATH)
    # 2. Filter
    df = keep_primary_tag(df)
    # 3. Top N
    top_tags = df["tag_id"].value_counts().head(TOP_N_TAGS).index
    df = df[df["tag_id"].isin(top_tags)]
    # 4. Clean
    df = apply_cleaning(df)
    # 5. Split
    df_train, df_val, df_test = split_dataset(df)
    # 6. Encode
    df_train, df_val, df_test, encoder = encode_labels(df_train, df_val, df_test)
    # 7. Tokenize
    tokenizer = build_tokenizer()
    tokenize_fn = tokenize_function_builder(tokenizer)
    train_ds, val_ds, test_ds = to_hf_datasets(df_train, df_val, df_test, tokenize_fn)

    # 8. Train
    model, id2label = train_eval_save(train_ds, val_ds, test_ds, tokenizer, encoder)

    # Test Inference
    print("TEST D'INFÉRENCE")
    sample_titles = [
        "How to deploy a FastAPI app with Docker?",
        "Understanding pointers in C",
        "Center a div in CSS Grid"
    ]
    results_topk = topk_predictions(model, tokenizer, sample_titles, id2label, device=default_device)
    for res in results_topk:
        print(f"Titre : {res['title']}")
        for pred in res['topk']:
            print(f"  - Tag {pred['tag_id']} : {pred['proba']:.4f}")


if __name__ == "__main__":
    main()