# -*- coding: utf-8 -*-

import os
import re
import json
import time
import datetime
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
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
LOG_HISTORY_FILE = "live_training_log.json"  # Fichier partagé pour le graphique

MAX_LENGTH = 64
TRAIN_RATIO = 0.8
RANDOM_STATE = 42
EPOCHS = 8
LR = 2e-5
BATCH_SIZE = 16

TOP_N_TAGS = 20
CONTINUE_TRAINING = True

HIDDEN_DIMS = [512, 256]
DROPOUT = 0.3
NUM_LAYERS_TO_FREEZE = 6

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    for pkg in ["punkt", "wordnet"]:
        nltk.download(pkg, quiet=True)
    _LEMMATIZER = WordNetLemmatizer()
except Exception as e:
    print(f"Warning NLTK: {e}")


# ===============================
# MODEL DEFINITION (Shared)
# ===============================
# (Identique à votre version précédente)
class BertWithExtraLayers(nn.Module):
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
# (Fonctions identiques, condensées pour la lisibilité)

def load_dataset(csv_path, n_rows: int | None = None):
    df = pd.read_csv(csv_path, nrows=n_rows).drop_duplicates().dropna()
    return df


def keep_primary_tag(df):
    df = df[df["tag_position"] == 0].copy()
    return df.sort_values("post_id").drop_duplicates(subset=["post_id"], keep="first")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    if '_LEMMATIZER' in globals():
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens if len(t) > 1]
    return " ".join(tokens)


def apply_cleaning(df):
    # On définit explicitement les colonnes à nettoyer (Input)
    # On NE TOUCHE PAS à "tag_name" (Target)
    cols_to_clean = ["title"]

    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    return df


def split_dataset(df, train_ratio=0.8):
    df_train, df_temp = train_test_split(df, test_size=(1 - train_ratio), random_state=RANDOM_STATE)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=RANDOM_STATE)
    return df_train, df_val, df_test


def encode_labels(df_train, df_val, df_test):
    encoder = LabelEncoder()
    encoder.fit(df_train["tag_name"])
    known = set(encoder.classes_)
    df_val = df_val[df_val["tag_name"].isin(known)].copy()
    df_test = df_test[df_test["tag_name"].isin(known)].copy()
    df_train["labels"] = encoder.transform(df_train["tag_name"])
    df_val["labels"] = encoder.transform(df_val["tag_name"])
    df_test["labels"] = encoder.transform(df_test["tag_name"])
    return df_train, df_val, df_test, encoder


# ===============================
# TRAINING UTILS & CALLBACKS
# ===============================

class LiveVisualizationCallback(TrainerCallback):
    """
    Callback pour :
    1. Calculer l'ETA
    2. Sauvegarder les logs dans un JSON que le processus graphique va lire.
    """

    def __init__(self):
        self.start_time = None
        # On vide le fichier de log au début
        with open(LOG_HISTORY_FILE, 'w') as f:
            json.dump({"train_loss": [], "eval_loss": [], "steps": [], "eval_steps": []}, f)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(">>> Démarrage de l'entraînement...")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            # 1. Calcul ETA
            current_step = state.global_step
            max_steps = state.max_steps
            elapsed = time.time() - self.start_time

            if current_step > 0:
                avg_time_per_step = elapsed / current_step
                remaining_steps = max_steps - current_step
                remaining_seconds = remaining_steps * avg_time_per_step
                eta_str = str(datetime.timedelta(seconds=int(remaining_seconds)))
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

                print(f"  [Progress] Step {current_step}/{max_steps} | "
                      f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                      f"Loss: {logs.get('loss', 'N/A')}")

            # 2. Sauvegarde pour le graphique (JSON partagé)
            try:
                # Lecture existante
                if os.path.exists(LOG_HISTORY_FILE):
                    with open(LOG_HISTORY_FILE, 'r') as f:
                        data = json.load(f)
                else:
                    data = {"train_loss": [], "eval_loss": [], "steps": [], "eval_steps": []}

                # Mise à jour
                if "loss" in logs:
                    data["train_loss"].append(logs["loss"])
                    data["steps"].append(state.global_step)
                if "eval_loss" in logs:
                    data["eval_loss"].append(logs["eval_loss"])
                    data["eval_steps"].append(state.global_step)

                # Écriture
                with open(LOG_HISTORY_FILE, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Warning Log Write: {e}")


def plotting_process_func(json_path):
    """
    Fonction exécutée dans un processus séparé.
    Elle gère l'affichage Matplotlib sans bloquer le thread principal.
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            ax.clear()
            ax.set_title("Training Metrics Live")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")

            if data["steps"]:
                ax.plot(data["steps"], data["train_loss"], label="Training Loss", color='blue', alpha=0.6)
            if data["eval_steps"]:
                ax.plot(data["eval_steps"], data["eval_loss"], label="Validation Loss", color='red', marker='o')

            ax.legend(loc="upper right")
            ax.grid(True)
        except:
            pass  # Ignorer les erreurs de lecture concurrentielle (fichier vide ou lock)

    ani = FuncAnimation(fig, update, interval=2000)  # Mise à jour toutes les 2s
    plt.show()  # Bloquant uniquement pour CE processus, pas pour l'entrainement


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
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}


# ... (Model init et optimizer identiques, condensés) ...
def load_or_init_model_custom(num_labels, classes_list):
    model = BertWithExtraLayers(MODEL_NAME, num_labels, HIDDEN_DIMS, DROPOUT).to(default_device)
    # Freeze logic simple
    for param in model.bert.embeddings.parameters(): param.requires_grad = False
    for i in range(NUM_LAYERS_TO_FREEZE):
        for param in model.bert.encoder.layer[i].parameters(): param.requires_grad = False

    model_path = os.path.join(SAVE_DIR, "pytorch_model.bin")
    if CONTINUE_TRAINING and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=default_device)
            model.load_state_dict(state_dict, strict=False)
            print("✓ Poids chargés")
        except:
            pass

    id2label = {i: str(tag) for i, tag in enumerate(classes_list)}
    return model, id2label


def get_optimizer_custom(model):
    optimizer_params = [
        {'params': [p for p in model.bert.encoder.layer[:6].parameters() if p.requires_grad], 'lr': 1e-5},
        {'params': [p for p in model.bert.encoder.layer[6:].parameters() if p.requires_grad], 'lr': 3e-5},
        {'params': model.extra_layers.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-5}
    ]
    # Nettoyage des listes vides
    optimizer_params = [x for x in optimizer_params if x['params']]
    return AdamW(optimizer_params, weight_decay=0.01)


# ===============================
# MAIN TRAIN LOOP
# ===============================

def train_eval_save(train_ds, val_ds, test_ds, tokenizer, encoder):
    model, id2label = load_or_init_model_custom(len(encoder.classes_), encoder.classes_)
    optimizer = get_optimizer_custom(model)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=20,  # Log fréquent pour voir le graphique bouger
        report_to="none",  # Désactive WandB par défaut
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Initialisation du Callback
    live_viz_callback = LiveVisualizationCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), live_viz_callback],
        optimizers=(optimizer, None)
    )

    # --- LANCEMENT DU PROCESSUS GRAPHIQUE ---
    print(">>> Lancement de la fenêtre de visualisation...")
    plot_process = multiprocessing.Process(target=plotting_process_func, args=(LOG_HISTORY_FILE,))
    plot_process.start()

    try:
        print("DÉBUT DE L'ENTRAÎNEMENT")
        trainer.train()
    except KeyboardInterrupt:
        print("\nArrêt manuel demandé.")
    finally:
        # Nettoyage à la fin
        print("Fin de l'entraînement ou interruption.")
        if plot_process.is_alive():
            print("Note : Fermez la fenêtre du graphique pour terminer le script complètement (ou attendez).")
            # On ne tue pas brutalement le process pour laisser l'utilisateur voir le graph final
            # plot_process.join() # Décommentez pour forcer l'attente de fermeture
            # plot_process.terminate() # Décommentez pour fermer la fenêtre automatiquement

    # ==================================================
    # SAUVEGARDE ET ÉVALUATION (PARTIE RESTAURÉE)
    # ==================================================
    print("ÉVALUATION SUR LE TEST SET")
    metrics = trainer.evaluate(test_ds)
    pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Sauvegarde des poids
    model_path = os.path.join(SAVE_DIR, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    # 2. Sauvegarde du tokenizer
    tokenizer.save_pretrained(SAVE_DIR)

    # 3. Sauvegarde id2label (pour faire le lien 0 -> "python")
    id2label_path = os.path.join(SAVE_DIR, "id2label.json")
    with open(id2label_path, "w") as f:
        json.dump(id2label, f, indent=2)

    # 4. Sauvegarde de la CONFIGURATION (C'est ce qui manquait !)
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

    # 5. Sauvegarde de l'encodeur (liste complète des classes)
    label_mapping = {"classes": encoder.classes_.tolist()}
    label_mapping_path = os.path.join(SAVE_DIR, "label_encoder.json")
    with open(label_mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)

    print(f"✅ MODÈLE COMPLET SAUVEGARDÉ DANS : {SAVE_DIR}")
    return model, id2label


# ===============================
# INFERENCE UTILS
# ===============================
def predict_proba_titles(model, tokenizer, titles, device=default_device, max_length=MAX_LENGTH):
    model.eval()
    batch = tokenizer(titles, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    return F.softmax(outputs.logits, dim=1).cpu()


def topk_predictions(model, tokenizer, titles, id2label, k=3):
    probs = predict_proba_titles(model, tokenizer, titles)
    top_probs, top_idx = torch.topk(probs, k, dim=1)
    results = []
    for i, title in enumerate(titles):
        pairs = [
            {"tag_name": id2label[int(idx)] if int(idx) in id2label else id2label[str(int(idx))], "proba": float(prob)}
            for prob, idx in zip(top_probs[i], top_idx[i])]
        results.append({"title": title, "topk": pairs})
    return results


def threshold_predictions(model, tokenizer, titles, id2label, threshold=0.35, device=default_device):
    """
    Retourne tous les tags dont la probabilité dépasse le seuil (threshold).
    Utilisé par l'API pour ne pas forcer 3 tags si aucun n'est pertinent.
    """
    probs = predict_proba_titles(model, tokenizer, titles, device=device)
    results = []

    for i, title in enumerate(titles):
        above = []
        for idx, p in enumerate(probs[i]):
            if p >= threshold:
                # Gestion robuste des clés (int ou str)
                idx_int = int(idx)
                tag_name = id2label[idx_int] if idx_int in id2label else id2label[str(idx_int)]

                above.append({"tag_name": tag_name, "proba": float(p)})

        # Tri décroissant par probabilité
        above.sort(key=lambda x: x["proba"], reverse=True)
        results.append({"title": title, "selected": above})

    return results


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    # Nécessaire pour multiprocessing sous Windows
    multiprocessing.freeze_support()

    print("STACKOVERFLOW TAG PREDICTOR - ENTRAÎNEMENT AVEC LIVE PLOT")

    # 1. Load & Process
    df = load_dataset(CSV_PATH)
    df = keep_primary_tag(df)
    top_tags = df["tag_id"].value_counts().head(TOP_N_TAGS).index
    df = df[df["tag_id"].isin(top_tags)]
    df = apply_cleaning(df)
    df_train, df_val, df_test = split_dataset(df)
    df_train, df_val, df_test, encoder = encode_labels(df_train, df_val, df_test)

    # 2. Tokenize
    tokenizer = build_tokenizer()
    tokenize_fn = tokenize_function_builder(tokenizer)
    train_ds, val_ds, test_ds = to_hf_datasets(df_train, df_val, df_test, tokenize_fn)

    # 3. Train
    model, id2label = train_eval_save(train_ds, val_ds, test_ds, tokenizer, encoder)

    # 4. Test Inference
    print("\nTEST D'INFÉRENCE RAPIDE")
    sample_titles = ["How to deploy a FastAPI app with Docker?", "Understanding pointers in C"]
    results = topk_predictions(model, tokenizer, sample_titles, id2label)
    for res in results:
        print(f"Title: {res['title']} -> {res['topk'][0]['tag_name']} ({res['topk'][0]['proba']:.2f})")