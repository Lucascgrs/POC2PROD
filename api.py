import json
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Import depuis votre fichier librairie
from StackOverflow import BertWithExtraLayers, topk_predictions, threshold_predictions

# =====================
# CONFIG
# =====================

MODEL_ID = "userfromsete/model_poc2prod_"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

# =====================
# LOAD MODEL FROM HUB
# =====================

def load_model():
    print(f"Downloading model from {MODEL_ID}...")

    # 1. Config
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="model_config.json", token=HF_TOKEN)
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # 2. Labels
    id2label_path = hf_hub_download(repo_id=MODEL_ID, filename="id2label.json", token=HF_TOKEN)
    with open(id2label_path, "r") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    # 3. Init Model (Classe importée de StackOverflow.py)
    model = BertWithExtraLayers(
        model_name=config_data["model_name"],
        num_labels=config_data["num_labels"],
        hidden_dims=config_data["hidden_dims"],
        dropout=config_data["dropout"]
    )

    # 4. Load Weights
    weights_path = hf_hub_download(repo_id=MODEL_ID, filename="pytorch_model.bin", token=HF_TOKEN)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 5. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

    return model, tokenizer, id2label

model, tokenizer, id2label = load_model()

# =====================
# FASTAPI APP
# =====================

app = FastAPI(title="StackOverflow Tag Predictor API")

class PredictRequest(BaseModel):
    title: str
    top_k: int = 3

class ThresholdRequest(BaseModel):
    title: str
    threshold: float = 0.35

class BatchPredictRequest(BaseModel):
    titles: list
    top_k: int = 3

class BatchThresholdRequest(BaseModel):
    titles: list
    threshold: float = 0.35

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Réutilisation de la fonction générique StackOverflow
    results = topk_predictions(model, tokenizer, [req.title], id2label, k=req.top_k, device=DEVICE)
    # Adaptation du format de sortie pour correspondre à l'API existante
    return {"title": req.title, "predictions": results[0]["topk"]}

@app.post("/predict_threshold")
def predict_threshold(req: ThresholdRequest):
    results = threshold_predictions(model, tokenizer, [req.title], id2label, threshold=req.threshold, device=DEVICE)
    return {"title": req.title, "predictions": results[0]["selected"]}

@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    results = topk_predictions(model, tokenizer, req.titles, id2label, k=req.top_k, device=DEVICE)
    # Reformatage léger : topk_predictions renvoie une liste de dicts avec "title" et "topk"
    # L'API attendait {"titles": [...], "predictions": [[...], [...]]}
    # Pour être conforme au return original :
    preds_only = [r["topk"] for r in results]
    return {"titles": req.titles, "predictions": preds_only}

@app.post("/batch_predict_threshold")
def batch_predict_threshold(req: BatchThresholdRequest):
    results = threshold_predictions(model, tokenizer, req.titles, id2label, threshold=req.threshold, device=DEVICE)
    # L'API originale renvoyait une structure légèrement différente, on s'adapte
    output = []
    for r in results:
        output.append({"title": r["title"], "predictions": r["selected"]})
    return {"titles": req.titles, "predictions": output}