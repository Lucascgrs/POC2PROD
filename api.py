import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from huggingface_hub import hf_hub_download

# =====================
# CONFIG
# =====================

MODEL_ID = "userfromsete/model_poc2prod_"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")


# =====================
# DÉFINITION DU MODÈLE CUSTOM
# =====================

class BertWithExtraLayers(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        # On passe le token ici aussi au cas où le modèle de base serait restreint (rare pour bert-base)
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            return_dict=True)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.extra_layers(pooled_output)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


# =====================
# LOAD MODEL FROM HUB
# =====================

def load_model():
    print(f"Downloading model from {MODEL_ID}...")

    # AJOUT DU PARAMÈTRE token=HF_TOKEN À CHAQUE APPEL

    # 1. Config
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="model_config.json", token=HF_TOKEN)
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # 2. Labels
    id2label_path = hf_hub_download(repo_id=MODEL_ID, filename="id2label.json", token=HF_TOKEN)
    with open(id2label_path, "r") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    # 3. Init Model
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

    # 5. Tokenizer (Ajout token ici aussi)
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


# =====================
# PREDICTION FUNCTIONS
# =====================

def predict_logits(titles: list):
    batch = tokenizer(titles, return_tensors="pt", truncation=True, padding=True, max_length=64)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)
    return probs.cpu()


def process_predictions(probs, top_k):
    top_probs, top_idx = torch.topk(probs, top_k, dim=1)
    results = []
    for i, p in zip(top_idx, top_probs):
        pairs = []
        for idx, prob in zip(i, p):
            tag_id = int(id2label[int(idx)])
            pairs.append({"tag_id": tag_id, "proba": float(prob)})
        results.append(pairs)
    return results


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    probs = predict_logits([req.title])
    results = process_predictions(probs, req.top_k)
    return {"title": req.title, "predictions": results[0]}


@app.post("/predict_threshold")
def predict_threshold(req: ThresholdRequest):
    probs = predict_logits([req.title])
    results = []
    for idx, p in enumerate(probs[0]):
        if p >= req.threshold:
            tag_id = int(id2label[idx])
            results.append({"tag_id": tag_id, "proba": float(p)})
    results = sorted(results, key=lambda x: x["proba"], reverse=True)
    return {"title": req.title, "predictions": results}


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    probs = predict_logits(req.titles)
    results = process_predictions(probs, req.top_k)
    return {"titles": req.titles, "predictions": results}


@app.post("/batch_predict_threshold")
def batch_predict_threshold(req: BatchThresholdRequest):
    probs = predict_logits(req.titles)
    results = []
    for i, title_probs in zip(req.titles, probs):
        above_threshold = []
        for idx, p in enumerate(title_probs):
            if p >= req.threshold:
                tag_id = int(id2label[idx])
                above_threshold.append({"tag_id": tag_id, "proba": float(p)})
        above_threshold = sorted(above_threshold, key=lambda x: x["proba"], reverse=True)
        results.append({"title": i, "predictions": above_threshold})
    return {"titles": req.titles, "predictions": results}