import json
import requests
import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer
from huggingface_hub import hf_hub_download

# Import depuis votre fichier librairie
from StackOverflow import BertWithExtraLayers

# ==========================
# CONFIG
# ==========================

API_URL = "http://localhost:8000"
MODEL_ID = "userfromsete/model_poc2prod_"
MAX_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(
    page_title="StackOverflow Tag Predictor",
    page_icon="🔮",
    layout="centered"
)

@st.cache_resource
def load_model_tokenizer_id2label():
    # 1. Config
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="model_config.json", token=HF_TOKEN)
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # 2. Labels
    id2label_path = hf_hub_download(repo_id=MODEL_ID, filename="id2label.json", token=HF_TOKEN)
    with open(id2label_path, "r") as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}

    # 3. Instancier le modèle Custom (Classe importée de StackOverflow.py)
    model = BertWithExtraLayers(
        model_name=config_data["model_name"],
        num_labels=config_data["num_labels"],
        hidden_dims=config_data["hidden_dims"],
        dropout=config_data["dropout"]
    )

    # 4. Charger les poids
    weights_path = hf_hub_download(repo_id=MODEL_ID, filename="pytorch_model.bin", token=HF_TOKEN)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 5. Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

    return model, tokenizer, id2label

@st.cache_resource
def build_lime_explainer(id2label):
    num_classes = len(id2label)
    class_names = [str(id2label[i]) for i in range(num_classes)]
    explainer = LimeTextExplainer(class_names=class_names)
    return explainer

def make_predict_proba_fn(model, tokenizer):
    """
    Fonction helper locale pour LIME qui a besoin d'un format spécifique (numpy array).
    """
    def _predict(texts):
        if isinstance(texts, str):
            texts = [texts]

        batch = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

        probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    return _predict

# ==========================
# UI
# ==========================

st.title("🔮 StackOverflow Tag Predictor")
st.write("Modèle BERT fine-tuné pour prédire les tags d’un post à partir de son titre.")
st.markdown("---")

tab_pred, tab_explain = st.tabs(["🔮 Prédiction simple", "🧠 Explicabilité (LIME)"])

# --------------------------------------------------
# 🔮 Onglet 1 : Prédiction via API FastAPI
# --------------------------------------------------
with tab_pred:
    st.subheader("Prédiction via l’API FastAPI")
    mode = st.radio("Mode de prédiction :", ["Top-K", "Threshold", "Batch Top-K", "Batch Threshold"], horizontal=True)

    if mode in ["Top-K", "Threshold"]:
        title_input = st.text_input("✍️ Titre StackOverflow :", "")
    else:
        title_input = None

    if mode == "Top-K":
        top_k = st.slider("Nombre de tags à prédire (k)", 1, 10, 3)
        if st.button("Prédire (API)", type="primary"):
            if not title_input.strip():
                st.warning("Merci de saisir un titre.")
            else:
                payload = {"title": title_input, "top_k": top_k}
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        preds = result.get("predictions", [])
                        if not preds:
                            st.info("Aucune prédiction retournée.")
                        else:
                            st.markdown("### Résultats")
                            df = pd.DataFrame(preds)
                            df["proba"] = df["proba"].round(3)
                            st.table(df)
                    else:
                        st.error(f"Erreur API : code {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur de connexion à l’API : {e}")

    elif mode == "Threshold":
        threshold = st.slider("Seuil de probabilité", 0.0, 1.0, 0.35, 0.01)
        if st.button("Prédire (API)", type="primary"):
            if not title_input.strip():
                st.warning("Merci de saisir un titre.")
            else:
                payload = {"title": title_input, "threshold": threshold}
                try:
                    response = requests.post(f"{API_URL}/predict_threshold", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        preds = result.get("predictions", [])
                        if not preds:
                            st.info("Aucun tag au-dessus du seuil.")
                        else:
                            st.markdown("### Résultats")
                            df = pd.DataFrame(preds)
                            df["proba"] = df["proba"].round(3)
                            st.table(df)
                    else:
                        st.error(f"Erreur API : code {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur de connexion à l’API : {e}")

    elif mode == "Batch Top-K":
        st.subheader("Batch Prediction (Top-K)")
        top_k_batch = st.slider("Nombre de tags (k)", 1, 10, 3)
        titles_batch = st.text_area("Saisir une liste de titres (séparés par des retours à la ligne)", "")
        if st.button("Prédire le batch (API)", type="primary"):
            if not titles_batch.strip():
                st.warning("Merci de saisir des titres.")
            else:
                titles = [t.strip() for t in titles_batch.splitlines() if t.strip()]
                payload = {"titles": titles, "top_k": top_k_batch}
                try:
                    response = requests.post(f"{API_URL}/batch_predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        preds_list = result.get("predictions", [])
                        titles_res = result.get("titles", [])
                        st.markdown("### Résultats")
                        data_display = []
                        for t, p_list in zip(titles_res, preds_list):
                            tags_str = ", ".join([f"{item['tag_id']} ({item['proba']:.2f})" for item in p_list])
                            data_display.append({"Titre": t, "Prédictions": tags_str})
                        st.table(pd.DataFrame(data_display))
                    else:
                        st.error(f"Erreur API : code {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur de connexion à l’API : {e}")

    elif mode == "Batch Threshold":
        st.subheader("Batch Prediction (Threshold)")
        threshold_batch = st.slider("Seuil de probabilité", 0.0, 1.0, 0.35, 0.01)
        titles_batch = st.text_area("Saisir une liste de titres (séparés par des retours à la ligne)", "")
        if st.button("Prédire le batch avec seuil (API)", type="primary"):
            if not titles_batch.strip():
                st.warning("Merci de saisir des titres.")
            else:
                titles = [t.strip() for t in titles_batch.splitlines() if t.strip()]
                payload = {"titles": titles, "threshold": threshold_batch}
                try:
                    response = requests.post(f"{API_URL}/batch_predict_threshold", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        preds_list = result.get("predictions", [])
                        st.markdown("### Résultats")
                        data_display = []
                        for item in preds_list:
                            t = item['title']
                            p_list = item['predictions']
                            tags_str = ", ".join([f"{p['tag_id']} ({p['proba']:.2f})" for p in p_list]) if p_list else "Aucun tag"
                            data_display.append({"Titre": t, "Prédictions": tags_str})
                        st.table(pd.DataFrame(data_display))
                    else:
                        st.error(f"Erreur API : code {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur de connexion à l’API : {e}")

# --------------------------------------------------
# 🧠 Onglet 2 : Explicabilité (LIME)
# --------------------------------------------------
with tab_explain:
    st.subheader("Explication des prédictions avec LIME")
    explain_title = st.text_input("✍️ Titre à expliquer :", "")

    try:
        model, tokenizer, id2label = load_model_tokenizer_id2label()
        explainer = build_lime_explainer(id2label)
        predict_proba_fn = make_predict_proba_fn(model, tokenizer)
        st.success("Modèle chargé pour LIME.")
    except Exception as e:
        st.error(f"Erreur chargement modèle local: {e}")
        st.stop()

    top_k_for_display = st.slider("Afficher les k meilleurs tags prédits pour analyse", 1, 10, 3)

    if st.button("Expliquer la prédiction", type="primary"):
        if not explain_title.strip():
            st.warning("Merci de saisir un titre.")
        else:
            probs = predict_proba_fn([explain_title])[0]
            indices = np.argsort(probs)[::-1]
            top_indices = indices[:top_k_for_display]
            top_info = []
            for idx in top_indices:
                idx = int(idx)
                tag_name = id2label.get(idx, id2label.get(str(idx), "Unknown"))
                top_info.append({"tag_id": tag_name, "proba": float(probs[idx])})

            st.markdown("### Top prédictions (modèle local)")
            df_top = pd.DataFrame(top_info)
            df_top["proba"] = df_top["proba"].round(3)
            st.table(df_top)

            top_label_idx = int(top_indices[0])
            top_tag_name = id2label.get(top_label_idx, id2label.get(str(top_label_idx), "Unknown"))

            with st.spinner("Calcul de l’explication LIME (quelques secondes)..."):
                explanation = explainer.explain_instance(
                    explain_title,
                    predict_proba_fn,
                    num_features=10,
                    labels=[top_label_idx]
                )

            st.markdown(f"### Explication pour le tag : **{top_tag_name}**")
            weights = explanation.as_list(label=top_label_idx)
            df_weights = pd.DataFrame(weights, columns=["Token", "Contribution"])
            st.write("Les barres **vertes** indiquent les mots qui confirment ce tag.")
            st.bar_chart(df_weights.set_index("Token"))
            with st.expander("Voir les valeurs exactes"):
                st.table(df_weights)