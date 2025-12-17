import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer
from huggingface_hub import hf_hub_download

# --- IMPORTS LOCAUX ---
# On importe directement la logique au lieu de l'appeler via API
from StackOverflow import (
    BertWithExtraLayers,
    topk_predictions,
    threshold_predictions
)

# ==========================
# CONFIGURATION
# ==========================

# Remplacer par votre ID Hugging Face réel
MODEL_ID = "userfromsete/model_poc2prod_"
MAX_LENGTH = 64
# Sur HF Spaces gratuit, forcer le CPU évite souvent des plantages,
# mais on tente CUDA si dispo.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(
    page_title="StackOverflow Tag Predictor",
    page_icon="🔮",
    layout="centered"
)


# ==========================
# CHARGEMENT DU MODÈLE (CACHÉ)
# ==========================
@st.cache_resource
def load_model_resources():
    """
    Charge le modèle, le tokenizer et le dictionnaire de labels une seule fois.
    Utilise le cache Streamlit pour ne pas recharger à chaque clic.
    """
    try:
        # 1. Config
        config_path = hf_hub_download(repo_id=MODEL_ID, filename="model_config.json", token=HF_TOKEN)
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # 2. Labels
        id2label_path = hf_hub_download(repo_id=MODEL_ID, filename="id2label.json", token=HF_TOKEN)
        with open(id2label_path, "r") as f:
            raw = json.load(f)
        # Conversion des clés string (JSON) en int
        id2label = {int(k): v for k, v in raw.items()}

        # 3. Modèle
        model = BertWithExtraLayers(
            model_name=config_data["model_name"],
            num_labels=config_data["num_labels"],
            hidden_dims=config_data["hidden_dims"],
            dropout=config_data["dropout"]
        )

        # 4. Poids
        weights_path = hf_hub_download(repo_id=MODEL_ID, filename="pytorch_model.bin", token=HF_TOKEN)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        # 5. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

        return model, tokenizer, id2label

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle depuis Hugging Face : {e}")
        return None, None, None


@st.cache_resource
def build_lime_explainer(id2label):
    num_classes = len(id2label)
    # On s'assure que l'ordre correspond aux indices 0, 1, 2...
    class_names = [str(id2label[i]) for i in range(num_classes)]
    explainer = LimeTextExplainer(class_names=class_names)
    return explainer


def make_predict_proba_fn(model, tokenizer):
    """Wrapper pour LIME"""

    def _predict(texts):
        if isinstance(texts, str):
            texts = [texts]

        # Tokenisation
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
# INTERFACE UTILISATEUR
# ==========================

st.title("🔮 StackOverflow Tag Predictor")
st.write("Modèle BERT fine-tuné (Chargement direct sans API externe).")
st.markdown("---")

# Chargement initial
model, tokenizer, id2label = load_model_resources()

if model is None:
    st.stop()  # Arrête l'app si le modèle n'est pas chargé

tab_pred, tab_explain = st.tabs(["🔮 Prédiction & Batch", "🧠 Explicabilité (LIME)"])

# --------------------------------------------------
# 🔮 Onglet 1 : Prédiction (Directe)
# --------------------------------------------------
with tab_pred:
    st.subheader("Prédiction")
    mode = st.radio("Mode :", ["Top-K", "Threshold", "Batch Top-K", "Batch Threshold"], horizontal=True)

    # --- Entrées ---
    if mode in ["Top-K", "Threshold"]:
        title_input = st.text_input("✍️ Titre StackOverflow :", "")
        titles_to_predict = [title_input] if title_input.strip() else []
    else:
        titles_batch_text = st.text_area("Saisir des titres (un par ligne)", "")
        titles_to_predict = [t.strip() for t in titles_batch_text.splitlines() if t.strip()]

    # --- Paramètres ---
    k_val = 3
    thresh_val = 0.35

    if "Top-K" in mode:
        k_val = st.slider("Nombre de tags (k)", 1, 10, 3)
    else:
        thresh_val = st.slider("Seuil de probabilité", 0.0, 1.0, 0.35, 0.01)

    # --- Bouton Action ---
    if st.button("Lancer la prédiction", type="primary"):
        if not titles_to_predict:
            st.warning("Veuillez saisir au moins un titre.")
        else:
            with st.spinner("Calcul en cours..."):
                try:
                    # LOGIQUE DIRECTE (Remplacement de l'appel API)
                    if "Top-K" in mode:
                        # Utilise la fonction importée de StackOverflow.py
                        results = topk_predictions(
                            model, tokenizer, titles_to_predict, id2label, k=k_val
                        )
                        # Formatage pour affichage
                        data_display = []
                        for res in results:
                            tags_str = ", ".join([f"{item['tag_name']} ({item['proba']:.2f})" for item in res['topk']])
                            data_display.append({"Titre": res['title'], "Prédictions": tags_str})

                        st.table(pd.DataFrame(data_display))

                    else:  # Threshold
                        # Utilise la fonction importée de StackOverflow.py
                        results = threshold_predictions(
                            model, tokenizer, titles_to_predict, id2label, threshold=thresh_val, device=DEVICE
                        )
                        # Formatage
                        data_display = []
                        for res in results:
                            tags_list = res.get('selected', [])
                            if not tags_list:
                                tags_str = "(Aucun tag au-dessus du seuil)"
                            else:
                                tags_str = ", ".join(
                                    [f"{item['tag_name']} ({item['proba']:.2f})" for item in tags_list])
                            data_display.append({"Titre": res['title'], "Prédictions": tags_str})

                        st.table(pd.DataFrame(data_display))

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")

# --------------------------------------------------
# 🧠 Onglet 2 : Explicabilité (LIME)
# --------------------------------------------------
with tab_explain:
    st.subheader("Explication des prédictions avec LIME")
    explain_title = st.text_input("✍️ Titre à expliquer :", "", key="lime_input")

    explainer = build_lime_explainer(id2label)
    predict_proba_fn = make_predict_proba_fn(model, tokenizer)

    top_k_for_display = st.slider("Afficher les k meilleurs tags pour analyse", 1, 10, 3, key="lime_slider")

    if st.button("Expliquer", type="primary"):
        if not explain_title.strip():
            st.warning("Merci de saisir un titre.")
        else:
            # 1. Prédiction simple pour afficher le classement
            probs = predict_proba_fn([explain_title])[0]
            indices = np.argsort(probs)[::-1]
            top_indices = indices[:top_k_for_display]

            top_info = []
            for idx in top_indices:
                idx = int(idx)
                tag_name = id2label.get(idx, id2label.get(str(idx), "Unknown"))
                top_info.append({"Tag": tag_name, "Probabilité": float(probs[idx])})

            st.markdown("### Classement du modèle")
            st.table(pd.DataFrame(top_info))

            # 2. LIME
            top_label_idx = int(top_indices[0])  # On explique le 1er choix
            top_tag_name = id2label.get(top_label_idx, id2label.get(str(top_label_idx), "Unknown"))

            st.markdown(f"### Pourquoi le modèle a choisi **{top_tag_name}** ?")
            with st.spinner("Calcul LIME (peut être lent sur CPU)..."):
                explanation = explainer.explain_instance(
                    explain_title,
                    predict_proba_fn,
                    num_features=10,
                    labels=[top_label_idx]
                )

            weights = explanation.as_list(label=top_label_idx)
            df_weights = pd.DataFrame(weights, columns=["Mot (Token)", "Contribution"])

            st.write("Les barres **vertes** poussent vers ce tag, les **rouges** éloignent.")
            st.bar_chart(df_weights.set_index("Mot (Token)"))