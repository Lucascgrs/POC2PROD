# StackOverflow Tag Predictor

Ce projet permet de **prédire automatiquement le tag d’un post StackOverflow à partir de son titre**, grâce à un modèle **BERT fine-tuné**.

---

## Modes d’utilisation

### **1️⃣ Mode simple : utiliser uniquement le script principal**
Le fichier **`StackOverflow.py`** permet :

- de **charger** un modèle déjà fine-tuné,  
- ou d’**entraîner** un nouveau modèle depuis zéro,  
- puis de réaliser des **prédictions** directement en Python.

C’est la manière la plus simple d’utiliser le projet.

---

### **2️⃣ Mode API : FastAPI**

Le fichier **`api.py`** expose une API web avec plusieurs endpoints :

- `POST /predict` → prédiction top-k  
- `POST /predict_threshold` → prédiction avec seuil  
- `GET /health` → statut du serveur

Lancer l’API :

```bash
uvicorn api:app --reload --port 8000
```


## Option 2 : Interface Streamlit + Explicabilité

Le fichier **`app.py`** fournit une interface web :

- **Onglet “Prédiction simple”** : appelle l’API FastAPI pour prédire les tags à partir d’un titre.  
- **Onglet “Explicabilité (LIME)”** : utilise directement le modèle local pour expliquer quels mots du titre ont le plus influencé la prédiction.

### Lancement

```bash
python -m streamlit run app.py 
```