#!/bin/bash

echo "🚀 Démarrage de l'API FastAPI..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

sleep 5

echo "🚀 Démarrage de Streamlit..."
python -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0