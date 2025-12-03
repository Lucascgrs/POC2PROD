FROM python:3.10-slim

# Création d'un utilisateur non-root (bonne pratique HF)
WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Permissions pour le script
RUN chmod +x entrypoint.sh

# Le port 7860 est celui exposé par HF Spaces
EXPOSE 7860

CMD ["./entrypoint.sh"]