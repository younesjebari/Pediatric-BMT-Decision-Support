# Utiliser une image Python officielle légère
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du projet dans le conteneur
COPY . .

# Indiquer le port utilisé par Streamlit
EXPOSE 8501

# La commande pour lancer l'application web
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]