FROM python:3.12-slim

WORKDIR /app

# Installer les dependances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY . .

# Lancer les tests au build pour valider l'image
RUN python -m pytest tests/ -v

EXPOSE 5000

CMD ["python", "app/app.py"]
