import os
import sys
import sqlite3
import functools
import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, render_template, request, flash, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

# Configuration du chemin pour les imports src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import preprocess_data, IMPORTANT_FEATURES

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'medpredict-bmt-2026')

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.joblib')

model = None
explainer = None

# --- Initialisation Base de Données ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

# --- Auth Decorators ---
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# --- Chargement Modèle ---
def load_resources():
    global model, explainer
    try:
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
    except:
        model, explainer = None, None

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Récupération et conversion des données du formulaire
            data = {feat: float(request.form.get(feat, 0)) for feat in IMPORTANT_FEATURES}
            X_input = pd.DataFrame([data])
            
            # Prédiction
            proba = model.predict_proba(X_input)[0]
            
            # SHAP Individuel
            shap_values = explainer.shap_values(X_input)
            # Pour RF binaire, shap_values[1] correspond à la classe 'Décès'
            val = shap_values[1] if isinstance(shap_values, list) else shap_values
            
            session['prediction'] = {
                'survival_prob': round(proba[0] * 100, 1),
                'death_prob': round(proba[1] * 100, 1),
                'shap': val[0].tolist() 
            }
            return redirect(url_for('results'))
        except Exception as e:
            flash(f"Erreur : {str(e)}", "error")
    
    return render_template('predict.html')

@app.route('/results')
@login_required
def results():
    return render_template('results.html', pred=session.get('prediction'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        flash('Identifiants incorrects', 'error')
    return render_template('auth/login.html')

@app.route('/')
@login_required
def index():
    return render_template('home.html')

if __name__ == '__main__':
    init_db()
    load_resources()
    app.run(debug=True)