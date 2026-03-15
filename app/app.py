import os
import sys
import sqlite3
import functools
import numpy as np
import pandas as pd
import joblib
import shap
from flask import (Flask, render_template, request, flash,
                   redirect, url_for, session)
from werkzeug.security import generate_password_hash, check_password_hash

# Add src/ to path for data_processing imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'medpredict-bmt-2026')

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migrate: add is_admin column if missing (existing DB)
    cursor = conn.execute("PRAGMA table_info(users)")
    columns = [row['name'] for row in cursor.fetchall()]
    if 'is_admin' not in columns:
        conn.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0')
    conn.commit()
    # Create default admin account if none exists
    admin = conn.execute('SELECT id FROM users WHERE is_admin = 1').fetchone()
    if not admin:
        conn.execute(
            'INSERT OR IGNORE INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)',
            ('admin', generate_password_hash('admin'))
        )
        conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter pour acceder a cette page.', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter pour acceder a cette page.', 'info')
            return redirect(url_for('login'))
        if not session.get('is_admin'):
            flash('Acces reserve aux administrateurs.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Model / SHAP
# ---------------------------------------------------------------------------
// FEATURE_COLUMNS will be set dynamically after model loading
FEATURE_COLUMNS = []  # Placeholder, will be updated
FEATURE_LABELS = {
    'CD3dkgx10d8': 'Dose CD3+',
    'CD34kgx10d6': 'Dose CD34+',
    'Rbodymass': 'Masse Corporelle',
    'Recipientage': 'Age Receveur',
    'PLTrecovery': 'Recup. Plaquettes',
    'Disease': 'Type de Maladie',
    'Relapse': 'Rechute',
    'extcGvHD': 'GvHD Chronique',
    'Donorage': 'Age Donneur',
    'HLAmatch': 'Compatibilite HLA',
    'Riskgroup': 'Groupe de Risque'
}

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.joblib'))
model = None
explainer = None

def load_model():
    global model, explainer, FEATURE_COLUMNS
    try:
        model = joblib.load(MODEL_PATH)
        # Determine explainer type based on model
        if hasattr(model, 'feature_importances_'):  # Tree-based model
            explainer = shap.TreeExplainer(model)
        else:  # Other model (e.g., SVM) - use KernelExplainer
            # Create a small background dataset for KernelExplainer
            background = pd.DataFrame(np.random.rand(100, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
            explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Dynamically set FEATURE_COLUMNS from training data preprocessing
        from scipy.io import arff
        from data_processing import select_features_from_eda, handle_missing_values, handle_outliers
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bone-marrow.arff')
        raw_data, _ = arff.loadarff(data_path)
        df = pd.DataFrame(raw_data)
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode('utf-8')
        df = select_features_from_eda(df, target='survival_status')
        df = handle_missing_values(df)
        df = handle_outliers(df)
        FEATURE_COLUMNS = [col for col in df.columns if col != 'survival_status']
        
        print(f"Model loaded: {type(model).__name__}, Features: {FEATURE_COLUMNS}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        explainer = None

def compute_shap_summary():
    try:
        from scipy.io import arff
        from data_processing import select_features_from_eda, handle_missing_values, handle_outliers

        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bone-marrow.arff')
        raw_data, _ = arff.loadarff(data_path)
        df = pd.DataFrame(raw_data)
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode('utf-8')

        df = select_features_from_eda(df, target='survival_status')  # Use dynamic selection
        df = handle_missing_values(df)
        df = handle_outliers(df)

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

        X = df[FEATURE_COLUMNS]
        shap_values = explainer.shap_values(X)

        mean_shap = np.mean(shap_values, axis=0)
        abs_mean_shap = np.mean(np.abs(shap_values), axis=0)
        max_abs = max(abs(v) for v in mean_shap) if len(mean_shap) > 0 else 1

        features = []
        for i, col in enumerate(FEATURE_COLUMNS):
            features.append({
                'name': FEATURE_LABELS.get(col, col),
                'feature_key': col,
                'value': float(mean_shap[i]),
                'abs_value': float(abs_mean_shap[i]),
                'bar_width': min(50, abs(float(mean_shap[i])) / max_abs * 50) if max_abs > 0 else 0
            })
        features.sort(key=lambda x: x['abs_value'], reverse=True)
        return features
    except Exception as e:
        print(f"Error computing SHAP summary: {e}")
        return None

# ---------------------------------------------------------------------------
// ... (auth routes unchanged)

# ---------------------------------------------------------------------------
// App routes
# ---------------------------------------------------------------------------
// ... (index, logout unchanged)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        form_data = session.get('form_data', {})
        return render_template('predict.html', form_data=form_data)

    if model is None or explainer is None:
        flash("Le modele n'est pas charge.", 'error')
        return redirect(url_for('predict'))

    try:
        # Collect raw input
        raw_data = {
            'Recipientage': float(request.form.get('Recipientage', 9.6)),
            'Rbodymass': float(request.form.get('Rbodymass', 33.0)),
            'Disease': int(request.form.get('Disease', 0)),
            'Riskgroup': int(request.form.get('Riskgroup', 0)),
            'Relapse': int(request.form.get('Relapse', 0)),
            'Donorage': float(request.form.get('Donorage', 33.5)),
            'HLAmatch': int(request.form.get('HLAmatch', 0)),
            'extcGvHD': int(request.form.get('extcGvHD', 0)),
            'CD34kgx10d6': float(request.form.get('CD34kgx10d6', 9.7)),
            'CD3dkgx10d8': float(request.form.get('CD3dkgx10d8', 4.3)),
            'PLTrecovery': float(request.form.get('PLTrecovery', 21)),
        }

        # Create DataFrame and apply preprocessing (same as training)
        input_df = pd.DataFrame([raw_data])
        input_df = select_features_from_eda(input_df, target='survival_status')  # Dynamic selection
        input_df = handle_missing_values(input_df)
        input_df = handle_outliers(input_df)

        # Ensure order matches FEATURE_COLUMNS
        X_input = input_df[FEATURE_COLUMNS]

        proba = model.predict_proba(X_input)[0]
        survival_prob = float(round(float(proba[0]) * 100, 1))
        death_prob = float(round(float(proba[1]) * 100, 1))

        shap_values = explainer.shap_values(X_input)
        individual_shap = shap_values[0] if isinstance(shap_values, list) else shap_values
        max_abs_shap = float(max(abs(float(v)) for v in individual_shap)) if len(individual_shap) > 0 else 1.0

        shap_features = []
        for i, col in enumerate(FEATURE_COLUMNS):
            sv = float(individual_shap[i])
            shap_features.append({
                'name': FEATURE_LABELS.get(col, col),
                'feature_key': col,
                'value': sv,
                'bar_width': min(100.0, abs(sv) / max_abs_shap * 80) if max_abs_shap > 0 else 0.0
            })
        shap_features.sort(key=lambda x: abs(x['value']), reverse=True)

        prediction = {
            'survival_probability': survival_prob,
            'death_probability': death_prob,
            'shap_features': shap_features,
        }

        session['prediction'] = prediction
        session['form_data'] = raw_data

        return redirect(url_for('results'))

    except Exception as e:
        flash(f'Erreur lors de la prediction : {str(e)}', 'error')
        return redirect(url_for('predict'))

# ---------------------------------------------------------------------------
// ... (results, shap_page, admin routes unchanged)

# ---------------------------------------------------------------------------
// Startup
# ---------------------------------------------------------------------------
// ... (init_db, load_model, if __name__ == '__main__' unchanged)