import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

def train_final_model(X_train, y_train):
    # 1. Équilibrage
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # 2. Entraînement du champion (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    
    # 3. Sauvegarde du modèle
    joblib.dump(model, 'models/final_model.joblib')
    print("Modèle Random Forest entraîné et sauvegardé dans models/")
    return model