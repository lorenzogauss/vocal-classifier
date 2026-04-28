import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- 1. Caricamento e Pulizia ---
try:
    df = pd.read_csv('training_set.csv')

    if 'label' not in df.columns:
        cols = list(df.columns)
        cols[-1] = 'label'
        df.columns = cols

    print(f"Dataset caricato. Dimensioni: {df.shape}")
    print(f"Classi trovate: {df['label'].unique()}")
except Exception as e:
    print(f"Errore durante il caricamento: {e}")
    exit()

# --- 2. Preparazione Dati ---
X = df.drop('label', axis=1)
y = df['label']


# Split Training e Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Configurazione Random Forest ---
# n_estimators=200
# max_features='sqrt'
# min_samples_leaf=2
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15,
    min_samples_leaf=2, 
    random_state=42, 
    n_jobs=-1
)

print("Addestramento in corso...")
model.fit(X_train, y_train)

# --- 4. Valutazione ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n" + "="*30)
print(f" RISULTATI ADDESTRAMENTO")
print(f"="*30)
print(f"Accuratezza: {acc*100:.2f}%")
print("\nReport dettagliato:")
print(classification_report(y_test, y_pred))

# --- 5. Salvataggio ---
joblib.dump(model, 'classificatore_voci.pkl')
joblib.dump(model.classes_, 'classi_nomi.pkl') 

print("\nModello salvato con successo!")