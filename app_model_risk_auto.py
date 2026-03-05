# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    classification_report, accuracy_score
)

# =========================
# 🎨 CONFIGURATION GÉNÉRALE & STYLE
# =========================
st.set_page_config(page_title="Modélisation Universelle", layout="wide", page_icon="🤖")

# --- CSS personnalisé ---
st.markdown("""
    <style>
        body {
            background-color: #F7F9FB;
            color: #1E1E1E;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #005B96;
        }
        .stButton>button {
            background-color: #007ACC;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            border: none;
        }
        .stButton>button:hover {
            background-color: #005B96;
            color: white;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            padding-top: 1em;
            border-top: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🧭 EN-TÊTE
# =========================
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/02/Deep_Learning_Logo.png", width=90)
with col2:
    st.title("Modélisation Universelle des Données — IA Interprétable")

st.markdown("""
> 💡 *Cette application détecte automatiquement le type de tâche (régression ou classification),  
encode les variables, entraîne un modèle adapté et fournit des visualisations interactives.*
""")

st.markdown("---")

# =========================
# 📁 CHARGEMENT DES DONNÉES
# =========================
st.sidebar.header("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("📂 Charger un fichier Excel (.xlsx)", type=["xlsx"])

if not uploaded_file:
    st.info("➡️ Importez un fichier Excel pour commencer.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Erreur de lecture du fichier : {e}")
    st.stop()

st.subheader("📋 Aperçu du jeu de données")
st.dataframe(df.head())

# =========================
# 🎯 PARAMÈTRES DE MODÉLISATION
# =========================
all_columns = list(df.columns)
target = st.sidebar.selectbox("🎯 Choisir la variable cible", all_columns)
features = [c for c in all_columns if c != target]

if len(features) < 1:
    st.error("⚠️ Le dataset doit contenir au moins une variable explicative.")
    st.stop()

test_size = st.sidebar.slider("Taille du jeu de test", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# =========================
# 🧠 DÉTECTION DU TYPE DE TÂCHE
# =========================
if df[target].dtype == "object" or df[target].nunique() <= 10:
    task_type = "classification"
else:
    task_type = "regression"

st.sidebar.markdown(f"🧭 **Type de tâche détecté :** `{task_type}`")

# =========================
# ⚙️ CHOIX DU MODÈLE
# =========================
if task_type == "regression":
    model_name = st.sidebar.selectbox("Modèle de régression", ["Régression linéaire", "Random Forest"])
else:
    model_name = st.sidebar.selectbox("Modèle de classification", ["Régression logistique", "Random Forest"])

if model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 25)
    max_depth = st.sidebar.slider("max_depth (0 = None)", 0, 30, 0, 1)
    rf_max_depth = None if max_depth == 0 else max_depth

# =========================
# 🔄 PRÉTRAITEMENT & PIPELINE
# =========================
X = df[features].copy()
y = df[target].copy()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

if task_type == "regression":
    model = LinearRegression() if model_name == "Régression linéaire" else RandomForestRegressor(
        n_estimators=n_estimators, max_depth=rf_max_depth, random_state=random_state, n_jobs=-1)
else:
    model = LogisticRegression(max_iter=1000) if model_name == "Régression logistique" else RandomForestClassifier(
        n_estimators=n_estimators, max_depth=rf_max_depth, random_state=random_state, n_jobs=-1)

pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

# =========================
# 🚀 ENTRAÎNEMENT
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# =========================
# 📈 ÉVALUATION
# =========================
st.subheader("📊 Évaluation du modèle")

if task_type == "regression":
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.success(f"**MSE :** {mse:.4f} | **R² :** {r2:.4f}")
else:
    acc = accuracy_score(y_test, y_pred)
    st.success(f"**Accuracy :** {acc*100:.2f}%")

# =========================
# 📉 VISUALISATIONS
# =========================
st.subheader("📊 Visualisations")

if task_type == "regression":
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.7, color="#005B96")
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="red")
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Valeurs prédites")
    ax.set_title("Prédiction vs Réalité")
    st.pyplot(fig)
else:
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    st.pyplot(fig)
    st.write("**Rapport de classification :**")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# =========================
# 📌 IMPORTANCE DES VARIABLES
# =========================
st.subheader("📌 Importance des variables")
model_in_pipe = pipe.named_steps["model"]

if hasattr(model_in_pipe, "feature_importances_"):
    encoded_features = pipe.named_steps["preprocessor"].transformers_[1][1].named_steps["encoder"].get_feature_names_out(cat_cols)
    all_feature_names = np.concatenate([num_cols, encoded_features])
    importances = pd.Series(model_in_pipe.feature_importances_, index=all_feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    importances.plot(kind="barh", ax=ax, color="#007ACC")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
elif hasattr(model_in_pipe, "coef_"):
    st.write("Coefficients du modèle :")
    st.dataframe(pd.DataFrame(model_in_pipe.coef_).T)
else:
    st.info("Ce modèle ne fournit pas de coefficients interprétables.")

# =========================
# ⬇️ EXPORT DES RÉSULTATS
# =========================
st.subheader("📥 Export des résultats")
results_df = pd.DataFrame({"Réel": y_test, "Prédit": y_pred})

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Prédictions", index=False)
buf.seek(0)

st.download_button(
    "⬇️ Télécharger le rapport complet (Excel)",
    data=buf,
    file_name="rapport_modele_universel.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# =========================
# 👣 PIED DE PAGE
# =========================
st.markdown("""
<div class='footer'>
Développé avec ❤️ par <b>Dr. MOUALE </b> — Application IA universelle © 2025
</div>
""", unsafe_allow_html=True)
