# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

st.set_page_config(page_title="Analyse du Risque de Construction Immobilière", layout="wide")
st.title("Modélisation du Risque dans la Construction Immobilière — Application Interactive")

st.markdown("""
### 🏗️ Application de modélisation du risque de Construction Immobilière
Permet d’importer, explorer et visualiser des données, ainsi que de construire et évaluer des modèles prédictifs de risque de construction.  
L’application offre également la possibilité de définir, de manière **manuelle ou automatique**, un **diapason de risque de construction**, et de **classer les entreprises immobilières** selon leur niveau de risque : **critique**, **moyen** ou **excellent**.
""")


# =========================
# 🔧 Sidebar — Paramètres
# =========================
st.sidebar.header("Paramètres")

uploaded_file = st.sidebar.file_uploader("Fichier Excel (.xlsx)", type=["xlsx"])

test_size = st.sidebar.slider("Taille du jeu de test", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

model_name = st.sidebar.selectbox("Modèle", ["Régression linéaire", "Random Forest"])

if model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 25)
    max_depth = st.sidebar.slider("max_depth (0 = None)", 0, 30, 0, 1)
    rf_max_depth = None if max_depth == 0 else max_depth

st.sidebar.markdown("---")
th_mode = st.sidebar.radio("Seuils du diapason", ["Manuels", "Automatiques (quantiles)"])

# Seuils par défaut
low_default, high_default = 0.45, 0.65

if th_mode == "Manuels":
    low = st.sidebar.slider("Seuil bas (Critique < x)", 0.0, 1.0, low_default, 0.01)
    high = st.sidebar.slider("Seuil haut (Excellent ≥ x)", 0.0, 1.0, high_default, 0.01)
    if low >= high:
        st.sidebar.error("⚠️ Le seuil bas doit être strictement inférieur au seuil haut.")
else:
    q_low = st.sidebar.slider("Quantile bas", 0.0, 0.5, 0.35, 0.01)
    q_high = st.sidebar.slider("Quantile haut", 0.5, 1.0, 0.65, 0.01)
    if q_low >= q_high:
        st.sidebar.error("⚠️ Le quantile bas doit être < quantile haut.")

st.sidebar.markdown("---")
st.sidebar.caption("Colonnes attendues :")
st.sidebar.code(
    "Niveau ingénieurs\nNiveau techniciens\nExpérience ingénieurs\n"
    "Expérience techniciens\nTechnologie exploitée\nImpacc Climat\nExpérience entreprise\n"
    "indice_risk_const",
    language="markdown",
)

# =========================
# 🧾 Corps — Données
# =========================
def required_columns():
    return [
        "Niveau ingénieurs",
        "Niveau techniciens",
        "Expérience ingénieurs",
        "Expérience techniciens",
        "Technologie exploitée",
        "Impacc Climat",
        "Expérience entreprise",
        "indice_risk_const",
    ]

if not uploaded_file:
    st.info("➡️ Charge un fichier **.xlsx** dans la barre latérale pour commencer.")
    st.stop()

# Lecture
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Erreur de lecture du fichier : {e}")
    st.stop()

missing = [c for c in required_columns() if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes : {missing}")
    st.dataframe(df.head())
    st.stop()

st.subheader("Aperçu des données")
st.dataframe(df.head())

features = [
    "Niveau ingénieurs",
    "Niveau techniciens",
    "Expérience ingénieurs",
    "Expérience techniciens",
    "Technologie exploitée",
    "Impacc Climat",
    "Expérience entreprise",
]
target = "indice_risk_const"

X = df[features].copy()
y = df[target].astype(float).copy()

# Seuils automatiques si demandé
if th_mode == "Automatiques (quantiles)":
    low = float(y.quantile(q_low))
    high = float(y.quantile(q_high))

st.markdown(
    f"**Seuils utilisés** → "
    f"Critique: *x < {low:.3f}* | "
    f"Moyen: *{low:.3f} ≤ x < {high:.3f}* | "
    f"Excellent: *x ≥ {high:.3f}*"
)

# =========================
# 🧠 Entraînement
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

if model_name == "Régression linéaire":
    model = LinearRegression()
else:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=rf_max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
st.subheader("📈 Évaluation du modèle")
c1, c2 = st.columns(2)
with c1:
    st.metric("MSE (test)", f"{mse:.4f}")
with c2:
    st.write("Taille test:", len(y_test))

# =========================
# 🏷️ Classification
# =========================
def classify_risk(v: float) -> str:
    if v < low:
        return "Critique"
    elif v < high:
        return "Moyen"
    else:
        return "Excellent"

df_results = X_test.copy()
df_results["Risque_Réel"] = y_test
df_results["Risque_Prédit"] = y_pred
df_results["Diapason_Réel"] = df_results["Risque_Réel"].apply(classify_risk)
df_results["Diapason_Prédit"] = df_results["Risque_Prédit"].apply(classify_risk)

st.subheader("🗂️ Résultats (échantillon)")
st.dataframe(df_results.head(20))

# =========================
#  Visualisations
# =========================
st.subheader("📊 Visualisations")

# 1) Répartition par catégorie (barres)
colA, colB = st.columns(2)
with colA:
    st.write("Répartition par catégorie (prédite)")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    df_results["Diapason_Prédit"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Diapason")
    ax1.set_ylabel("Nombre")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig1)

with colB:
    st.write("Camembert des catégories (prédit)")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    df_results["Diapason_Prédit"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", startangle=90, shadow=True, ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

# 2) Scatter: réel vs prédit
st.write("Prédiction vs Valeurs réelles (idéal = ligne pointillée)")
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.scatter(y_test, y_pred, alpha=0.7)
min_v = min(y_test.min(), y_pred.min())
max_v = max(y_test.max(), y_pred.max())
ax3.plot([min_v, max_v], [min_v, max_v], linestyle="--")
ax3.set_xlabel("Risque réel")
ax3.set_ylabel("Risque prédit")
st.pyplot(fig3)

# 3) Matrice de confusion + rapport
y_test_cl = df_results["Diapason_Réel"]
y_pred_cl = df_results["Diapason_Prédit"]
labels = ["Critique", "Moyen", "Excellent"]

cm = confusion_matrix(y_test_cl, y_pred_cl, labels=labels)
st.write("Matrice de confusion (heatmap)")
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels, ax=ax4
)
ax4.set_xlabel("Prédit")
ax4.set_ylabel("Réel")
st.pyplot(fig4)

report_dict = classification_report(y_test_cl, y_pred_cl, target_names=labels, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
st.write("Rapport de classification")
st.dataframe(df_report)

# 4) Importance des variables / coefficients
st.subheader("Importance des facteurs")
if model_name == "Régression linéaire":
    coefs = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
    st.write("Coefficients (ordre décroissant par |coefficient|)")
    st.dataframe(coefs.rename("Coefficient"))
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    coefs.sort_values().plot(kind="barh", ax=ax5)
    ax5.set_xlabel("Coefficient")
    st.pyplot(fig5)
else:
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    st.write("Importances (Random Forest)")
    st.dataframe(importances.rename("Importance"))
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    importances.plot(kind="barh", ax=ax6)
    ax6.set_xlabel("Importance")
    st.pyplot(fig6)

# =========================
# ⬇ Export Excel
# =========================
st.subheader("⬇ Export des résultats")

def make_results_excel(_df_results: pd.DataFrame, _df_report: pd.DataFrame, _cm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _df_results.to_excel(writer, sheet_name="Résultats", index=True)
        _df_report.to_excel(writer, sheet_name="Classification_Report")
        df_cm = pd.DataFrame(
            _cm,
            index=[f"Réel_{l}" for l in labels],
            columns=[f"Prédit_{l}" for l in labels],
        )
        df_cm.to_excel(writer, sheet_name="Confusion_Matrix")
    buf.seek(0)
    return buf.read()

excel_bytes = make_results_excel(df_results, df_report, cm)

st.download_button(
    "📥 Télécharger le rapport complet (Excel)",
    data=excel_bytes,
    file_name="rapport_risque.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# =========================
# Astuces d’utilisation
# =========================
with st.expander("Astuces"):
    st.markdown(
        """
- Utilise **Random Forest** si la relation n’est pas linéaire ou si tu veux plus de robustesse.
- Les **seuils manuels** permettent d’aligner le diapason avec vos critères métiers.
- Les **seuils automatiques** (quantiles) s’adaptent à la distribution observée.
- Vérifie la **MSE** et la **matrice de confusion** pour juger la qualité.
- Les **coefficients / importances** aident à interpréter les facteurs clés.
"""
    )
