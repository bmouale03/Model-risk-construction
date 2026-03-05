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
Importe, explore et visualise des données, puis construis un modèle prédictif de risque de construction.
Tu peux aussi définir le **diapason de risque** (Critique, Moyen, Excellent) et classer les entreprises automatiquement.
""")

# ===========================================================
# 🔧 SIDEBAR
# ===========================================================
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
low_default, high_default = 0.45, 0.65

if th_mode == "Manuels":
    low = st.sidebar.slider("Seuil bas (Critique < x)", 0.0, 1.0, low_default, 0.01)
    high = st.sidebar.slider("Seuil haut (Excellent ≥ x)", 0.0, 1.0, high_default, 0.01)
    if low >= high:
        st.sidebar.error("⚠️ Le seuil bas doit être < au seuil haut.")
else:
    q_low = st.sidebar.slider("Quantile bas", 0.0, 0.5, 0.35, 0.01)
    q_high = st.sidebar.slider("Quantile haut", 0.5, 1.0, 0.65, 0.01)
    if q_low >= q_high:
        st.sidebar.error("⚠️ Le quantile bas doit être < quantile haut.")

# ===========================================================
# 📘 Données
# ===========================================================
def required_columns():
    return [
        "Niveau ingénieurs", "Niveau techniciens",
        "Expérience ingénieurs", "Expérience techniciens",
        "Technologie exploitée", "Impact Climat",
        "Expérience entreprise", "indice_risk_const"
    ]

if not uploaded_file:
    st.info("➡️ Charge un fichier Excel (.xlsx) dans la barre latérale pour commencer.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors de la lecture du fichier : {e}")
    st.stop()

missing = [c for c in required_columns() if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes : {missing}")
    st.dataframe(df.head())
    st.stop()

st.subheader("Aperçu des données")
st.dataframe(df.head())

features = [
    "Niveau ingénieurs", "Niveau techniciens",
    "Expérience ingénieurs", "Expérience techniciens",
    "Technologie exploitée", "Impact Climat",
    "Expérience entreprise"
]
target = "indice_risk_const"

X = df[features].copy()
y = df[target].astype(float).copy()

if th_mode == "Automatiques (quantiles)":
    low = float(y.quantile(q_low))
    high = float(y.quantile(q_high))

st.markdown(
    f"**Seuils utilisés** → "
    f"Critique: *x < {low:.3f}* | "
    f"Moyen: *{low:.3f} ≤ x < {high:.3f}* | "
    f"Excellent: *x ≥ {high:.3f}*"
)

# ===========================================================
# 🧠 Modélisation
# ===========================================================
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
        n_jobs=-1
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
st.subheader("📈 Évaluation du modèle")
st.metric("MSE (test)", f"{mse:.4f}")

# ===========================================================
# 🏷️ Classification
# ===========================================================
def classify_risk(v):
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

# ===========================================================
# 📊 Visualisations
# ===========================================================
st.subheader("📊 Visualisations")

colA, colB = st.columns(2)
with colA:
    fig1, ax1 = plt.subplots()
    df_results["Diapason_Prédit"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_title("Répartition par catégorie prédite")
    st.pyplot(fig1)
with colB:
    fig2, ax2 = plt.subplots()
    df_results["Diapason_Prédit"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
    ax2.set_ylabel("")
    ax2.set_title("Camembert des catégories")
    st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.scatter(y_test, y_pred, alpha=0.7)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
ax3.set_xlabel("Risque réel")
ax3.set_ylabel("Risque prédit")
ax3.set_title("Risque réel vs prédit")
st.pyplot(fig3)

labels = ["Critique", "Moyen", "Excellent"]
cm = confusion_matrix(df_results["Diapason_Réel"], df_results["Diapason_Prédit"], labels=labels)
fig4, ax4 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax4)
ax4.set_xlabel("Prédit")
ax4.set_ylabel("Réel")
ax4.set_title("Matrice de confusion")
st.pyplot(fig4)

report_dict = classification_report(
    df_results["Diapason_Réel"],
    df_results["Diapason_Prédit"],
    target_names=labels,
    output_dict=True
)
df_report = pd.DataFrame(report_dict).transpose()
st.dataframe(df_report)

st.subheader("Importance des facteurs")
if model_name == "Régression linéaire":
    coefs = pd.Series(model.coef_, index=features)
    fig5, ax5 = plt.subplots()
    coefs.sort_values().plot(kind="barh", ax=ax5)
    ax5.set_title("Coefficients du modèle linéaire")
    st.pyplot(fig5)
else:
    importances = pd.Series(model.feature_importances_, index=features)
    fig6, ax6 = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax6)
    ax6.set_title("Importances des variables (Random Forest)")
    st.pyplot(fig6)

# ===========================================================
# 📉 Heatmap des erreurs par facteur
# ===========================================================
st.subheader("🔥 Analyse des erreurs par facteur")

df_errors = X_test.copy()
df_errors["Risque_Réel"] = y_test
df_errors["Risque_Prédit"] = y_pred
df_errors["Erreur_Risque"] = y_test - y_pred

corr = df_errors.corr(numeric_only=True)["Erreur_Risque"].sort_values(ascending=False).to_frame()

st.markdown("**Corrélation des erreurs avec les variables explicatives**")
fig_err, ax_err = plt.subplots(figsize=(6, 4))
sns.heatmap(corr.T, annot=True, cmap="coolwarm", center=0, ax=ax_err)
ax_err.set_title("Heatmap des erreurs par facteur (corrélation)")
st.pyplot(fig_err)

# ===========================================================
# 📤 Export Excel
# ===========================================================
st.subheader("⬇ Export du rapport complet")

def make_results_excel():
    buf = io.BytesIO()

    if model_name == "Régression linéaire":
        df_coeffs = pd.DataFrame({"Facteur": features, "Coefficient": model.coef_})
        df_coeffs.loc[len(df_coeffs)] = ["Constante (intercept)", model.intercept_]
    else:
        df_coeffs = pd.DataFrame({"Facteur": features, "Importance": model.feature_importances_})

    df_errors_stats = df_errors.describe().T
    df_errors_stats["MAE"] = df_errors["Erreur_Risque"].abs().mean()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Résultats")
        df_report.to_excel(writer, sheet_name="Classification_Report")
        pd.DataFrame(cm, index=labels, columns=labels).to_excel(writer, sheet_name="Confusion_Matrix")
        df_coeffs.to_excel(writer, sheet_name="Facteurs_Coefficients", index=False)
        df_errors_stats.to_excel(writer, sheet_name="Erreurs_Facteurs")

    buf.seek(0)
    return buf.read()

excel_bytes = make_results_excel()

st.download_button(
    "📥 Télécharger le rapport Excel complet",
    data=excel_bytes,
    file_name="rapport_risque.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ===========================================================
# ℹ️ À propos du projet
# ===========================================================
st.header("À propos du projet")
st.markdown("""
Cette application Streamlit a été conçue pour **analyser et prédire le risque de construction** 
d'une entreprise en fonction de 7 facteurs explicatifs :

- Niveau ingénieurs  
- Niveau techniciens  
- Expérience ingénieurs  
- Expérience techniciens  
- Technologie exploitée  
- Impact Climat  
- Expérience entreprise  

**Objectifs :**  
- Fournir une estimation de l’`indice_risk_const`  
- Classer les entreprises selon un diapason **Critique / Moyen / Excellent**  
- Offrir des visualisations interactives et un rapport téléchargeable  

---

👨‍💻 **Auteur :** Dr. MOUALE  
🏛️ **Institution :** Université Nangui Abrogoua, Abidjan — Côte d’Ivoire  
📅 **Version :** Octobre 2025  
""")
