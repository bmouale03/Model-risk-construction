#
#Ajoutes dans le rapport  téléchargeable les erreurs par facteur au niveau de page facteurs_coefficients.

import io 
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import math

# =========================
# CONFIGURATION GÉNÉRALE
# =========================
st.set_page_config(page_title="Analyse du Risque de Construction", layout="wide")

st.title("🏗️ Modélisation du Risque dans la Construction Immobilière")

tabs = st.tabs(["🧮 Modélisation du risque", "🧠 Simulation UCB", "ℹ️ À propos du projet"])

# =========================
# 🧮 ONGLET 1 — Modélisation du Risque
# =========================
with tabs[0]:
    st.header("🧮 Modélisation et évaluation du risque")

    st.sidebar.header("Paramètres du modèle")

    uploaded_file = st.sidebar.file_uploader("📂 Importer un fichier Excel (.xlsx)", type=["xlsx"])
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
            st.sidebar.error("⚠️ Le seuil bas doit être inférieur au seuil haut.")
    else:
        q_low = st.sidebar.slider("Quantile bas", 0.0, 0.5, 0.35, 0.01)
        q_high = st.sidebar.slider("Quantile haut", 0.5, 1.0, 0.65, 0.01)
        if q_low >= q_high:
            st.sidebar.error("⚠️ Le quantile bas doit être inférieur au quantile haut.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Colonnes attendues :")
    st.sidebar.code(
        "Niveau ingénieurs\nNiveau techniciens\nExpérience ingénieurs\n"
        "Expérience techniciens\nTechnologie exploitée\nImpact Climat\nExpérience entreprise\n"
        "indice_risk_const",
        language="markdown",
    )

    if not uploaded_file:
        st.info("➡️ Importez un fichier **Excel (.xlsx)** pour démarrer l’analyse.")
        st.stop()

    # === Lecture des données ===
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        st.stop()

    required_columns = [
        "Niveau ingénieurs",
        "Niveau techniciens",
        "Expérience ingénieurs",
        "Expérience techniciens",
        "Technologie exploitée",
        "Impact Climat",
        "Expérience entreprise",
        "indice_risk_const",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {missing}")
        st.dataframe(df.head())
        st.stop()

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())

    features = required_columns[:-1]
    target = "indice_risk_const"

    X = df[features].copy()
    y = df[target].astype(float).copy()

    if th_mode == "Automatiques (quantiles)":
        low = float(y.quantile(q_low))
        high = float(y.quantile(q_high))

    st.markdown(
        f"**Seuils utilisés :** Critique < {low:.3f} | Moyen : [{low:.3f}, {high:.3f}) | Excellent ≥ {high:.3f}"
    )

    # === Entraînement du modèle ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_name == "Régression linéaire":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=rf_max_depth, random_state=random_state, n_jobs=-1
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.metric("MSE (test)", f"{mse:.4f}")

    # === Classification ===
    def classify(v):
        if v < low:
            return "Critique"
        elif v < high:
            return "Moyen"
        else:
            return "Excellent"

    df_res = X_test.copy()
    df_res["Risque_Réel"] = y_test
    df_res["Risque_Prédit"] = y_pred
    df_res["Classe_Réelle"] = df_res["Risque_Réel"].apply(classify)
    df_res["Classe_Prédit"] = df_res["Risque_Prédit"].apply(classify)

    # === Indicateurs de performance ===
    acc = accuracy_score(df_res["Classe_Réelle"], df_res["Classe_Prédit"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_res["Classe_Réelle"], df_res["Classe_Prédit"], average="weighted"
    )

    st.subheader("📈 Performance du modèle")
    st.markdown(f"""
    - **Exactitude globale :** {acc*100:.2f} %  
    - **Précision moyenne pondérée :** {precision*100:.2f} %  
    - **Rappel moyen pondéré :** {recall*100:.2f} %  
    - **F1-score moyen pondéré :** {f1*100:.2f} %
    """)

    fig_perf, ax_perf = plt.subplots(figsize=(6, 0.8))
    ax_perf.barh(["Performance globale"], [acc * 100], color="seagreen")
    ax_perf.set_xlim(0, 100)
    ax_perf.set_xlabel("Pourcentage de bonnes classifications")
    for i, v in enumerate([acc * 100]):
        ax_perf.text(v + 1, i, f"{v:.1f}%", va="center")
    st.pyplot(fig_perf)

    st.subheader("🗂️ Extrait des résultats")
    st.dataframe(df_res.head(20))

    # === VISUALISATIONS ===
    st.subheader("📊 Visualisations du modèle")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Répartition par catégorie prédite")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df_res["Classe_Prédit"].value_counts().plot(kind="bar", color="skyblue", ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("Camembert des catégories prédites")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        df_res["Classe_Prédit"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", startangle=90, shadow=True, ax=ax2
        )
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # Risque réel vs prédit
    st.write("Risque réel vs prédit")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(y_test, y_pred, alpha=0.7, color="orange", edgecolor="black")
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    ax3.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black")
    st.pyplot(fig3)

    # Histogramme
    st.write("Distribution du risque réel et prédit")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(y_test, color="blue", label="Risque réel", kde=True, ax=ax4)
    sns.histplot(y_pred, color="orange", label="Risque prédit", kde=True, ax=ax4)
    ax4.legend()
    st.pyplot(fig4)
    #st.write("Classes réelles présentes :", df_res["Classe_Réelle"].unique())
    #st.write("Classes prédites présentes :", df_res["Classe_Prédit"].unique())


    # Matrice de confusion
    labels = ["Critique", "Moyen", "Excellent"]
    cm = confusion_matrix(df_res["Classe_Réelle"], df_res["Classe_Prédit"], labels=labels)
    st.write("Matrice de confusion")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax5)
    st.pyplot(fig5)
   # === ВАЖНОСТЬ ПЕРЕМЕННЫХ ===
    st.subheader("Важность факторов")

    if model_name == "Линейная регрессия":
        df_coeffs = pd.DataFrame({
            "Фактор": features,
            "Коэффициент": model.coef_,
        })
        df_coeffs["Ошибка (средняя абсолютная)"] = np.abs(y_test - y_pred).mean()
        df_coeffs.loc[len(df_coeffs)] = [
            "Константа (intercept)",
            model.intercept_,
            np.nan
        ]
    else:
        df_coeffs = pd.DataFrame({
            "Фактор": features,
            "Важность (Random Forest)": model.feature_importances_
        })
      # === ВАЖНОСТЬ ПЕРЕМЕННЫХ ===
    st.subheader("🧩 Важность факторов")

    if model_name == "Линейная регрессия":
        df_coeffs = pd.DataFrame({
            "Фактор": features,
            "Коэффициент": model.coef_,
        })
        df_coeffs["Ошибка (средняя абсолютная)"] = np.abs(y_test - y_pred).mean()
        df_coeffs.loc[len(df_coeffs)] = [
            "Константа (intercept)",
            model.intercept_,
            np.nan
        ]
    else:
        df_coeffs = pd.DataFrame({
            "Фактор": features,
            "Важность (Random Forest)": model.feature_importances_
        })

# =========================
# 🧠 ONGLET 2 — Simulation UCB
# =========================
with tabs[1]:
    st.header("🧠 Simulation UCB — Stratégies de Construction")
    n_strategies = st.slider("Nombre de stratégies", 2, 10, 5, 1)
    n_rounds = st.slider("Nombre de projets simulés", 100, 1000, 300, 50)
    c = st.slider("Paramètre de confiance (c)", 0.5, 3.0, 2.0, 0.1)

    true_means = [round(np.random.uniform(0.4, 0.9), 2) for _ in range(n_strategies)]
    st.write("Performances réelles (inconnues du modèle) :", true_means)

    rewards = np.zeros(n_strategies)
    counts = np.zeros(n_strategies)

    for t in range(1, n_rounds + 1):
        ucb_values = np.zeros(n_strategies)
        for i in range(n_strategies):
            if counts[i] > 0:
                mean_reward = rewards[i] / counts[i]
                confidence = c * math.sqrt(2 * math.log(t) / counts[i])
                ucb_values[i] = mean_reward + confidence
            else:
                ucb_values[i] = float('inf')
        strategy = np.argmax(ucb_values)
        reward = np.random.rand() < true_means[strategy]
        rewards[strategy] += reward
        counts[strategy] += 1

    st.subheader("📊 Résultats UCB")
    for i in range(n_strategies):
        st.write(f"Stratégie {i+1} : {int(counts[i])} sélections — Moyenne estimée : {rewards[i]/counts[i]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, n_strategies + 1), counts, color="skyblue", edgecolor="black")
    ax.set_xlabel("Stratégie")
    ax.set_ylabel("Nombre de sélections")
    st.pyplot(fig)

# =========================
# ℹ️ ONGLET 3 — À propos du projet
# =========================
with tabs[2]:
    st.header("ℹ️ À propos du projet")
    st.markdown("""
    Cette application a été conçue pour **analyser et prédire le risque de construction** 
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
    - choisir parmi plusieurs stratégies de construction celle qui minimise le risque (ou maximise la performance)
    - Chaque stratégie pourrait représenter une méthode de gestion de chantier, un type de matériau, ou une approche logistique.
    - Chaque stratégie est testée plusieurs fois, mais celles qui ont de meilleures performances et plus de confiance deviennent dominantes.
    - L’UCB(Upper Confidence Bound) apprend, projet après projet, quelle stratégie minimise le risque tout en explorant intelligemment les autres options.
    ---
    👨‍🔬 **Auteur :** Dr. MOUALE  
    📅 Version : Octobre 2025  
    ⚙️ **Technologies utilisées :** Python, Streamlit, scikit-learn, matplotlib, seaborn, pandas  
    """)
