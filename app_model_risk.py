# -*- coding: utf-8 -*-
"""
Application Streamlit : Modèle de prédiction de risque de construction
Auteur : [Ton Nom]
Date : 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# ================================================
# Configuration de la page
# ================================================
st.set_page_config(
    page_title="Modèle de prédiction de Risque de Construction",
    layout="wide"
)

st.title("Modèle de Risque en Construction")
st.markdown("Analyse et prédiction du risque à partir de 7 facteurs explicatifs.")

# ================================================
# Fonctions utiles
# ================================================
def classify_risk(value):
    if value < 0.45:
        return "Critique"
    elif value < 0.65:
        return "Moyen"
    else:
        return "Excellent"

def generate_example_data(n=100):
    np.random.seed(42)
    data = pd.DataFrame({
        "Niveau ingénieurs": np.random.randint(1, 6, n),
        "Niveau techniciens": np.random.randint(1, 6, n),
        "Expérience ingénieurs": np.random.randint(1, 11, n),
        "Expérience techniciens": np.random.randint(1, 11, n),
        "Technologie exploitée": np.random.randint(1, 4, n),
        "Impacc Climat": np.random.randint(1, 4, n),
        "Expérience entreprise": np.random.randint(1, 21, n),
    })
    data["indice_risk_const"] = np.round(
        0.3 + 0.4 * np.random.rand(n), 3
    )
    return data

# ================================================
# Interface avec onglets
# ================================================
tab1, tab2 = st.tabs([" Analyse", "ℹ À propos du projet"])

with tab1:
    st.header("Importation et configuration")

    # Upload de fichier
    uploaded_file = st.file_uploader("📂 Importer un fichier Excel", type=["xlsx"])

    # Exemple de données
    if st.button(" Exemple de données"):
        df = generate_example_data()
        st.success("Exemple de données générées ")
    elif uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("Fichier chargé ")
    else:
        df = None

    if df is not None:
        st.subheader("Aperçu des données")
        st.dataframe(df.head())

        # Séparation des variables
        X = df[['Niveau ingénieurs', 'Niveau techniciens', 'Expérience ingénieurs',
                'Expérience techniciens', 'Technologie exploitée', 'Impacc Climat',
                'Expérience entreprise']]
        y = df['indice_risk_const']

        # Sélection du modèle
        model_choice = st.radio("Choisir un modèle :", ["Régression Linéaire", "Random Forest"])

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entraînement du modèle
        if model_choice == "Régression Linéaire":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Résultats
        st.subheader(" Résultats des prédictions")
        df_results = X_test.copy()
        df_results["Risque_Réel"] = y_test
        df_results["Risque_Prédit"] = y_pred
        df_results["Diapason"] = df_results["Risque_Prédit"].apply(classify_risk)

        st.dataframe(df_results.head(15))

        # Erreur
        mse = mean_squared_error(y_test, y_pred)
        st.metric("Erreur quadratique moyenne (MSE)", f"{mse:.4f}")

        # Visualisations
        st.subheader("Visualisations")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            df_results["Diapason"].value_counts().plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title("Répartition par catégorie de risque")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7, color="orange")
            ax.plot([0,1],[0,1], color="red", linestyle="--")
            ax.set_title("Prédiction vs Réel")
            st.pyplot(fig)

        # Matrice de confusion
        st.subheader(" Matrice de confusion")
        y_test_classes = y_test.apply(classify_risk)
        y_pred_classes = pd.Series(y_pred).apply(classify_risk)

        cm = confusion_matrix(y_test_classes, y_pred_classes, labels=["Critique", "Moyen", "Excellent"])

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Critique", "Moyen", "Excellent"],
                    yticklabels=["Critique", "Moyen", "Excellent"], ax=ax)
        ax.set_title("Matrice de confusion (Heatmap)")
        st.pyplot(fig)

        # Rapport de classification
        st.subheader("Rapport de classification")
        report_dict = classification_report(y_test_classes, y_pred_classes,
                                            target_names=["Critique", "Moyen", "Excellent"],
                                            output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        st.dataframe(df_report)

        # Export Excel
        st.subheader(" Exporter les résultats")
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_results.to_excel(writer, sheet_name="Résultats", index=False)
            pd.DataFrame(cm, 
                         index=["Réel_Critique", "Réel_Moyen", "Réel_Excellent"],
                         columns=["Prédit_Critique", "Prédit_Moyen", "Prédit_Excellent"]
                        ).to_excel(writer, sheet_name="Confusion_Matrix")
            df_report.to_excel(writer, sheet_name="Classification_Report")

        st.download_button(
            label="Télécharger le rapport Excel",
            data=output.getvalue(),
            file_name="rapport_risque.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ================================================
# Onglet 2 : À propos du projet
# ================================================
with tab2:
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
    """)

    st.subheader("Guide d’utilisation")

    with st.expander("ℹ️  Charger vos données"):
        st.markdown("""
        - Cliquez sur **Importer un fichier Excel** pour téléverser vos données.  
        - Votre fichier doit contenir les colonnes suivantes :  
          *Niveau ingénieurs, Niveau techniciens, Expérience ingénieurs, Expérience techniciens, Technologie exploitée, Impact Climat, Expérience entreprise, indice_risk_const*  
        """)

    with st.expander("2️⃣ Tester avec des données d’exemple"):
        st.markdown("""
        - Cliquez sur le bouton **Exemple de données** si vous n’avez pas encore de fichier.  
        - Cela génère un jeu de données simulées pour tester l’application.  
        """)

    with st.expander("3️⃣ Choisir un modèle de prédiction"):
        st.markdown("""
        - **Régression Linéaire** : simple et interprétable.  
        - **Random Forest** : plus robuste, capture mieux la complexité.  
        """)

    with st.expander("4️⃣ Lancer l’analyse"):
        st.markdown("""
        - Le modèle est entraîné sur **80% des données** et testé sur **20%**.  
        - Les prédictions sont affichées et classées en :  
          - **Critique** : risque < 0.45  
          - **Moyen** : 0.45 ≤ risque < 0.65  
          - **Excellent** : risque ≥ 0.65  
        """)

    with st.expander("5️⃣ Explorer les résultats"):
        st.markdown("""
        - **Histogrammes** et **camemberts** des classes  
        - **Scatter plot** : comparaison risques réels vs prédits  
        - **Matrice de confusion** : qualité de classification  
        """)

    with st.expander("6️⃣ Exporter le rapport"):
        st.markdown("""
        - Téléchargez un **fichier Excel complet** avec :  
          - Résultats prédits + catégories  
          - Matrice de confusion  
          - Indicateurs de performance (**Précision, Rappel, F1-score**)  
        """)

    st.markdown("""
    ---
    👨‍💻 Auteur : *Dr. MOUALE*  
    📅 Date : 2025  
    📜 Licence : Libre usage académique et professionnel  
    """)
