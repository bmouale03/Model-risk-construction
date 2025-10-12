import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# ----------------------------
# Interface
# ----------------------------
st.title("📊 Analyse de l'indice de risque de construction")
st.write("Importez votre fichier Excel contenant la variable cible `indice_risk_const`.")

# Paramètres de quantiles
q_low = st.slider("Quantile bas (Critique)", 0.0, 0.5, 0.33, 0.01)
q_high = st.slider("Quantile haut (Excellent)", 0.5, 1.0, 0.66, 0.01)

# Upload du fichier
uploaded_file = st.file_uploader("📁 Importer un fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # X / y
    X = df.drop(columns=['indice_risk_const'])
    y = df['indice_risk_const']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Seuils dynamiques
    q33 = pd.Series(y_pred).quantile(q_low)
    q66 = pd.Series(y_pred).quantile(q_high)

    st.subheader("📌 Seuils calculés")
    st.write(f"**Critique < {q33:.2f} | Moyen entre {q33:.2f} et {q66:.2f} | Excellent ≥ {q66:.2f}**")

    # Classification
    def classer(val):
        if val < q33:
            return "Critique"
        elif val < q66:
            return "Moyen"
        else:
            return "Excellent"

    y_pred_classes = [classer(val) for val in y_pred]
    y_real_classes = [classer(val) for val in y_test]

    # ----------------------------
    # Graphique : distribution
    # ----------------------------
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.hist(y_pred, bins=20, color='skyblue', edgecolor='black')
    ax1.axvline(q33, color='red', linestyle='--', label=f'Seuil Critique ({q_low*100:.0f}%)')
    ax1.axvline(q66, color='green', linestyle='--', label=f'Seuil Excellent ({q_high*100:.0f}%)')
    ax1.set_xlabel("Valeurs prédites")
    ax1.set_ylabel("Fréquence")
    ax1.set_title("Distribution des prédictions")
    ax1.legend()
    st.pyplot(fig1)

    # ----------------------------
    # Tableau de performances
    # ----------------------------
    acc_global = accuracy_score(y_real_classes, y_pred_classes) * 100
    rapport = classification_report(y_real_classes, y_pred_classes, output_dict=True)

    resultats = pd.DataFrame([{
        "Global (%)": round(acc_global, 2),
        "Critique (%)": round(rapport.get("Critique", {}).get("recall", 0) * 100, 2),
        "Moyen (%)": round(rapport.get("Moyen", {}).get("recall", 0) * 100, 2),
        "Excellent (%)": round(rapport.get("Excellent", {}).get("recall", 0) * 100, 2)
    }])

    st.subheader("📋 Tableau de performances")
    st.dataframe(resultats.style.background_gradient(axis=None, cmap="RdYlGn"))

    # ----------------------------
    # Matrice de confusion
    # ----------------------------
    labels = ["Critique", "Moyen", "Excellent"]
    cm = confusion_matrix(y_real_classes, y_pred_classes, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title("Matrice de confusion (%)")
    ax2.set_xlabel("Prédit")
    ax2.set_ylabel("Réel")
    st.pyplot(fig2)

    # ----------------------------
    # Génération PDF
    # ----------------------------
    if st.button("📑 Générer le rapport PDF"):
        # Sauvegarde des images
        fig1.savefig("graph_distribution.png")
        fig2.savefig("graph_confusion.png")

        doc = SimpleDocTemplate("rapport_performance.pdf", pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("<b>Rapport de performance - Indice de risque</b>", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Seuil Critique &lt; {q33:.2f} | Moyen entre {q33:.2f} et {q66:.2f} | Excellent ≥ {q66:.2f}", styles["Normal"]))
        elements.append(Spacer(1, 20))

        # Tableau
        table_data = [resultats.columns.tolist()] + resultats.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
            ("ALIGN", (0,0), (-1,-1), "CENTER")
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("<b>Distribution des prédictions</b>", styles["Heading2"]))
        elements.append(Image("graph_distribution.png", width=400, height=250))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("<b>Matrice de confusion</b>", styles["Heading2"]))
        elements.append(Image("graph_confusion.png", width=350, height=250))
        elements.append(Spacer(1, 20))

        doc.build(elements)
        st.success("✅ Rapport PDF généré : rapport_performance.pdf")
        with open("rapport_performance.pdf", "rb") as f:
            st.download_button("📥 Télécharger le rapport", data=f, file_name="rapport_performance.pdf")
