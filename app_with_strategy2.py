#
# Добавить в загружаемый отчет ошибки по факторам
# на уровне страницы factors_coefficients.
#

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
# ОБЩАЯ КОНФИГУРАЦИЯ
# =========================
st.set_page_config(
    page_title="Анализ строительных рисков",
    layout="wide"
)

st.title("🏗️ Моделирование рисков в строительстве недвижимости")

tabs = st.tabs([
    "🧮 Моделирование риска",
    "🧠 Симуляция UCB",
    "ℹ️ О проекте"
])

# =========================
# 🧮 ВКЛАДКА 1 — МОДЕЛИРОВАНИЕ РИСКА
# =========================
with tabs[0]:
    st.header("🧮 Моделирование и оценка риска")

    st.sidebar.header("Параметры модели")

    uploaded_file = st.sidebar.file_uploader(
        "📂 Загрузить файл Excel (.xlsx)",
        type=["xlsx"]
    )
    test_size = st.sidebar.slider(
        "Размер тестовой выборки",
        0.1, 0.5, 0.2, 0.05
    )
    random_state = st.sidebar.number_input(
        "Random state",
        min_value=0,
        value=42,
        step=1
    )
    model_name = st.sidebar.selectbox(
        "Модель",
        ["Линейная регрессия", "Random Forest"]
    )

    if model_name == "Random Forest":
        n_estimators = st.sidebar.slider(
            "n_estimators",
            50, 500, 200, 25
        )
        max_depth = st.sidebar.slider(
            "max_depth (0 = без ограничения)",
            0, 30, 0, 1
        )
        rf_max_depth = None if max_depth == 0 else max_depth

    st.sidebar.markdown("---")
    th_mode = st.sidebar.radio(
        "Режим порогов",
        ["Ручной", "Автоматический (квантили)"]
    )

    low_default, high_default = 0.45, 0.65

    if th_mode == "Ручной":
        low = st.sidebar.slider(
            "Нижний порог (Критический < x)",
            0.0, 1.0, low_default, 0.01
        )
        high = st.sidebar.slider(
            "Верхний порог (Отличный ≥ x)",
            0.0, 1.0, high_default, 0.01
        )
        if low >= high:
            st.sidebar.error(
                "⚠️ Нижний порог должен быть меньше верхнего."
            )
    else:
        q_low = st.sidebar.slider(
            "Нижний квантиль",
            0.0, 0.5, 0.35, 0.01
        )
        q_high = st.sidebar.slider(
            "Верхний квантиль",
            0.5, 1.0, 0.65, 0.01
        )
        if q_low >= q_high:
            st.sidebar.error(
                "⚠️ Нижний квантиль должен быть меньше верхнего."
            )

    st.sidebar.markdown("---")
    st.sidebar.caption("Ожидаемые столбцы:")
    st.sidebar.code(
        "Уровень инженеров\n"
        "Уровень техников\n"
        "Опыт инженеров\n"
        "Опыт техников\n"
        "Используемая технология\n"
        "Климатическое воздействие\n"
        "Опыт компании\n"
        "Индекс_качества",
        language="markdown"
    )

    if not uploaded_file:
        st.info(
            "➡️ Загрузите файл **Excel (.xlsx)** для начала анализа."
        )
        st.stop()

    # === Загрузка данных ===
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    required_columns = [
        "Уровень_инженеров",
        "Уровень_техников",
        "Опыт_инженеров",
        "Опыт_техников",
        "Используемые_технологи",
        "Климатическое_воздействие",
        "Опыта_компании",
        "Индекс_качества",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        st.error(f"Отсутствующие столбцы: {missing}")
        st.dataframe(df.head())
        st.stop()

    st.subheader("Предварительный просмотр набора данных")
    st.dataframe(df.head())

    features = required_columns[:-1]
    target = "Индекс_качества"

    X = df[features].copy()
    y = df[target].astype(float).copy()

    if th_mode == "Автоматический (квантили)":
        low = float(y.quantile(q_low))
        high = float(y.quantile(q_high))

    st.markdown(
        f"""
        **Используемые пороги:**
        Критический < {low:.3f} |
        Средний : [{low:.3f}, {high:.3f}) |
        Отличный ≥ {high:.3f}
        """
    )

    # === Обучение модели ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    if model_name == "Линейная регрессия":
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
    st.metric("MSE (тест)", f"{mse:.4f}")

    # === Классификация ===
    def classify(v):
        if v < low:
            return "Критический"
        elif v < high:
            return "Средний"
        else:
            return "Отличный"

    df_res = X_test.copy()
    df_res["Risque_Réel"] = y_test
    df_res["Risque_Prédit"] = y_pred
    df_res["Classe_Réelle"] = df_res["Risque_Réel"].apply(classify)
    df_res["Classe_Prédit"] = df_res["Risque_Prédit"].apply(classify)

    # === Метрики качества ===
    acc = accuracy_score(df_res["Classe_Réelle"], df_res["Classe_Prédit"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_res["Classe_Réelle"], df_res["Classe_Prédit"], average="weighted"
    )

    st.subheader("Качество модели")
    st.markdown(f"""
    - **Общая точность:** {acc*100:.2f} %  
    - **Средняя взвешенная точность:** {precision*100:.2f} %  
    - **Средний взвешенный recall:** {recall*100:.2f} %  
    - **Средний F1-score:** {f1*100:.2f} %
    """)

    fig_perf, ax_perf = plt.subplots(figsize=(6, 0.8))
    ax_perf.barh(["Качество модели"], [acc * 100], color="seagreen")
    ax_perf.set_xlim(0, 100)
    ax_perf.set_xlabel("Процент правильных классификаций")
    for i, v in enumerate([acc * 100]):
        ax_perf.text(v + 1, i, f"{v:.1f}%", va="center")
    st.pyplot(fig_perf)

    st.subheader("Фрагмент результатов")
    st.dataframe(df_res.head(20))

  # === VISUALISATIONS ===
    st.subheader("Визуализация моделей")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Распределение по предсказанным категориям")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df_res["Classe_Prédit"].value_counts().plot(kind="bar", color="skyblue", ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("Круговая диаграмма предсказанных категорий")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        df_res["Classe_Prédit"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", startangle=90, shadow=True, ax=ax2
        )
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # Risque réel vs prédit
    st.write("Реальный риск против предсказуемого")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(y_test, y_pred, alpha=0.7, color="orange", edgecolor="black")
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    ax3.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black")
    st.pyplot(fig3)

    # Histogramme
    st.write("Распределение реального и предсказуемого риска")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(y_test, color="blue", label="Risque réel", kde=True, ax=ax4)
    sns.histplot(y_pred, color="orange", label="Risque prédit", kde=True, ax=ax4)
    ax4.legend()
    st.pyplot(fig4)

    # Matrice de confusion
    labels = ["Critique", "Moyen", "Excellent"]
    cm = confusion_matrix(df_res["Classe_Réelle"], df_res["Classe_Prédit"], labels=labels)
    st.write("Matrice de confusion")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax5)
    st.pyplot(fig5)

    

   
