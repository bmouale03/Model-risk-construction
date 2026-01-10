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

st.title("Моделирование рисков в строительстве недвижимости")

tabs = st.tabs([
    "Моделирование риска",
    "Симуляция UCB",
    "ℹО проекте"
])

# =========================
#  ВКЛАДКА 1 — МОДЕЛИРОВАНИЕ РИСКА
# =========================
with tabs[0]:
    st.header("Моделирование и оценка риска")

    st.sidebar.header("Параметры модели")

    uploaded_file = st.sidebar.file_uploader(
        "Загрузить файл Excel (.xlsx)",
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
                "Нижний порог должен быть меньше верхнего."
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
                "Нижний квантиль должен быть меньше верхнего."
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
            "Загрузите файл **Excel (.xlsx)** для начала анализа."
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
    df_res["Реальный риск"] = y_test
    df_res["Предсказанный риск"] = y_pred
    df_res["Реальный класс"] = df_res["Реальный риск"].apply(classify)
    df_res["Предсказанный класс"] = df_res["Предсказанный риск"].apply(classify)

    # === Метрики качества ===
    acc = accuracy_score(
        df_res["Реальный класс"],
        df_res["Предсказанный класс"]
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_res["Реальный класс"],
        df_res["Предсказанный класс"],
        average="weighted"
    )

    st.subheader("Качество модели")
    st.markdown(f"""
    - **Общая точность:** {acc*100:.2f} %  
    - **Средняя взвешенная точность:** {precision*100:.2f} %  
    - **Средний взвешенный recall:** {recall*100:.2f} %  
    - **Средний F1-score:** {f1*100:.2f} %
    """)

    st.subheader("Фрагмент результатов")
    st.dataframe(df_res.head(20))
    # === VISUALISATIONS ===
    st.subheader("Визуализация моделей")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Распределение по предсказанным категориям")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df_res["Предсказанный класс"].value_counts().plot(kind="bar", color="skyblue", ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("Круговая диаграмма предсказанных категорий")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        df_res["Предсказанный класс"].value_counts().plot(
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
    sns.histplot(y_test, color="blue", label="Реальный риск", kde=True, ax=ax4)
    sns.histplot(y_pred, color="orange", label="Предсказанный риск", kde=True, ax=ax4)
    ax4.legend()
    st.pyplot(fig4)

    # Matrice de confusion
    labels = ["Критический", "Средний", "Отличный"]
    cm = confusion_matrix(df_res["Реальный класс"], df_res["Предсказанный класс"], labels=labels)
    st.write("Матрица ошибок")
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

    # === ЭКСПОРТ ОТЧЕТА ===
    def export_excel():
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_res.to_excel(
                writer,
                sheet_name="Результаты",
                index=False
            )
            df_report = pd.DataFrame(
                classification_report(
                    df_res["Реальный класс"],
                    df_res["Предсказанный класс"],
                    output_dict=True
                )
            ).transpose()
            df_report.to_excel(
                writer,
                sheet_name="Отчет_Классификации"
            )
            df_coeffs.to_excel(
                writer,
                sheet_name="Факторы_Коэффициенты",
                index=False
            )
        buf.seek(0)
        return buf

    st.download_button(
        "Скачать полный отчет (Excel)",
        data=export_excel(),
        file_name="risk_report_ru.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================
# ВКЛАДКА 2 — СИМУЛЯЦИЯ UCB
# =========================
with tabs[1]:
    st.header("Симуляция UCB — Стратегии строительства")

    n_strategies = st.slider(
        "Количество стратегий",
        2, 10, 5, 1
    )
    n_rounds = st.slider(
        "Количество симулированных проектов",
        100, 1000, 300, 50
    )
    c = st.slider(
        "Параметр доверия (c)",
        0.5, 3.0, 2.0, 0.1
    )

    true_means = [
        round(np.random.uniform(0.4, 0.9), 2)
        for _ in range(n_strategies)
    ]
    st.write(
        "Истинные характеристики стратегий (неизвестны модели):",
        true_means
    )

    rewards = np.zeros(n_strategies)
    counts = np.zeros(n_strategies)

    for t in range(1, n_rounds + 1):
        ucb_values = np.zeros(n_strategies)
        for i in range(n_strategies):
            if counts[i] > 0:
                mean_reward = rewards[i] / counts[i]
                confidence = c * math.sqrt(
                    2 * math.log(t) / counts[i]
                )
                ucb_values[i] = mean_reward + confidence
            else:
                ucb_values[i] = float("inf")

        strategy = np.argmax(ucb_values)
        reward = np.random.rand() < true_means[strategy]
        rewards[strategy] += reward
        counts[strategy] += 1

    st.subheader("📊 Результаты UCB")
    for i in range(n_strategies):
        st.write(
            f"Стратегия {i+1} : "
            f"{int(counts[i])} выборов — "
            f"Оценка среднего: {rewards[i]/counts[i]:.3f}"
        )

# =========================
# ℹ️ ВКЛАДКА 3 — О ПРОЕКТЕ
# =========================
with tabs[2]:
    st.header("ℹ️ О проекте")
    st.markdown("""
    Данное приложение предназначено для **анализа и прогнозирования
    строительных рисков** компании на основе 7 объясняющих факторов:

    - Уровень инженеров  
    - Уровень техников  
    - Опыт инженеров  
    - Опыт техников  
    - Используемая технология  
    - Климатическое воздействие  
    - Опыт компании  

    **Цели проекта:**
    - Оценка показателя `indice_risk_const`
    - Классификация компаний: **Критический / Средний / Отличный**
    - Интерактивная визуализация и экспортируемый отчет
    - Выбор оптимальной стратегии строительства, минимизирующей риск
    - Каждая стратегия может представлять метод управления,
      материал или логистический подход
    - Алгоритм UCB обучается проект за проектом,
      балансируя исследование и эксплуатацию

    ---
    **Автор:** Dr. MOUALE  
    **Версия:** Октябрь 2025  
    **Технологии:** Python, Streamlit, scikit-learn,matplotlib, seaborn, pandas
    """)
