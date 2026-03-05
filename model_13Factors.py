# ==========================================================
# НАУЧНАЯ ВЕРСИЯ — МОДЕЛИРОВАНИЕ СТРОИТЕЛЬНЫХ РИСКОВ
# ==========================================================

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Анализ строительных рисков",
    layout="wide"
)

st.title("Научное моделирование строительных рисков")

tabs = st.tabs([
    "Моделирование риска",
    "Симуляция UCB",
    "ℹ О проекте"
])

# ==========================================================
# TAB 1 — MODELING
# ==========================================================

with tabs[0]:

    st.header("Интеллектуальное прогнозирование риска")

    uploaded_file = st.file_uploader(
        "Загрузите Excel-файл (.xlsx)",
        type=["xlsx"]
    )

    if not uploaded_file:
        st.info("Загрузите файл для анализа.")
        st.stop()

    # ======================
    # DATA PREPARATION
    # ======================

    raw_df = pd.read_excel(uploaded_file)
    raw_df.columns = raw_df.iloc[0]
    df = raw_df.drop(index=0).reset_index(drop=True)

    df = df.loc[:, ~df.columns.str.contains("ИТОГО", case=False)]
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric)

    target_column = "Индекс риска(%)"

    if target_column not in df.columns:
        st.error("Не найден столбец 'Индекс риска(%)'")
        st.stop()

    df = df.rename(columns={target_column: "Индекс_качества"})

    st.success(f"Данные успешно загружены. Наблюдений: {len(df)}")

    # ======================
    # DATA PREVIEW
    # ======================

    st.subheader("Предварительный просмотр данных")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Первые строки датасета")
        st.dataframe(df.head())

    with col2:
        st.write("Статистическое описание")
        st.dataframe(df.describe())

    st.write("Размер датасета:", df.shape)

    features = [c for c in df.columns if c != "Индекс_качества"]
    target = "Индекс_качества"

    X = df[features]
    y = df[target]

    st.write(f"Количество факторов: {len(features)}")

    # ======================
    # SPLIT
    # ======================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ======================
    # MODELS
    # ======================

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    }

    results_summary = []

    for name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="neg_mean_squared_error"
        )

        results_summary.append({
            "Модель": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "CV_MSE_mean": -cv_scores.mean()
        })

    df_models = pd.DataFrame(results_summary).sort_values("RMSE")

    st.subheader("Сравнение моделей")
    st.dataframe(df_models)

    best_model_name = df_models.iloc[0]["Модель"]
    st.success(f"Лучшая модель: {best_model_name}")

    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # ======================
    # METRICS
    # ======================

    st.markdown(f"""
    ### Метрики лучшей модели
    - RMSE: {df_models.iloc[0]['RMSE']:.4f}
    - MAE: {df_models.iloc[0]['MAE']:.4f}
    - R²: {df_models.iloc[0]['R2']:.4f}
    - CV MSE: {df_models.iloc[0]['CV_MSE_mean']:.4f}
    """)

    # ======================
    # REAL VS PREDICTED
    # ======================

    st.subheader("Реальный риск vs Предсказанный риск")

    fig_risk, ax = plt.subplots(figsize=(7,6))

    sns.scatterplot(x=y_test, y=y_pred, ax=ax)

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--"
    )

    ax.set_xlabel("Реальный риск")
    ax.set_ylabel("Предсказанный риск")

    st.pyplot(fig_risk)

    # ======================
    # RESIDUAL PLOT
    # ======================

    st.subheader("Анализ остатков")

    residuals = y_test - y_pred

    fig_res, ax = plt.subplots(figsize=(7,5))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        ax=ax
    )

    ax.axhline(0, linestyle="--", color="red")

    ax.set_xlabel("Предсказанные значения")
    ax.set_ylabel("Остатки")

    st.pyplot(fig_res)

    # ======================
    # PERMUTATION IMPORTANCE
    # ======================

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_squared_error"
    )

    factor_importance = pd.DataFrame({
        "Фактор": features,
        "Влияние_на_MSE": -perm.importances_mean
    }).sort_values("Влияние_на_MSE", ascending=False)

    st.subheader("Permutation Importance")
    st.dataframe(factor_importance)

    fig_imp, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=factor_importance,
        x="Влияние_на_MSE",
        y="Фактор",
        ax=ax
    )

    st.pyplot(fig_imp)

    # ======================
    # FEATURE DISTRIBUTION
    # ======================

    st.subheader("Распределение факторов")

    selected_feature = st.selectbox(
        "Выберите фактор",
        features
    )

    fig_dist, ax = plt.subplots(figsize=(7,5))

    sns.histplot(
        df[selected_feature],
        kde=True,
        ax=ax
    )

    st.pyplot(fig_dist)

    # ======================
    # SHAP
    # ======================

    st.subheader("SHAP-анализ")

    if best_model_name == "Random Forest":

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer(X_test)
        shap_array = shap_values.values

        fig = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)

    else:

        scaler = best_model.named_steps["scaler"]
        linear_model = best_model.named_steps["model"]

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        explainer = shap.LinearExplainer(
            linear_model,
            X_train_scaled,
            feature_perturbation="interventional"
        )

        shap_array = explainer.shap_values(X_test_scaled)

        fig = plt.figure()
        shap.summary_plot(
            shap_array,
            X_test_scaled,
            feature_names=X.columns,
            show=False
        )
        st.pyplot(fig)

    # ======================
    # CLASSIFICATION
    # ======================

    low = y.quantile(0.35)
    high = y.quantile(0.65)

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

    acc = accuracy_score(
        df_res["Реальный класс"],
        df_res["Предсказанный класс"]
    )

    st.metric("Accuracy", f"{acc*100:.2f}%")

    # ======================
    # CONFUSION MATRIX
    # ======================

    st.subheader("Матрица ошибок")

    cm = confusion_matrix(
        df_res["Реальный класс"],
        df_res["Предсказанный класс"],
        labels=["Критический","Средний","Отличный"]
    )

    fig_cm, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Критический","Средний","Отличный"],
        yticklabels=["Критический","Средний","Отличный"],
        ax=ax
    )

    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Реальный класс")

    st.pyplot(fig_cm)

    # ======================
    # EXPORT
    # ======================

    def export_excel():

        buf = io.BytesIO()

        with pd.ExcelWriter(buf, engine="openpyxl") as writer:

            df_res.to_excel(writer, sheet_name="Результаты", index=False)
            df_models.to_excel(writer, sheet_name="Сравнение_Моделей", index=False)
            factor_importance.to_excel(writer, sheet_name="Ошибки_по_Факторам", index=False)

            pd.DataFrame(
                shap_array,
                columns=X.columns
            ).to_excel(writer, sheet_name="SHAP_Values", index=False)

        buf.seek(0)
        return buf

    st.download_button(
        "Скачать полный отчет",
        data=export_excel(),
        file_name="scientific_risk_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==========================================================
# TAB 2 — UCB
# ==========================================================

with tabs[1]:

    st.header("UCB-симуляция стратегий строительства")

    n_strategies = st.slider("Стратегии", 2, 10, 5)
    n_rounds = st.slider("Проекты", 100, 1000, 300)
    c = st.slider("Параметр доверия", 0.5, 3.0, 2.0)

    true_means = [round(np.random.uniform(0.4, 0.9), 2) for _ in range(n_strategies)]

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
                ucb_values[i] = float("inf")

        strategy = np.argmax(ucb_values)
        reward = np.random.rand() < true_means[strategy]

        rewards[strategy] += reward
        counts[strategy] += 1

    st.subheader("Результаты UCB")

    for i in range(n_strategies):
        st.write(
            f"Стратегия {i+1}: "
            f"{int(counts[i])} выборов — "
            f"Средняя оценка: {rewards[i]/counts[i]:.3f}"
        )

# ==========================================================
# TAB 3 — ABOUT
# ==========================================================

with tabs[2]:

    st.header("О проекте")

    st.markdown("""
    Научная система прогнозирования строительных рисков.

    Возможности:
    - Регрессионный анализ
    - Кросс-валидация
    - SHAP интерпретация
    - Permutation Importance
    - Confusion Matrix
    - Residual Analysis
    - UCB стратегии
    - Экспорт отчета

    Автор: Dr. MOUALE
    """)