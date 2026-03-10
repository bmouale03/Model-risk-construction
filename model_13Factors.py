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

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Анализ строительных рисков",
    layout="wide"
)

st.title("Научное моделирование строительных рисков В Кот Д'Ивуаре")

tabs = st.tabs([
    "Моделирование риска",
    "Симуляция UCB",
    "Информации О проекте"
])

# ==========================================================
# TAB 1 — MODELING
# ==========================================================

with tabs[0]:

    st.header("Интеллектуальное прогнозирование риска В Кот Д'Ивуаре")

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

    # ==========================================================
    # НАУЧНЫЙ АНАЛИЗ
    # ==========================================================

    st.header("Научный анализ распределений")

    q25 = y.quantile(0.25)
    q35 = y.quantile(0.35)
    q50 = y.quantile(0.50)
    q65 = y.quantile(0.65)
    q75 = y.quantile(0.75)

    quant_table = pd.DataFrame({
        "Квантиль": ["Q25","Q35","Median","Q65","Q75"],
        "Значение": [q25,q35,q50,q65,q75]
    })

    st.subheader("Пороговые значения индекса риска")
    st.dataframe(quant_table)

    # --------------------
    # Distribution with Legend
    # --------------------

    fig_dist, ax = plt.subplots(figsize=(8,6))
    sns.histplot(y, bins=20, kde=True, ax=ax)

    # Линии для квантилей
    ax.axvline(q35, color="red", linestyle="--", label=f"Quantile 33% ({q35:.2f})")
    ax.axvline(q65, color="green", linestyle="--", label=f"Quantile 66% ({q65:.2f})")

    ax.set_xlabel("Индекс риска")
    ax.set_ylabel("Частота")

    # Легенда
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    st.pyplot(fig_dist)
    plt.close(fig_dist)

    # --------------------
    # Boxplot
    # --------------------

    fig_box, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(y=y, ax=ax)
    st.pyplot(fig_box)
    plt.close(fig_box)

    # --------------------
    # Correlation Heatmap
    # --------------------

    corr = df.corr()
    fig_corr, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # --------------------
    # Factor Impact
    # --------------------

    cols = 3
    rows = int(np.ceil(len(features)/cols))
    fig_factors, axes = plt.subplots(rows, cols, figsize=(15, rows*4))
    axes = axes.flatten()
    for i, f in enumerate(features):
        sns.regplot(x=df[f], y=df[target], ax=axes[i], scatter_kws={"alpha":0.6})
        axes[i].set_title(f)
    for j in range(i+1,len(axes)):
        fig_factors.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig_factors)
    plt.close(fig_factors)

    # ==========================================================
    # SPLIT
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==========================================================
    # MODELS
    # ==========================================================

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
            n_estimators=300, random_state=42, n_jobs=-1
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
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
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

    # ==========================================================
    # MODEL COEFFICIENTS
    # ==========================================================

    st.subheader("Коэффициенты модели")

    if best_model_name in ["Linear Regression","Ridge Regression"]:
        linear_model = best_model.named_steps["model"]
        intercept = linear_model.intercept_
        coefficients = pd.DataFrame({
            "Фактор":["B0 (Intercept)"] + features,
            "Коэффициент":np.concatenate(([intercept], linear_model.coef_))
        })
        st.write("Константа B0:",intercept)
        st.dataframe(coefficients)
        equation = f"Risk = {intercept:.4f}"
        for coef, feature in zip(linear_model.coef_, features):
            equation += f" + ({coef:.4f} * {feature})"
    else:
        coefficients = pd.DataFrame({
            "Фактор": features,
            "Важность": best_model.feature_importances_
        })
        st.dataframe(coefficients)
        equation = "Tree based model — equation not linear"

    # ==========================================================
    # FACTOR ERRORS
    # ==========================================================

    st.subheader("Ошибки по факторам (MSE, MAE)")

    factor_errors = []
    for f in features:
        X_f_train = X_train[[f]]
        X_f_test = X_test[[f]]
        fm = LinearRegression() if best_model_name in ["Linear Regression","Ridge Regression"] else RandomForestRegressor(n_estimators=200, random_state=42)
        fm.fit(X_f_train, y_train)
        y_f_pred = fm.predict(X_f_test)
        mse_f = mean_squared_error(y_test, y_f_pred)
        mae_f = mean_absolute_error(y_test, y_f_pred)
        factor_errors.append({"Фактор": f, "MSE": mse_f, "MAE": mae_f})

    df_factor_errors = pd.DataFrame(factor_errors).sort_values("MSE")
    st.dataframe(df_factor_errors)

    # ==========================================================
    # SHAP ANALYSIS
    # ==========================================================

    st.subheader("SHAP-анализ факторов")
    shap_summary_df = None
    shap_values_df = None

    try:
        if best_model_name in ["Linear Regression","Ridge Regression"]:
            explainer = shap.Explainer(best_model.named_steps["model"], X_train)
        else:
            explainer = shap.TreeExplainer(best_model)

        shap_values = explainer(X_test)

        # SHAP Beeswarm
        fig_shap = plt.figure(figsize=(10,6))
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        # Таблица сводки SHAP
        shap_summary_df = pd.DataFrame(
            np.abs(shap_values.values).mean(0),
            index=features,
            columns=["Mean_Abs_SHAP"]
        ).sort_values("Mean_Abs_SHAP", ascending=False)
        st.subheader("SHAP значения (сводка)")
        st.dataframe(shap_summary_df)

        # SHAP Values для каждой строки (для отчета)
        shap_values_df = pd.DataFrame(shap_values.values, columns=features)
        shap_values_df.insert(0, "Index", X_test.index)
    except Exception as e:
        st.error(f"Ошибка при расчете SHAP: {e}")

    # ==========================================================
    # METRICS
    # ==========================================================

    st.metric("RMSE", df_models.iloc[0]['RMSE'])
    st.metric("R2", df_models.iloc[0]['R2'])

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================

    low = y.quantile(0.35)
    high = y.quantile(0.65)

    def classify(v):
        if v < low: return "Критический"
        elif v < high: return "Средний"
        else: return "Отличный"

    df_res = X_test.copy()
    df_res["Реальный риск"] = y_test
    df_res["Предсказанный риск"] = y_pred
    df_res["Реальный класс"] = df_res["Реальный риск"].apply(classify)
    df_res["Предсказанный класс"] = df_res["Предсказанный риск"].apply(classify)
    acc = accuracy_score(df_res["Реальный класс"], df_res["Предсказанный класс"])
    st.metric("Accuracy", f"{acc*100:.2f}%")

    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================

    cm = confusion_matrix(
        df_res["Реальный класс"],
        df_res["Предсказанный класс"],
        labels=["Критический","Средний","Отличный"]
    )

    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # ==========================================================
    # EXPORT EXCEL
    # ==========================================================

    def export_excel():
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_res.to_excel(writer,"Результаты",index=False)
            df_models.to_excel(writer,"Сравнение_Моделей",index=False)
            coefficients.to_excel(writer,"Model_Coefficients",index=False)
            df_factor_errors.to_excel(writer,"Factor_Errors",index=False)
            if shap_values_df is not None:
                shap_values_df.to_excel(writer,"SHAP_Values",index=False)
            eq_df = pd.DataFrame({"Model Equation":[equation]})
            eq_df.to_excel(writer,"Model_Equation",index=False)
        buf.seek(0)
        return buf

    st.download_button(
        "Скачать полный отчет",
        data=export_excel(),
        file_name="scientific_risk_report.xlsx"
    )

# ==========================================================
# TAB 2 — UCB
# ==========================================================

with tabs[1]:
    st.header("UCB симуляция")
    n_strategies = st.slider("Стратегии",2,10,5)
    n_rounds = st.slider("Проекты",100,1000,300)
    c = st.slider("Параметр доверия",0.5,3.0,2.0)
    true_means = [np.random.uniform(0.4,0.9) for _ in range(n_strategies)]
    rewards = np.zeros(n_strategies)
    counts = np.zeros(n_strategies)
    for t in range(1,n_rounds+1):
        ucb = np.zeros(n_strategies)
        for i in range(n_strategies):
            if counts[i] > 0:
                mean = rewards[i]/counts[i]
                conf = c * math.sqrt(2*math.log(t)/counts[i])
                ucb[i] = mean + conf
            else:
                ucb[i] = float("inf")
        choice = np.argmax(ucb)
        reward = np.random.rand() < true_means[choice]
        rewards[choice] += reward
        counts[choice] += 1

# ==========================================================
# TAB 3 — ABOUT
# ==========================================================

with tabs[2]:
    st.markdown("""
Научная система анализа строительных рисков.

Возможности:

• ML прогнозирование  
• Quantile risk thresholds (с легендой)  
• Correlation analysis  
• Factor impact plots  
• Factor-specific errors (MSE/MAE)  
• SHAP анализ  
• SHAP Values в отчете  
• UCB simulation  
• Model coefficients (B0…B13)
""")