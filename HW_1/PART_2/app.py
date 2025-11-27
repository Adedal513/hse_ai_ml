import pandas as pd
import streamlit as st
from src.config import ORIGINAL_FEATURES, AutoOptions
from src.preprocess import TechFieldsTransformer, HierarchicalImputer, FeatureEngineeringGenerator
import joblib
import altair as alt
import pickle
import numpy as np
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    pipeline = joblib.load("pipeline.pkl")
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return pipeline, feature_names

model, feature_names = load_model()

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти к:", ["EDA", "Предсказание", "Визуализация весов"])

if page == 'Предсказание':
    st.title('Предсказание цены автомобиля')
    st.write("""
    Здесь вы можете либо загрузить CSV со списком автомобилей, 
    либо вручную ввести характеристики одного автомобиля.
    """)

    mode = st.radio(
        "Выберите способ ввода:",
        ["Загрузка CSV", "Ручной ввод"],
        horizontal=True
    )

    if mode == 'Загрузка CSV':
        file = st.file_uploader("Загрузите CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)

            missing = [c for c in ORIGINAL_FEATURES if c not in df.columns]
            if missing:
                st.error(f"В файле отсутствуют обязательные признаки: {missing}")
                st.stop()
            
            extra = [c for c in df.columns if c not in ORIGINAL_FEATURES]
            if extra:
                st.warning(f"В файле присутствуют неизвестные признаки: {extra}. Они будут проигнорированы.")
                df = df.drop(labels=extra, axis=1)
            
            st.write("### Загруженные данные:")
            st.write(f'Объём: {len(df)} записей')
            st.write(df.head())

            if st.button('Предсказать', type='primary', width="stretch"):
                preds = model.predict(df)
                df["prediction"] = preds

                st.write("### Результат:")
                st.write(df)

                st.download_button(
                    "Скачать результат",
                    df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv"
                )
    else:
        st.write("Введите характеристики авто:")

        # Категориальные поля
        brand = st.selectbox("Name", AutoOptions.BRANDS)
        fuel = st.selectbox("Fuel Type", AutoOptions.FUELS)
        transmission = st.selectbox("Transmission", AutoOptions.TRANSMISSIONS)
        owner = st.selectbox("Owner Type", AutoOptions.OWNER)
        seller_type = st.selectbox("Seller Type", AutoOptions.SELLER_TYPE)

        # Числовые поля с единицами и шагами
        year = st.number_input(
            "Year", 
            min_value=1960, 
            max_value=datetime.now().year, 
            step=1,
        )

        km_driven = st.number_input(
            "KM Driven", 
            min_value=0, 
            max_value=int(800_000), 
            step=1000
        )

        mileage = st.number_input(
            "Mileage (kmpl или km/kg)", 
            min_value=0.0,
            step=0.1, 
        )

        engine = st.number_input(
            "Engine (CC)", 
            min_value=0,
            step=10,
        )

        max_power = st.number_input(
            "Max Power (bhp)",
            min_value=0,
            step=1,
        )

        seats = st.number_input(
            "Seats", 
            min_value=1, 
            max_value=20, 
            step=1
        )

        # Собираем данные в DataFrame для модели
        input_data = {
            "name": brand,
            "fuel": fuel,
            "transmission": transmission,
            "owner": owner,
            "seller_type": seller_type,
            "year": year,
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        }

        df_input = pd.DataFrame([input_data])

        # Кнопка для предсказания
        if st.button("Предсказать"):
            st.write("### Введённые данные")
            st.write(df_input)

            pred = model.predict(df_input)[0]
            st.success(f"Предсказанная цена: **{pred:,.2f}**")
elif page == 'Визуализация весов':
    st.title("Важность признаков модели")

    preprocess = model.named_steps["preprocess"]
    num_pipeline = preprocess.named_transformers_["num"]
    poly = num_pipeline.named_steps["poly"]
    ridge = model.named_steps["ridge"]

    num_features = preprocess.transformers_[0][2]
    poly_num_features = poly.get_feature_names_out(num_features)
    cat_features = preprocess.named_transformers_["cat"].get_feature_names_out()
    feature_names = list(poly_num_features) + list(cat_features)

    coef = ridge.coef_
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef)
    })

    coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)
    top_features = coef_df.head(20)

    st.write(f"### Топ-20 признаков по абсолютной важности")
    st.dataframe(top_features[['feature', 'coefficient']].reset_index(drop=True))

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features['feature'][::-1], top_features['coefficient'][::-1])
    ax.set_xlabel("Коэффициент Ridge")
    ax.set_title(f"Топ-20 признаков")
    st.pyplot(fig)

elif page == 'EDA':
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        tabs = st.tabs(["Overview", "Numerical", "Categorical", "Scatter / Pairplot"])
    
        with tabs[0]:
            st.subheader("Обзор данных")
            st.dataframe(df.head())
            st.write("### Основные статистики числовых колонок")
            st.dataframe(df.describe())
            st.write("### Информация по колонкам")
            buffer = df.info(buf=None)
            st.text(str(buffer))
            st.write("### Пропущенные значения по колонкам")
            st.dataframe(df.isna().sum())

        with tabs[1]:
            st.subheader("Числовые признаки")
            col_to_plot = st.selectbox("Выберите колонку для распределения", numeric_cols, key="num_dist")
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{col_to_plot}:Q", bin=alt.Bin(maxbins=50)),
                y='count()'
            )
            st.altair_chart(chart, use_container_width=True)

            if len(numeric_cols) > 1:
                st.write("### Матрица корреляций")
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            # Boxplot по числовым vs категориальным
            if categorical_cols:
                st.write("### Boxplot числовой vs категориальный")
                cat_col = st.selectbox("Категориальная колонка для boxplot", categorical_cols, key="box_cat")
                num_col = st.selectbox("Числовая колонка для boxplot", numeric_cols, key="box_num")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        with tabs[2]:
            st.subheader("Категориальные признаки")
            for col in categorical_cols:
                chart = alt.Chart(df).mark_bar().encode(
                    x=f"{col}:N",
                    y='count()'
                )
                st.altair_chart(chart, use_container_width=True)

        # -------------------------------
        # Scatter / Pairplot
        # -------------------------------
        with tabs[3]:
            st.subheader("Scatter / Pairplot")
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Выберите X колонку", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Выберите Y колонку", numeric_cols, key="scatter_y")
                color_col = st.selectbox("Цвет по (опционально)", [None] + categorical_cols, key="scatter_color")

                scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
                    x=f"{x_col}:Q",
                    y=f"{y_col}:Q",
                    color=f"{color_col}:N" if color_col else alt.value("steelblue"),
                    tooltip=[x_col, y_col] + ([color_col] if color_col else [])
                )
                st.altair_chart(scatter, use_container_width=True)
