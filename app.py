import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Предсказание цены авто", layout="wide")

@st.cache_resource
def load_models():
    models_dict = {}
    
    scaler = joblib.load('models/scaler.joblib')
    
    models_dict['sklearn_ridge'] = joblib.load('models/ridge_model.joblib')
    models_dict['sklearn_gb'] = joblib.load('models/gb_model.joblib')
    models_dict['sklearn_rf'] = joblib.load('models/rf_model.joblib')
    models_dict['sklearn_stacking'] = joblib.load('models/stacking_model.joblib')

    models_dict['catboost'] = CatBoostRegressor()
    models_dict['catboost'].load_model('models/catboost_model.cbm')

    models_dict['keras_fcnn'] = load_model('models/keras_model.keras')
    
    return models_dict, scaler

models_dict, scaler = load_models()

page = st.sidebar.radio("Навигация", ["1. О разработчике", "2. О датасете", "3. Визуализация", "4. Инференс (Прогноз)"])

# --- СТРАНИЦА 1: О разработчике ---
if page == "1. О разработчике":
    st.title("Разработка Web-приложения для инференса ML-моделей")
    # st.image("your_photo.jpg", width=200)
    st.markdown("**Студент:** Дядченко Дмитрий Константинович")
    st.markdown("**Группа:** ФИТ-241")
    st.markdown("**Тема:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных (Регрессия)")

# --- СТРАНИЦА 2: О датасете ---
elif page == "2. О датасете":
    st.title("Описание набора данных")
    st.markdown("### Предметная область")
    st.write("Датасет содержит информацию о подержанных автомобилях. Целевая переменная: `price_usd` (цена в долларах).")
    st.markdown("### Предобработка данных")
    st.write("Были обработаны пропуски, закодированы категориальные признаки (One-Hed Encoding), применено масштабирование (StandardScaler).")
    st.markdown("### EDA (Разведочный анализ)")
    st.write("В ходе анализа были выявлены сильные корреляции между годом выпуска, пробегом и целевой переменной (ценой).")

# --- СТРАНИЦА 3: Визуализация ---
elif page == "3. Визуализация":
    st.title("Визуальный анализ данных")
    
    try:
        df = pd.read_csv("processed_cars.csv") 
        
        st.subheader("1. Распределение целевой переменной (Цена)")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['price_usd'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("2. Тепловая карта корреляций")
        numeric_cols = ['odometer_value', 'year_produced', 'engine_capacity', 'price_usd', 
                        'number_of_photos', 'up_counter', 'duration_listed', 'car_age']
        binary_features = [col for col in df.columns if col.startswith('feature_')] + \
                          ['engine_has_gas', 'has_warranty', 'is_exchangeable']
        
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        binary_features = [col for col in binary_features if col in df.columns]
        
        corr_features = numeric_cols + binary_features
        corr_matrix = df[corr_features].corr()
        
        fig2, ax2 = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                    linewidths=0.5, square=True, annot_kws={'size': 8}, ax=ax2)
        ax2.set_title('Корреляционная матрица числовых и бинарных признаков', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig2)

        st.subheader("3. Зависимость цены от года выпуска")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='car_age', y='price_usd', data=df, alpha=0.5, ax=ax3)
        st.pyplot(fig3)

        st.subheader("4. Важность признаков (на примере Random Forest)")
        rf_model = models_dict['sklearn_rf']
        feature_names = scaler.feature_names_in_
        importances = rf_model.feature_importances_
        
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=fi_df, ax=ax4)
        ax4.set_title("Топ-15 важных признаков (Random Forest)", fontsize=14)
        ax4.set_xlabel("Важность")
        ax4.set_ylabel("Признак")
        plt.tight_layout()
        st.pyplot(fig4)
        
    except FileNotFoundError:
        st.warning("Файл 'processed_cars.csv' не найден. Убедитесь, что он находится в той же папке, что и app.py")

# --- СТРАНИЦА 4: Инференс ---
elif page == "4. Инференс (Прогноз)":
    st.title("Прогнозирование стоимости автомобиля")
    
    st.subheader("Ввод данных")
    st.write("Загрузите CSV файл с данными или введите значения вручную.")
    
    uploaded_file = st.file_uploader("Загрузить CSV", type=["csv"])
    
    if uploaded_file is not None:
        input_df_raw = pd.read_csv(uploaded_file)
        try:
            input_df = input_df_raw[scaler.feature_names_in_]
        except KeyError as e:
            st.error(f"В загруженном файле отсутствуют необходимые признаки: {e}")
            st.stop()
    else:
        input_data = {}
        for col in scaler.feature_names_in_:
            input_data[col] = st.number_input(f"{col}", value=0.0, step=1.0)
        input_df = pd.DataFrame([input_data])

    if st.button("Рассчитать стоимость"):
        X_input_scaled = scaler.transform(input_df)
        
        st.subheader("Результаты предсказания:")

        model_predictions = {
            "Ridge Regression": models_dict['sklearn_ridge'].predict(X_input_scaled)[0],
            "Gradient Boosting": models_dict['sklearn_gb'].predict(X_input_scaled)[0],
            "CatBoost": models_dict['catboost'].predict(X_input_scaled)[0],
            "Random Forest": models_dict['sklearn_rf'].predict(X_input_scaled)[0],
            "Stacking": models_dict['sklearn_stacking'].predict(X_input_scaled)[0],
            "Нейросеть (Keras)": models_dict['keras_fcnn'].predict(X_input_scaled)[0][0]
        }

        cols_row1 = st.columns(3)
        cols_row2 = st.columns(3)
        display_cols = [cols_row1, cols_row2]
        
        for i, (model_name, pred_value) in enumerate(model_predictions.items()):
            row_idx = i // 3 
            col_idx = i % 3
            
            with display_cols[row_idx][col_idx]:
                st.metric(model_name, f"${pred_value:,.2f}")
                
        avg_price = np.mean(list(model_predictions.values()))
        
        st.divider()
        st.success(f"Средняя прогнозируемая стоимость автомобиля (по ансамблю из 6 моделей) составляет: **${avg_price:,.2f}**")
