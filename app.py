import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from PIL import Image
import streamlit.components.v1 as components

st.set_page_config(
    page_title="РГР: Инференс моделей ML",
    page_icon=Image.open('icon.png'),
    layout="wide" 
)

@st.cache_data
def load_data():
    if os.path.exists('diamonds_processed.csv'):
        return pd.read_csv('diamonds_processed.csv')
    return None

df = load_data()

models_dict = {}



with open('ml1_polynomial.pkl', 'rb') as f:
    models_dict["ML1: Decision Tree"] = pickle.load(f)

with open('ml2_boosting.pkl', 'rb') as f:
    models_dict["ML2: Gradient Boosting"] = pickle.load(f)

cb_model = CatBoostRegressor()
cb_model.load_model('ml3_catboost.cbm')
models_dict["ML3: CatBoost"] = cb_model

with open('ml4_bagging.pkl', 'rb') as f:
    models_dict["ML4: Random Forest"] = pickle.load(f)


with open('ml5_stacking.pkl', 'rb') as f:
    models_dict["ML5: Stacking"] = pickle.load(f)


with open('ml6_neural_network.pkl', 'rb') as f:
    models_dict["ML6: Neural Network"] = pickle.load(f)


with open('scaler.pkl', 'rb') as f:
    models_dict['scaler'] = pickle.load(f)


st.sidebar.title("🎮 Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["Профиль", "Описание датасета", "Визуализация данных", "Инференс моделей"]
)




# --- СТРАНИЦА 1: ПРОФИЛЬ ---
if page == "Профиль":
    st.markdown("<h1 style='color: #4A3E43; font-weight: 800; font-family: sans-serif; margin-bottom: 5px;'>Расчётно-графическая работа</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0 0 25px 0; border: none; height: 1px; background-color: #EEDAE5;'>", unsafe_allow_html=True)

    if os.path.exists('Lissy.png'):
        img_file = 'Lissy.png'
    elif os.path.exists('Lissy.jpg'):
        img_file = 'Lissy.jpg'
    else:
        img_file = None
        st.error("Файл фотографии не найден в папке проекта!")

    if img_file:
        with open(img_file, "rb") as f:
            img_bytes = f.read()
        import base64
        encoded_img = base64.b64encode(img_bytes).decode()
    else:
        encoded_img = ""

    cute_profile_html = f"""
    <div style="font-family: 'Segoe UI', Roboto, sans-serif; 
                background-color: #FFF5F8; 
                border: 2px dashed #FFC0CB; 
                border-radius: 24px; 
                padding: 25px; 
                box-shadow: 0px 6px 18px rgba(255, 192, 203, 0.25); 
                position: relative; 
                overflow: hidden;
                display: flex;
                gap: 25px;
                align-items: stretch;">
        
        <div style="position: absolute; top: 8px; left: 12px; color: #FFB6D9; font-size: 16px; opacity: 0.7;">✰</div>
        <div style="position: absolute; bottom: 8px; right: 12px; color: #FFB6D9; font-size: 16px; opacity: 0.7;">✰</div>
        <div style="position: absolute; top: 12px; right: 15px; color: #FFB6D9; font-size: 12px; opacity: 0.5;">✨</div>
        
        <div style="flex: 3; display: flex; flex-direction: column; gap: 15px;">
            <h3 style='color: #E066A6; font-size: 20px; font-weight: 700; margin: 0 0 5px 0; display: flex; align-items: center; gap: 8px;'>
                🎀 РАЗРАБОТЧИК 🎀
            </h3>
            
            <div style="background-color: #FFF0F5; color: #5A4A50; padding: 15px; border-radius: 14px; border-left: 6px solid #FFC0CB; box-shadow: 0px 2px 8px rgba(255, 192, 203, 0.1);">
                <span style="font-weight: 800; color: #E066A6; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">ТЕМА РГР</span>
                <p style="margin: 4px 0 0 0; font-size: 14px; line-height: 1.5; font-weight: 500;">
                    Разработка веб-приложения для прогнозирования стоимости алмазов на основе моделей машинного обучения.
                </p>
            </div>
            
            <div style="background-color: #FFFFFF; color: #5A4A50; padding: 15px; border-radius: 14px; border: 1px solid #FFE4E1; box-shadow: 0px 2px 8px rgba(255, 192, 203, 0.05);">
                <span style="font-weight: 800; color: #A39399; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">СТУДЕНТ</span>
                <p style="margin: 2px 0 10px 0; font-size: 16px; font-weight: 600; color: #4A3E43;">Вырышева Алиса Викторовна</p>
                
                <div style="display: flex; gap: 40px; border-top: 1px dashed #FFE4E1; padding-top: 8px;">
                    <div>
                        <span style="color: #A39399; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">Группа</span>
                        <p style="margin: 1px 0 0 0; font-weight: 700; color: #E066A6; font-size: 14px;">ФИТ-241</p>
                    </div>
                    <div>
                        <span style="color: #A39399; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">ВУЗ</span>
                        <p style="margin: 1px 0 0 0; font-weight: 700; color: #5A4A50; font-size: 14px;">ОмГТУ</p>
                    </div>
                </div>
            </div>
            
            
        </div>
        
        <div style="flex: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #FFFFFF; border: 1px solid #FFE4E1; border-radius: 18px; padding: 12px; box-shadow: inset 0px 2px 6px rgba(255, 182, 193, 0.1); max-width: 220px;">
            
            <div style="width: 100%; display: flex; justify-content: center; align-items: center; overflow: hidden; border-radius: 12px; border: 3px solid #FFF2F6; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); margin-bottom: 8px;">
                <img src="data:image/png;base64,{encoded_img}" style="width: 100%; height: auto; display: block; object-fit: contain;"/>
            </div>
            
            <div style="background-color: #EFE5FA; color: #52444C; font-size: 12px; font-weight: 700; padding: 4px 15px; border-radius: 20px; text-align: center; width: 80%;">
                Алиса
            </div>
        </div>
        
    </div>
    """
    
    components.html(cute_profile_html, height=500)
        

            
    st.markdown("<hr style='margin: 30px 0 15px 0; border: none; height: 1px; background-color: #EEDAE5;'>", unsafe_allow_html=True)
    st.info("⬅️ Переключите раздел в левом меню («Навигация»), чтобы перейти к анализу данных или прогнозированию.")




# --- СТРАНИЦА 2: ОПИСАНИЕ ДАТАСЕТА ---
elif page == "Описание датасета":
    st.title("📊 Анализ структуры данных")
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего строк (алмазов)", f"{df.shape[0]:,}")
        col2.metric("Количество фичей (X)", df.shape[1] - 1)
        col3.metric("Целевой признак (Y)", "price ($)")
        
        st.markdown("---")
        
        st.write("### 📜 Подробное описание признаков")
        
        st.markdown("""
        #### Физические и качественные параметры:
        * **carat** — вес бриллианта в каратах (диапазон в данных: от 0.2 до 5.01)
        * **cut** — качество огранки (в порядке возрастания: *Fair, Good, Very Good, Premium, Ideal*)
        * **color** — цвет бриллианта (от **J** — худший, с желтизной, до **D** — лучший, абсолютно бесцветный)
        * **clarity** — чистота бриллианта, отсутствие дефектов (от **I1** — худший, до **IF** — лучший/идеальный. Полный порядок: *I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF*)
        
        #### Геометрические параметры (Пропорции):
        * **depth** — общая глубина в процентах (от 43% до 79%). Вычисляется по формуле: `z / mean(x, y) = 2 * z / (x + y)`
        * **table** — ширина верхней площадки бриллианта относительно самой широкой точки (от 43% до 95%)
        * **x** — длина алмаза в мм (от 0 до 10.74)
        * **y** — ширина алмаза в мм (от 0 до 58.9)
        * **z** — высота/глубина алмаза в мм (от 0 до 31.8)
        
        #### Таргет (Целевая переменная):
        * **price** — цена алмаза в долларах США (в диапазоне от \$326 до \$18,823)
        """)

        st.markdown("---")
        st.write("### 🛠️ Особенности предобработки данных и EDA")
        st.markdown("""
        1. **Удаление дубликатов и пропусков:** В ходе первичного анализа (EDA) явных пропусков в данных обнаружено не было. Были удалены строки-дубликаты.
        2. **Кодирование категориальных признаков:** Текстовые параметры качества (`cut`), цвета (`color`) и чистоты (`clarity`) имеют чёткий внутренний порядок. К ним было применено порядковое кодирование (**Ordinal Encoding**) в соответствии с их экспертной ценностью.
        3. **Масштабирование:** Для корректной работы полносвязной нейронной сети (ML6) выполнено масштабирование числовых признаков с помощью **StandardScaler**.
        """)
        
        st.markdown("---")
        
        st.write("### 🔍 Первые 10 строк датасета")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write("### 📈 Основные статистические характеристики")
        st.dataframe(df.describe(), use_container_width=True)
        
    else:
        st.error("🚨 Не удалось найти файл 'diamonds_processed.csv'. Пожалуйста, проверь его наличие в папке проекта.")




# --- СТРАНИЦА 3: ВИЗУАЛИЗАЦИЯ ДАННЫХ ---
elif page == "Визуализация данных":
    st.title("📈 Интерактивная визуализация признаков")
    if df is not None:
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        with row1_col1:
            st.write("#### 1. Распределение целевого признака (Цена)")
            fig, ax = plt.subplots()
            sns.histplot(df['price'], bins=30, kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)
            
        with row1_col2:
            st.write("#### 2. Зависимость стоимости от веса (Carat)")
            sample_df = df.sample(min(1000, len(df)), random_state=42)
            fig, ax = plt.subplots()
            feature_x = 'carat' if 'carat' in df.columns else df.columns[0]
            sns.scatterplot(data=sample_df, x=feature_x, y='price', alpha=0.6, color='coral', ax=ax)
            st.pyplot(fig)
            
        with row2_col1:
            st.write("#### 3. Анализ выбросов и цен по категориям")
            fig, ax = plt.subplots()
            feature_to_plot = 'cut' if 'cut' in df.columns else (df.columns[0] if df.columns[0] != 'price' else df.columns[1])
            sns.boxplot(data=df, x=feature_to_plot, y='price', palette='Set2', ax=ax)
            st.pyplot(fig)
            
        with row2_col2:
            st.write("#### 4. Тепловая карта корреляции числовых признаков")
            fig, ax = plt.subplots(figsize=(6, 4))
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 8})
            st.pyplot(fig)




# --- СТРАНИЦА 4: ИНФЕРЕНС МОДЕЛЕЙ ---
elif page == "Инференс моделей":
    st.markdown("<h1 style='color: #4A3E43; font-weight: 800;'>🔮 Инференс и прогнозирование</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 10px 0 25px 0; border: none; height: 1px; background-color: #EEDAE5;'>", unsafe_allow_html=True)
    
    if df is not None:
        prediction_mode = st.radio(
            "Выберите способ подачи данных:",
            ["✨ Одиночное предсказание", "📂 Пакетное предсказание (загрузка *.csv)"],
            horizontal=True
        )
        
        st.write("---")
        
        # ------------------ РЕЖИМ 1: ОДИНОЧНОЕ ПРЕДСКАЗАНИЕ ------------------
        if prediction_mode == "✨ Одиночное предсказание":
            st.markdown("<h3 style='color: #4A3E43; font-size: 20px; font-weight: 700; margin-bottom: 5px;'>💎 Параметры алмаза</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #5A4A50; font-size: 14.5px; margin-bottom: 20px;'>Задайте характеристики камня с помощью панелей ниже для расчёта стоимости.</p>", unsafe_allow_html=True)
            
            # 1. СЛОВАРИ МАППИНГА
            cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
            color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
            clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

            input_data = {}
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<span style='color: #E066A6; font-weight: 600; font-size: 14px;'>🌸 Главные свойства</span>", unsafe_allow_html=True)
                
                input_data['carat'] = st.slider("Вес (Carat, кт)", 
                                                float(df['carat'].min()), float(df['carat'].max() * 1.2), float(df['carat'].mean()), step=0.01)
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>Средний вес в базе: ~0.8 кт</i></div>", unsafe_allow_html=True)
                
                selected_cut = st.selectbox("Огранка (Cut)", list(cut_mapping.keys()), index=4) 
                input_data['cut'] = cut_mapping[selected_cut]
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>Fair — худшая, Ideal — лучшая</i></div>", unsafe_allow_html=True)
                
                selected_color = st.selectbox("Цвет (Color)", list(color_mapping.keys()), index=1) 
                input_data['color'] = color_mapping[selected_color]
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>J — худший, D — лучший (чистый)</i></div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<span style='color: #E066A6; font-weight: 600; font-size: 14px;'>🌸 Чистота и пропорции</span>", unsafe_allow_html=True)
                
                selected_clarity = st.selectbox("Чистота (Clarity)", list(clarity_mapping.keys()), index=6) 
                input_data['clarity'] = clarity_mapping[selected_clarity]
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>I1 — с дефектами, IF — идеальная</i></div>", unsafe_allow_html=True)
                
                input_data['depth'] = st.slider("Глубина (Depth, %)", 
                                                float(df['depth'].min()), float(df['depth'].max()), float(df['depth'].mean()), step=0.1)
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>Рыночная норма: 59% – 63%</i></div>", unsafe_allow_html=True)
                
                input_data['table'] = st.slider("Площадка (Table, %)", 
                                                float(df['table'].min()), float(df['table'].max()), float(df['table'].mean()), step=0.1)
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>Норма верхней грани: 54% – 60%</i></div>", unsafe_allow_html=True)

            with col3:
                st.markdown("<span style='color: #E066A6; font-weight: 600; font-size: 14px;'>🌸 Физические размеры</span>", unsafe_allow_html=True)
                
                input_data['x'] = st.slider("Длина (X, мм)", 
                                            float(df['x'].min()), float(df['x'].max()), float(df['x'].mean()), step=0.01)
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>Норма для 1 карата: ~6.5 мм</i></div>", unsafe_allow_html=True)
                
                input_data['y'] = st.slider("Ширина (Y, мм)", 
                                            float(df['y'].min()), float(df['y'].max()), float(df['y'].mean()), step=0.01)
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>В идеале почти равна длине X</i></div>", unsafe_allow_html=True)
                
                input_data['z'] = st.slider("Высота (Z, мм)", 
                                            float(df['z'].min()), float(df['z'].max()), float(df['z'].mean()), step=0.01)
                st.markdown("<div style='color: #A38591; font-size: 11.5px; margin-top: -12px; margin-bottom: 15px;'>✧ <i>Обычно около 60% от длины X</i></div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 15px 0 25px 0; border: none; height: 1px; background-color: #EEDAE5;'>", unsafe_allow_html=True)



            st.write("---")
            st.write("### ⚙️ Настройка предсказания")
            
            available_models = [k for k in models_dict.keys() if k != 'scaler']
            selected_model_name = st.selectbox("Выберите модель ML для инференса:", available_models)
            
            if st.button("👾 Рассчитать стоимость алмаза"):
                raw_input_df = pd.DataFrame([input_data])
                
                correct_order = [col for col in df.columns if col != 'price']
                input_df = raw_input_df[correct_order]
                
                model = models_dict[selected_model_name]
                
                try:
                    if "ML6" in selected_model_name or "NEURAL" in selected_model_name.upper():
                        if 'scaler' in models_dict:
                            scaled_input = models_dict['scaler'].transform(input_df)
                            scaled_input_df = pd.DataFrame(scaled_input, columns=correct_order)
                            raw_prediction = model.predict(scaled_input_df)
                        else:
                            st.error("🚨 Скалер ('scaler') не найден в models_dict! Нейросеть не может работать без него.")
                            raw_prediction = None
                    else:
                        raw_prediction = model.predict(input_df)
                    
                    if raw_prediction is not None:
                        if hasattr(raw_prediction, "__len__") and len(raw_prediction) > 0:
                            price_val = float(raw_prediction[0])
                        else:
                            price_val = float(raw_prediction)
                        
                        min_dataset_price = float(df['price'].min())
                        display_price = max(min_dataset_price, price_val)

                        st.markdown(
                            f"""
                            <div style="background-color: #FFF0F5; border: 2px solid #FFC0CB; border-radius: 14px; padding: 20px; text-align: center; margin-top: 15px; margin-bottom: 15px;">
                                <span style="color: #E066A6; font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                                    Стоимость по версии [{selected_model_name}]:
                                </span>
                                <h2 style="color: #4A3E43; margin: 8px 0 0 0; font-size: 32px; font-weight: 800;">${display_price:,.2f}</h2>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        if display_price > df['price'].median():
                            st.warning("👑 Элитный дорогой алмаз!")
                        else:
                            st.info("✨ Бюджетный вариант алмаза.")
                                
                except Exception as e:
                    st.error(f"Произошла ошибка во время инференса: {e}")


        # ------------------ РЕЖИМ 2: ПАКЕТНОЕ ПРЕДСКАЗАНИЕ ------------------
        else:
            st.markdown("### 📂 Загрузка файла для массового инференса")
            st.write("Загрузите файл *.csv, содержащий характеристики алмазов, чтобы посчитать стоимость для всей таблицы разом.")
            
            uploaded_file = st.file_uploader("Выберите файл датасета в формате *.csv", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    user_df = pd.read_csv(uploaded_file)
                    st.success("Файл успешно загружен!")
                    
                    st.write("**Превью загруженных данных (первые 5 строк):**")
                    st.dataframe(user_df.head())
                    
                    required_cols = [col for col in df.columns if col != 'price']
                    missing_cols = [col for col in required_cols if col not in user_df.columns]
                    
                    if missing_cols:
                        st.error(f"🚨 Ошибка валидации! В вашем файле отсутствуют обязательные признаки: {', '.join(missing_cols)}")
                    else:
                        available_models = [k for k in models_dict.keys() if k != 'scaler']
                        selected_model_batch = st.selectbox("Выберите модель ML для пакетного инференса:", available_models)
                        
                        if st.button("Запустить массовое прогнозирование"):
                            model = models_dict[selected_model_batch]
                            
                            batch_input = user_df[required_cols].copy()
                            
                            cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
                            color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
                            clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
                            
                            if batch_input['cut'].dtype == 'object':
                                batch_input['cut'] = batch_input['cut'].map(cut_mapping)
                            if batch_input['color'].dtype == 'object':
                                batch_input['color'] = batch_input['color'].map(color_mapping)
                            if batch_input['clarity'].dtype == 'object':
                                batch_input['clarity'] = batch_input['clarity'].map(clarity_mapping)
                                
                            if batch_input[['cut', 'color', 'clarity']].isna().any().any():
                                st.error("🚨 Ошибка кодирования! Проверьте правильность написания категорий (Cut, Color, Clarity) в файле. Названия должны быть как в датасете (н-р, 'Ideal', 'E', 'VVS1').")
                                predictions = None
                            else:
                                if "Neural Network" in selected_model_batch:
                                    if 'scaler' in models_dict:
                                        scaled_batch = models_dict['scaler'].transform(batch_input)
                                        predictions = model.predict(scaled_batch)
                                    else:
                                        st.error("Скалер не найден!")
                                        predictions = None
                                else:
                                    predictions = model.predict(batch_input)
                            
                            if predictions is not None:
                                user_df['predicted_price_USD'] = [round(float(p), 2) for p in predictions]
                                
                                st.write("**🎀 Прогноз готов! Результаты с рассчитанной стоимостью алмазов:**")
                                st.dataframe(user_df.head())
                                
                                csv_data = user_df.to_csv(index=False).encode('utf-8')
                                
                                st.download_button(
                                    label="📥 Скачать файл с прогнозами (*.csv)",
                                    data=csv_data,
                                    file_name="diamonds_with_prices.csv",
                                    mime="text/csv"
                                )
                except Exception as e:
                    st.error(f"Не удалось обработать файл. Ошибка: {e}")
                    
    else:
        st.error("🚨 Модели не могут работать без датасета. Загрузите или инициализируйте данные.")