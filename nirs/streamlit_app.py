"""
Веб-приложение на Streamlit для демонстрации моделей машинного обучения.
Задача: предсказание оценки аниме (score).

Запуск: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- Константы ----
TOP_GENRES = ['Action', 'Comedy', 'Fantasy', 'Adventure', 'Drama', 'Sci-Fi', 'Romance', 'Slice of Life']
FEATURES   = ['type_enc', 'rating_enc', 'source_enc', 'log_members', 'log_favorites',
              'episodes_fill', 'duration_min', 'status_enc'] + [f'genre_{g}' for g in TOP_GENRES]

# ---- Загрузка и подготовка данных ----
@st.cache_data
def load_and_prepare():
    df = pd.read_csv('anime_dataset.csv')
    df_ml = df.dropna(subset=['score']).copy()

    def parse_duration(d):
        if pd.isna(d): return np.nan
        d = str(d).lower(); h = m = 0
        if 'hr' in d:
            try: h = int(d.split('hr')[0].strip().split()[-1])
            except: pass
        if 'min' in d:
            try: m = int(d.split('min')[0].strip().split()[-1])
            except: pass
        total = h * 60 + m
        return total if total > 0 else np.nan

    df_ml['duration_min'] = df_ml['duration'].apply(parse_duration)

    type_map = {'TV': 0, 'Movie': 1, 'OVA': 2, 'ONA': 3, 'Special': 4,
                'TV Special': 5, 'Music': 6, 'CM': 7, 'PV': 8}
    df_ml['type_enc'] = df_ml['type'].map(type_map).fillna(9)

    rating_order = ['G - All Ages', 'PG - Children', 'PG-13 - Teens 13 or older',
                    'R - 17+ (violence & profanity)', 'R+ - Mild Nudity', 'Rx - Hentai']
    df_ml['rating_enc'] = df_ml['rating'].map({r: i for i, r in enumerate(rating_order)}).fillna(-1)

    source_freq = df_ml['source'].value_counts()
    df_ml['source_enc'] = df_ml['source'].map(source_freq).fillna(0)

    for g in TOP_GENRES:
        df_ml[f'genre_{g}'] = df_ml['genres'].fillna('').str.contains(g).astype(int)

    df_ml['log_members']   = np.log1p(df_ml['members'])
    df_ml['log_favorites'] = np.log1p(df_ml['favorites'])
    df_ml['episodes_fill'] = df_ml['episodes'].fillna(df_ml['episodes'].median())

    status_map = {'Finished Airing': 0, 'Currently Airing': 1, 'Not yet aired': 2}
    df_ml['status_enc'] = df_ml['status'].map(status_map).fillna(0)

    df_ml[FEATURES] = df_ml[FEATURES].fillna(df_ml[FEATURES].median())
    X = df_ml[FEATURES]
    y = df_ml['score']
    return X, y, df_ml

# ---- Обучение модели ----
def train_model(model_name, hyperparams, X_train, X_test, y_train, y_test):
    if model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(
            n_estimators=hyperparams['n_estimators'],
            learning_rate=hyperparams['learning_rate'],
            max_depth=hyperparams['max_depth'],
            random_state=42
        )
        model.fit(X_train, y_train)
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams.get('max_depth', None),
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
    elif model_name == 'Ridge':
        model = Ridge(alpha=hyperparams['alpha'])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        model.fit(X_train, y_train)
    elif model_name == 'KNN':
        model = KNeighborsRegressor(n_neighbors=hyperparams['n_neighbors'])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        model.fit(X_train, y_train)
    elif model_name == 'Decision Tree':
        model = DecisionTreeRegressor(
            max_depth=hyperparams.get('max_depth', None),
            random_state=42
        )
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    return model, pred, rmse, mae, r2

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Anime Score Predictor", page_icon="🎌", layout="wide")

st.title("🎌 Предсказание оценки аниме")
st.markdown("""
**НИРС по дисциплине «Технологии машинного обучения»**  
Задача регрессии: предсказание оценки аниме на основе характеристик тайтла.
""")

with st.spinner("Загрузка и подготовка данных..."):
    X, y, df_ml = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.success(f"✅ Данные загружены: {len(df_ml):,} записей, {len(FEATURES)} признаков")

# ---- Боковая панель ----
st.sidebar.header("⚙️ Настройка модели")

model_name = st.sidebar.selectbox(
    "Выберите модель:",
    ['Gradient Boosting', 'Random Forest', 'Ridge', 'KNN', 'Decision Tree']
)

hyperparams = {}

if model_name == 'Gradient Boosting':
    st.sidebar.markdown("### Гиперпараметры Gradient Boosting")
    hyperparams['n_estimators']  = st.sidebar.slider("n_estimators (количество деревьев)", 50, 300, 100, step=50)
    hyperparams['learning_rate'] = st.sidebar.select_slider("learning_rate", options=[0.01, 0.05, 0.1, 0.15, 0.2], value=0.1)
    hyperparams['max_depth']     = st.sidebar.slider("max_depth (глубина дерева)", 2, 8, 3)

elif model_name == 'Random Forest':
    st.sidebar.markdown("### Гиперпараметры Random Forest")
    hyperparams['n_estimators'] = st.sidebar.slider("n_estimators", 50, 300, 100, step=50)
    max_depth_choice = st.sidebar.radio("max_depth", ["None (без ограничений)", "10", "20"])
    hyperparams['max_depth'] = None if "None" in max_depth_choice else int(max_depth_choice)

elif model_name == 'Ridge':
    st.sidebar.markdown("### Гиперпараметры Ridge")
    hyperparams['alpha'] = st.sidebar.select_slider("alpha (регуляризация)", options=[0.001, 0.01, 0.1, 1, 10, 100], value=1)

elif model_name == 'KNN':
    st.sidebar.markdown("### Гиперпараметры KNN")
    hyperparams['n_neighbors'] = st.sidebar.slider("n_neighbors", 1, 50, 10)

elif model_name == 'Decision Tree':
    st.sidebar.markdown("### Гиперпараметры Decision Tree")
    max_depth_choice = st.sidebar.radio("max_depth", ["3", "5", "7", "10", "None"])
    hyperparams['max_depth'] = None if max_depth_choice == "None" else int(max_depth_choice)

# ---- Обучение ----
with st.spinner(f"🔄 Обучение модели {model_name}..."):
    model, pred, rmse, mae, r2 = train_model(model_name, hyperparams, X_train.copy(), X_test.copy(), y_train, y_test)

# ---- Метрики ----
st.header("📊 Метрики качества модели")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error — чем меньше, тем лучше")
col2.metric("MAE",  f"{mae:.4f}",  help="Mean Absolute Error — средняя ошибка в баллах")
col3.metric("R²",   f"{r2:.4f}",   help="Коэффициент детерминации (макс. = 1.0)")

# ---- Графики ----
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Предсказанные vs Фактические значения")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, y_test, alpha=0.3, s=8, color='steelblue')
    mn = min(y_test.min(), pred.min()); mx = max(y_test.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Идеальная линия')
    ax.set_xlabel('Предсказанные значения')
    ax.set_ylabel('Фактические значения')
    ax.set_title(f'{model_name}')
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("📉 Распределение остатков")
    fig, ax = plt.subplots(figsize=(6, 5))
    residuals = y_test.values - pred
    ax.hist(residuals, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Остатки (y_true - y_pred)')
    ax.set_ylabel('Частота')
    ax.set_title(f'Остатки модели {model_name}')
    st.pyplot(fig)
    plt.close()

# ---- Важность признаков (для деревьев) ----
if model_name in ['Random Forest', 'Decision Tree', 'Gradient Boosting']:
    st.subheader("🔍 Важность признаков")
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['coral' if v > importances.median() else 'steelblue' for v in importances]
    importances.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(f'Важность признаков — {model_name}')
    ax.set_xlabel('Feature Importance')
    st.pyplot(fig)
    plt.close()

# ---- Предсказание для конкретного аниме ----
st.header("🔮 Предсказать оценку для конкретного аниме")
st.markdown("Введите параметры аниме и получите предсказанную оценку:")

c1, c2, c3 = st.columns(3)
with c1:
    anime_type    = st.selectbox("Тип", ['TV', 'Movie', 'OVA', 'ONA', 'Special'])
    anime_rating  = st.selectbox("Возрастной рейтинг", ['G - All Ages', 'PG - Children',
                                 'PG-13 - Teens 13 or older', 'R - 17+ (violence & profanity)',
                                 'R+ - Mild Nudity', 'Rx - Hentai'])
with c2:
    anime_members   = st.number_input("Количество участников (members)", min_value=1, value=50000)
    anime_favorites = st.number_input("Количество в избранном (favorites)", min_value=0, value=500)
with c3:
    anime_episodes = st.number_input("Количество эпизодов", min_value=1, value=12)
    anime_genres   = st.multiselect("Жанры", TOP_GENRES, default=['Action'])

type_map_local = {'TV': 0, 'Movie': 1, 'OVA': 2, 'ONA': 3, 'Special': 4}
rating_map_local = {r: i for i, r in enumerate(['G - All Ages', 'PG - Children',
    'PG-13 - Teens 13 or older', 'R - 17+ (violence & profanity)', 'R+ - Mild Nudity', 'Rx - Hentai'])}

input_row = {
    'type_enc':      type_map_local.get(anime_type, 0),
    'rating_enc':    rating_map_local.get(anime_rating, 0),
    'source_enc':    5000,
    'log_members':   np.log1p(anime_members),
    'log_favorites': np.log1p(anime_favorites),
    'episodes_fill': anime_episodes,
    'duration_min':  24.0,
    'status_enc':    0,
}
for g in TOP_GENRES:
    input_row[f'genre_{g}'] = 1 if g in anime_genres else 0

input_df = pd.DataFrame([input_row])[FEATURES]
if model_name in ['Ridge', 'KNN']:
    scaler_pred = StandardScaler()
    scaler_pred.fit(X_train)
    input_scaled = scaler_pred.transform(input_df)
    single_pred = model.predict(input_scaled)[0]
else:
    single_pred = model.predict(input_df)[0]

single_pred = float(np.clip(single_pred, 1.0, 10.0))
st.success(f"🎌 Предсказанная оценка: **{single_pred:.2f} / 10.0**")

# ---- Информация о данных ----
with st.expander("📋 Информация о датасете"):
    st.write(f"**Всего аниме:** {len(df_ml):,}")
    st.write(f"**Диапазон оценок:** {y.min():.2f} — {y.max():.2f}")
    st.write(f"**Средняя оценка:** {y.mean():.2f}")
    st.write(f"**Используемые признаки:** {', '.join(FEATURES)}")
    st.dataframe(df_ml[['title', 'type', 'score', 'members', 'genres']].head(10).reset_index(drop=True))
