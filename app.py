import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Airbnb CDMX - BI Dashboard", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    url = "https://data.insideairbnb.com/mexico/df/mexico-city/2025-09-27/data/listings.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    cols = ['id', 'name', 'neighbourhood_cleansed', 'room_type', 'accommodates',
            'bedrooms', 'beds', 'price', 'minimum_nights',
            'number_of_reviews', 'review_scores_rating', 'host_is_superhost',
            'availability_365']
    df = df[cols].copy()
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
    df = df[df['price'] > 0]
    df = df[df['price'] <= df['price'].quantile(0.99)]
    df['bedrooms'] = df['bedrooms'].fillna(1)
    df['beds'] = df['beds'].fillna(1)
    df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
    df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
    return df

df = load_data()

# Sidebar filtros
st.sidebar.title("🔍 Filtros")
alcaldias = sorted(df['neighbourhood_cleansed'].unique())
alcaldia_sel = st.sidebar.multiselect("Alcaldía", alcaldias, default=alcaldias[:5])
room_types = sorted(df['room_type'].unique())
room_sel = st.sidebar.multiselect("Tipo de cuarto", room_types, default=room_types)
precio_max = st.sidebar.slider("Precio máximo (MXN)", 500, 10000, 5000, step=500)

df_filtered = df[
    (df['neighbourhood_cleansed'].isin(alcaldia_sel)) &
    (df['room_type'].isin(room_sel)) &
    (df['price'] <= precio_max)
]

# Header
st.title("🏠 Airbnb CDMX — Business Intelligence Dashboard")
st.markdown("**Curso:** Inteligencia de Negocios | **Dataset:** Inside Airbnb Mexico City (Sep 2025)")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Listings", f"{len(df_filtered):,}")
col2.metric("Precio Mediano", f"${df_filtered['price'].median():,.0f} MXN")
col3.metric("Rating Promedio", f"{df_filtered['review_scores_rating'].mean():.2f} ⭐")
col4.metric("Alcaldías", f"{df_filtered['neighbourhood_cleansed'].nunique()}")

st.divider()

# Gráficas EDA
st.subheader("📊 Análisis Exploratorio")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_filtered[df_filtered['price'] <= 5000]['price'], bins=50, color='steelblue', edgecolor='white')
    ax.axvline(df_filtered['price'].median(), color='red', linestyle='--', label=f"Mediana: ${df_filtered['price'].median():,.0f}")
    ax.axvline(df_filtered['price'].mean(), color='orange', linestyle='--', label=f"Media: ${df_filtered['price'].mean():,.0f}")
    ax.set_title("Distribución de Precios por Noche")
    ax.set_xlabel("Precio (MXN)")
    ax.legend()
    st.pyplot(fig)

with col2:
    top_alc = df_filtered.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top_alc.index, top_alc.values, color='coral')
    ax.set_title("Top 10 Alcaldías por Precio Promedio")
    ax.set_xlabel("Precio Promedio (MXN)")
    ax.invert_yaxis()
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    room_price = df_filtered.groupby('room_type')['price'].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(room_price.index, room_price.values, color='mediumseagreen')
    ax.set_title("Precio Mediano por Tipo de Cuarto")
    ax.set_ylabel("Precio Mediano (MXN)")
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df_filtered['review_scores_rating'], df_filtered['price'], alpha=0.3, color='mediumpurple')
    ax.set_title("Rating vs Precio")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Precio (MXN)")
    st.pyplot(fig)

st.divider()

# Modelo
st.subheader("🤖 Modelo Predictivo — Regresión Lineal")

df_model = df_filtered[['room_type', 'accommodates', 'bedrooms', 'beds',
                          'minimum_nights', 'number_of_reviews',
                          'review_scores_rating', 'availability_365', 'price']].dropna()

if len(df_model) > 100:
    le = LabelEncoder()
    df_model = df_model.copy()
    df_model['room_type_enc'] = le.fit_transform(df_model['room_type'])
    X = df_model[['room_type_enc', 'accommodates', 'bedrooms', 'beds',
                  'minimum_nights', 'number_of_reviews', 'review_scores_rating', 'availability_365']]
    y = df_model['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
    col2.metric("MAE", f"${mean_absolute_error(y_test, y_pred):,.2f} MXN")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_title("Precio Real vs Predicho")
    ax.set_xlabel("Precio Real (MXN)")
    ax.set_ylabel("Precio Predicho (MXN)")
    st.pyplot(fig)
else:
    st.warning("Selecciona más filtros para correr el modelo.")

st.divider()
st.markdown("**Propuesta de negocio:** Herramienta de pricing inteligente para anfitriones de Airbnb CDMX basada en datos reales.")
