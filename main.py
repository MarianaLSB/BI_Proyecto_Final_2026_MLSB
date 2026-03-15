import streamlit as st
from Modules.Data.airbnb_service import AirbnbService
from Modules.UI.header import show_header
from Modules.Viz.viz_service import AirbnbViz

st.set_page_config(page_title="Airbnb CDMX - BI Dashboard", page_icon="🏠", layout="wide")

# Header
show_header("Airbnb CDMX — Business Intelligence Dashboard")

# Cargar datos
service = AirbnbService()
df = service.get_full_data()

if df.empty:
    st.stop()

# Sidebar filtros
st.sidebar.title("Filtros")
alcaldias = sorted(df['neighbourhood_cleansed'].unique())
alcaldia_sel = st.sidebar.multiselect("Alcaldía", alcaldias, default=alcaldias[:5])
room_types = sorted(df['room_type'].unique())
room_sel = st.sidebar.multiselect("Tipo de cuarto", room_types, default=room_types)
precio_max = st.sidebar.slider("Precio máximo (MXN)", 500, 10000, 5000, step=500)
zoom_mapa = st.sidebar.slider("Zoom del mapa", min_value=10, max_value=16, value=11, step=1)

df_filtered = df[
    (df['neighbourhood_cleansed'].isin(alcaldia_sel)) &
    (df['room_type'].isin(room_sel)) &
    (df['price'] <= precio_max)
]

# KPIs
st.subheader("Indicadores Clave")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Listings", f"{len(df_filtered):,}")
col2.metric("Precio Promedio", f"${df_filtered['price'].median():,.0f} MXN")
col3.metric("Rating Promedio", f"{df_filtered['review_scores_rating'].mean():.2f} ⭐")
col4.metric("Alcaldías", f"{df_filtered['neighbourhood_cleansed'].nunique()}")
st.divider()
viz = AirbnbViz()
viz.render_mapa(df_filtered, zoom_mapa=zoom_mapa)
st.divider()

# EDA
st.subheader("Análisis Exploratorio de Datos")

col1, col2 = st.columns(2)
with col1:
    viz.render_precio_distribucion(df_filtered)
with col2:
    viz.render_top_alcaldias(df_filtered)

col3, col4 = st.columns(2)
with col3:
    viz.render_precio_room_type(df_filtered)
with col4:
    viz.render_rating_vs_precio(df_filtered)

st.divider()

# Modelo
st.subheader("Modelo Predictivo — Regresión Lineal")
st.markdown("Predicción de precio por noche basada en características del listing.")
viz.render_modelo(df_filtered)

st.divider()

# Propuesta
st.subheader("Propuesta de Negocio")
st.markdown("""
Con base en el análisis realizado, se propone una **herramienta de pricing inteligente 
para anfitriones de Airbnb en CDMX**:

- **Alcaldías premium:** Miguel Hidalgo y Cuajimalpa tienen los precios más altos
- **Rating importa:** +$237 MXN por punto de rating adicional
- **Tipo de propiedad:** Inmueble completo vs cuarto privado = ~$700 MXN de diferencia
- **Siguiente paso:** Modelo Random Forest + NLP en descripciones para mayor precisión
""")
