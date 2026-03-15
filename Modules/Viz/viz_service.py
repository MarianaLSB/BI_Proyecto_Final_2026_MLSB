import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

class AirbnbViz:

    def render_precio_distribucion(self, df):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[df['price'] <= 5000]['price'], bins=50, color='steelblue', edgecolor='white')
        ax.axvline(df['price'].median(), color='red', linestyle='--',
                   label=f"Mediana: ${df['price'].median():,.0f}")
        ax.axvline(df['price'].mean(), color='orange', linestyle='--',
                   label=f"Media: ${df['price'].mean():,.0f}")
        ax.set_title("Distribución de Precios por Noche")
        ax.set_xlabel("Precio (MXN)")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    def render_top_alcaldias(self, df):
        top = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top.index, top.values, color='coral')
        ax.set_title("Top 10 Alcaldías por Precio Promedio")
        ax.set_xlabel("Precio Promedio (MXN)")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    def render_precio_room_type(self, df):
        room_price = df.groupby('room_type')['price'].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(room_price.index, room_price.values, color='mediumseagreen')
        ax.set_title("Precio Mediano por Tipo de Cuarto")
        ax.set_ylabel("Precio Mediano (MXN)")
        ax.tick_params(axis='x', rotation=15)
        st.pyplot(fig)
        plt.close()

    def render_rating_vs_precio(self, df):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df['review_scores_rating'], df['price'], alpha=0.3, color='mediumpurple')
        ax.set_title("Rating vs Precio")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Precio (MXN)")
        st.pyplot(fig)
        plt.close()

    def render_modelo(self, df):
        df_model = df[['room_type', 'accommodates', 'bedrooms', 'beds',
                       'minimum_nights', 'number_of_reviews',
                       'review_scores_rating', 'availability_365', 'price']].dropna()

        if len(df_model) < 100:
            st.warning("Selecciona más datos para correr el modelo.")
            return

        le = LabelEncoder()
        df_model = df_model.copy()
        df_model['room_type_enc'] = le.fit_transform(df_model['room_type'])

        X = df_model[['room_type_enc', 'accommodates', 'bedrooms', 'beds',
                      'minimum_nights', 'number_of_reviews',
                      'review_scores_rating', 'availability_365']]
        y = df_model['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        col2.metric("MAE", f"${mean_absolute_error(y_test, y_pred):,.2f} MXN")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_title(f"Precio Real vs Predicho (R²={r2_score(y_test, y_pred):.2f})")
        ax.set_xlabel("Precio Real (MXN)")
        ax.set_ylabel("Precio Predicho (MXN)")
        st.pyplot(fig)
        plt.close()

    def render_mapa(self, df):
    st.markdown("### 🗺️ Mapa de Listings en CDMX")
    
    # Colores por tipo de cuarto
    color_map = {
        'Entire home/apt': [255, 90, 95],    # rojo Airbnb
        'Private room':    [0, 166, 153],     # verde azulado
        'Hotel room':      [255, 180, 0],     # amarillo
        'Shared room':     [147, 112, 219],   # morado
    }
    
    df_mapa = df[['latitude', 'longitude', 'room_type', 'price', 'name']].dropna()
    df_mapa = df_mapa.copy()
    df_mapa['color'] = df_mapa['room_type'].map(color_map).fillna([150, 150, 150])
    
    # Leyenda manual
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("🔴 Entire home/apt")
    col2.markdown("🟢 Private room")
    col3.markdown("🟡 Hotel room")
    col4.markdown("🟣 Shared room")
    
    st.map(df_mapa, latitude='latitude', longitude='longitude', 
           color='color', size=20)
    st.caption(f"Mostrando {len(df_mapa):,} listings")
