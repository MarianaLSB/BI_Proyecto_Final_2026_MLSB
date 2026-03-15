import pandas as pd
import streamlit as st

class AirbnbService:

    def __init__(self):
        self.url = "https://data.insideairbnb.com/mexico/df/mexico-city/2025-09-27/data/listings.csv.gz"

    @st.cache_data(ttl=3600)
    def get_full_data(_self):
        """
        Obtiene, limpia y prepara los datos de Airbnb CDMX.
        """
        try:
            cols = ['id', 'name', 'neighbourhood_cleansed', 'room_type', 'accommodates',
                    'bedrooms', 'beds', 'price', 'minimum_nights',
                    'number_of_reviews', 'review_scores_rating', 'host_is_superhost',
                    'availability_365']

            df = pd.read_csv(_self.url, compression='gzip', usecols=cols)

            # Limpiar precio
            df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
            df = df[df['price'] > 0]
            df = df[df['price'] <= df['price'].quantile(0.99)]

            # Imputar nulos
            df['bedrooms'] = df['bedrooms'].fillna(1)
            df['beds'] = df['beds'].fillna(1)
            df['bathrooms_text'] = 'N/A'
            df['review_scores_rating'] = df['review_scores_rating'].fillna(
                df['review_scores_rating'].median())
            df['host_is_superhost'] = df['host_is_superhost'].fillna('f')

            return df

        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return pd.DataFrame()
