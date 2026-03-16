# BI_Proyecto_Final_2026_MLSB
# 🏠 Airbnb CDMX — Business Intelligence Dashboard

**Alumna:** Mariana López Santibáñez Ballesteros  
**Curso:** Inteligencia de Negocios y Solución a la Ciencia de Datos  
**Universidad Panamericana | Marzo 2026**

---

## Descripción

Análisis de Business Intelligence sobre el mercado de Airbnb en Ciudad de México, utilizando datos reales de 27,051 listings scrapeados en septiembre 2025.

## 🎯 Problema de Negocio

¿Qué factores determinan el precio óptimo de un listing en Airbnb CDMX?

## 🔗 Dashboard Interactivo

👉 [proyecto-final-airbnb-mariana-lsb.streamlit.app](https://proyecto-final-airbnb-mariana-lsb.streamlit.app)

## Estructura del Proyecto
```
├── main.py                  # App principal de Streamlit
├── requirements.txt         # Dependencias
├── imagenes/                # Assets visuales
└── Modules/
    ├── Data/
    │   └── airbnb_service.py    # Carga y limpieza de datos
    ├── UI/
    │   └── header.py            # Componente de header
    └── Viz/
        └── viz_service.py       # Visualizaciones y modelo ML
```

## Modelo

- **Técnica:** Regresión Lineal (Aprendizaje Supervisado)
- **R² Score:** 0.30
- **MAE:** $587 MXN
- **Variables:** bedrooms, rating, accommodates, room_type, entre otras

## Dataset

- **Fuente:** [Inside Airbnb](https://insideairbnb.com/get-the-data/) — Mexico City
- **Fecha:** Septiembre 2025
- **Registros:** 23,331 (después de limpieza)
