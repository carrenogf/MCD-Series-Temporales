import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

st.set_page_config(layout="wide")
st.title("Análisis de Series Temporales con SARIMA")

# Cargar datos
df = pd.read_excel("dataset/series_diarias.xlsx")
df = df.sort_values("FECHA", ascending=True)
df.set_index("FECHA", inplace=True)
columnas = list(df.columns)
series = [serie for serie in columnas if serie not in ["VALOR_UVA", "TC_MINORISTA"]]
columns0 = st.columns(5)

# Selección de rango de fechas
st.sidebar.subheader("Filtrar por Fechas")
fecha_min = df.index.min().date()
fecha_max = df.index.max().date()

fecha_desde = st.sidebar.date_input("Fecha Desde", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
fecha_hasta = st.sidebar.date_input("Fecha Hasta", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

# Aplicar el filtro de fechas
df = df.loc[fecha_desde:fecha_hasta]
# Filtrar serie y eliminar NAs
with columns0[0]:
    option = st.selectbox("Elegir Serie", series)

if option:
    with columns0[1]:
        ajuste1 = st.radio("Ajuste 1", ["Ninguno", "Logarítmico"])
        if ajuste1 == "Logarítmico":
            df[option] = np.log(df[option])

    with columns0[2]:
        ajuste2 = st.radio("Ajuste 2", ["Ninguno", "VALOR_UVA", "TC_MINORISTA"])
        if ajuste2 == "VALOR_UVA":
            df = df.dropna(subset=["VALOR_UVA"])
            df["original"] = df[option]
            df[option] = df[option] / df["VALOR_UVA"] * df["VALOR_UVA"].iloc[-1]
        elif ajuste2 == "TC_MINORISTA":
            df = df.dropna(subset=["TC_MINORISTA"])
            df["original"] = df[option]
            df[option] = df[option] / df["TC_MINORISTA"] * df["TC_MINORISTA"].iloc[-1]

    with columns0[3]:
        ajuste3 = st.radio("Ajuste 2", ["Ninguno", "boxcox"])
        if ajuste3 == "boxcox":
            from scipy.stats import boxcox
            df = df.dropna(subset=[option])
            df[option] = boxcox(df[option])[0]

    df = df[option]
    df = df.dropna()

    # Gráfico de la serie
    with st.container():
        fig = px.line(df, title=f"Serie Diaria de {option}")
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig)

    # Formulario para ajustar el modelo SARIMA
    st.subheader("Parámetros del modelo SARIMA")
    p = st.number_input("Parámetro p (AR)", min_value=0, max_value=10, value=1, step=1)
    d = st.number_input("Parámetro d (Diferenciación)", min_value=0, max_value=2, value=0, step=1)
    q = st.number_input("Parámetro q (MA)", min_value=0, max_value=10, value=0, step=1)

    P = st.number_input("Parámetro P (AR estacional)", min_value=0, max_value=10, value=1, step=1)
    D = st.number_input("Parámetro D (Diferenciación estacional)", min_value=0, max_value=2, value=0, step=1)
    Q = st.number_input("Parámetro Q (MA estacional)", min_value=0, max_value=10, value=0, step=1)
    s = st.number_input("Periodicidad estacional (s)", min_value=0, max_value=365, value=0, step=1)

    # Selección del porcentaje para el train/test split
    st.subheader("División en Train/Test")
    train_size = st.slider("Porcentaje del dataset para entrenamiento", min_value=50, max_value=90, value=80)

    # División de los datos en conjuntos de entrenamiento y prueba
    train_size = train_size / 100
    train_data = df[:int(len(df) * train_size)]
    test_data = df[int(len(df) * train_size):]

    # Ajuste del modelo SARIMA
    if st.button("Ajustar Modelo SARIMA"):
        try:
            # Ajustar el modelo en los datos de entrenamiento
            model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
            model_fit = model.fit(disp=False)
            st.success("Modelo ajustado con éxito.")

            # Predicciones sobre el conjunto de prueba
            forecast_result = model_fit.get_forecast(steps=len(test_data))
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()

            # Mostrar resumen del modelo
            st.text(model_fit.summary())

            # Calcular métricas
            mae = mean_absolute_error(test_data, forecast)
            mse = mean_squared_error(test_data, forecast)
            rmse = math.sqrt(mse)
            r2 = 1 - (np.sum((test_data - forecast) ** 2) / np.sum((test_data - np.mean(test_data)) ** 2))

            # Mostrar métricas en la app
            st.metric("MAE", mae)
            st.metric("MSE", mse)
            st.metric("RMSE", rmse)
            st.metric("R²", r2)

            # Gráfico con los datos reales del entrenamiento, las predicciones y los intervalos de confianza
            fig, ax = plt.subplots()
            ax.plot(train_data.index, train_data, label="Datos Reales (Train)", color='blue')
            ax.plot(test_data.index, test_data, label="Datos Reales (Test)", color='green')
            ax.plot(test_data.index, forecast, color='red', label="Predicciones (Test)")
            ax.fill_between(test_data.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confianza')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al ajustar el modelo SARIMA: {e}")
