import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from io import StringIO  # Capturar la salida de consola
import contextlib  # Redirigir la salida de consola

st.set_page_config(layout="wide")
st.title("Análisis de Series Temporales con auto_arima")

# Cargar datos
df = pd.read_excel("dataset/series_diarias.xlsx")
df = df.sort_values("FECHA", ascending=True)
df.set_index("FECHA", inplace=True)
columnas = list(df.columns)
series = [serie for serie in columnas if serie not in ["VALOR_UVA", "TC_MINORISTA"]]
columns0 = st.columns(4)

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

    df = df[option].dropna()

    # Gráfico de la serie
    with st.container():
        fig = px.line(df, title=f"Serie Diaria de {option}")
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig)

    # Selección del porcentaje para el train/test split
    st.subheader("División en Train/Test")
    train_size = st.slider("Porcentaje del dataset para entrenamiento", min_value=50, max_value=90, value=80) / 100

    # Selección del número máximo de trials y estacionalidad (m)
    max_trials = st.number_input("Número máximo de intentos (trials)", min_value=1, max_value=100, value=10, step=1)
    m = st.number_input("Estacionalidad (m)", min_value=1, max_value=365, value=7, step=1)

    # División de los datos en conjuntos de entrenamiento y prueba
    train_data = df[:int(len(df) * train_size)]
    test_data = df[int(len(df) * train_size):]

    # Ajuste del modelo auto_arima
    if st.button("Ajustar Modelo auto_arima"):
        try:
            # Capturar la salida de consola en un buffer
            buffer = StringIO()
            with contextlib.redirect_stdout(buffer):
                model = auto_arima(
                    train_data, 
                    seasonal=True, 
                    m=m,  # Estacionalidad seleccionada por el usuario
                    trace=True,  # Muestra los intentos en la consola
                    error_action='ignore',  
                    suppress_warnings=True,
                    maxiter=max_trials  # Número máximo de trials
                )

            st.success("Modelo ajustado con éxito.")
            st.text(f"Parámetros seleccionados:\n{model.summary()}")

            # Mostrar los intentos en un desplegable
            trace_log = buffer.getvalue()
            with st.expander("Intentos del modelo (auto_arima)"):
                st.text(trace_log)

            # Predicciones sobre el conjunto de prueba
            forecast = model.predict(n_periods=len(test_data))

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

            # Gráfico con los datos reales y predicciones
            fig, ax = plt.subplots()
            ax.plot(train_data.index, train_data, label="Datos Reales (Train)", color='blue')
            ax.plot(test_data.index, test_data, label="Datos Reales (Test)", color='green')
            ax.plot(test_data.index, forecast, color='red', label="Predicciones (Test)")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al ajustar el modelo auto_arima: {e}")
