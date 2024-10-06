import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import os


st.set_page_config(layout="wide")

# Load data
cwd = os.getcwd()
df = pd.read_excel(os.path.join(cwd,"dataset","series_diarias.xlsx"))
df.set_index("FECHA", inplace=True)
series = df.columns

# filtrar serie y eliminar NAs
option = st.selectbox(
    "How would you like to be contacted?",
    series
)
if option:
    df = df[option] # filtra la serie desde lo seleccionado por el usuario
    df = df.dropna() # elimina los NAs

    st.title("Series Diarias")
    # gráfico de la serie
    with st.container():
        fig = px.line(df, title=f"Serie Diaria de {option}")
        st.plotly_chart(fig)
        
    with st.container():
        columns1 = st.columns(3)
        with columns1[0]:
            st.write("Describe()")
            st.write(df.describe()) 
        with columns1[1]:
            hist = px.histogram(df, x=option, title=f"Histograma de {option}")
            st.plotly_chart(hist)
        with columns1[2]:
            # transformo el dataset para el boxplot
            df = df.reset_index()
            df['Year'] = pd.to_datetime(df['FECHA'],utc=True).dt.strftime('%Y')    
            boxplot = px.box(df, x='Year', y=option, title=f"Boxplot Anual de {option}", color='Year')
            st.plotly_chart(boxplot)

    # Descomposición de la serie
    with st.container():
        st.subheader("Descomposición de la serie")
        columns2 = st.columns(2)
        
        with columns2[0]:
            period_default  = int(df["Year"].value_counts().mean())
            periodos = st.number_input("Periodos", min_value=1, max_value=int(len(df)/2), value=period_default)
            model = st.selectbox("Modelo", ["additive", "multiplicative"])
            
        with columns2[1]:
            sd = seasonal_decompose(df[option], model=model, period=periodos)
            st.write(sd.plot())
            
    with st.container():
        st.subheader("Autocorrelación (ACF)")
        columns3 = st.columns(2)
        with columns3[0]:
            acf_lags = st.number_input("acf_Lags", min_value=1, max_value=100, value=30)
            acf_difs = st.number_input("acf_Diferencias", min_value=0, max_value=10, value=0)
            if acf_difs > 0:
                df_dif = df[option].diff(acf_difs).dropna()
                plot = px.line(df_dif, title=f"{option} con {acf_difs} diferencias")
                st.plotly_chart(plot)
        with columns3[1]:
            if acf_difs > 0:
                acf = plot_acf(df_dif,lags=acf_lags, title=f"ACF de {option} con {acf_difs} diferencias")
                st.pyplot(acf)
            else:
                acf = plot_acf(df[option],lags=acf_lags, title=f"ACF de {option}")
                st.pyplot(acf)
                
    with st.container():
        st.subheader("Autocorrelación Parcial (PACF)")
        columns3 = st.columns(2)
        with columns3[0]:
            pacf_lags = st.number_input("pacf_Lags", min_value=1, max_value=100, value=30)
            pacf_difs = st.number_input("pacf_Diferencias", min_value=0, max_value=10, value=0)
            if pacf_difs > 0:
                df_dif = df[option].diff(pacf_difs).dropna()
                plot = px.line(df_dif, title=f"{option} con {pacf_difs} diferencias")
                st.plotly_chart(plot)
        with columns3[1]:
            if pacf_difs > 0:
                pacf = plot_pacf(df_dif,lags=pacf_lags, title=f"PACF de {option} con {pacf_difs} diferencias")
                st.pyplot(pacf)
            else:
                pacf = plot_pacf(df[option],lags=pacf_lags, title=f"PACF de {option}")
                st.pyplot(pacf)




    # Función para calcular autocovarianza
    def autocovarianza(serie, lag):
        return serie.cov(serie.shift(lag))

    # Autocovarianza
    with st.container():
        st.subheader("Autocovarianza")
        columns4 = st.columns(2)

        # Configurar el número de desfases (lags) y diferencias
        with columns4[0]:
            acov_lags = st.number_input("Número de lags", min_value=1, max_value=100, value=30)
            acov_difs = st.number_input("Número de diferencias", min_value=0, max_value=10, value=0)
            
            # Aplicar las diferencias si es necesario
            if acov_difs > 0:
                df_dif = df[option].diff(acov_difs).dropna()
                plot = px.line(df_dif, title=f"{option} con {acov_difs} diferencias")
                st.plotly_chart(plot)
            else:
                df_dif = df[option]
        
        # Mostrar el gráfico de autocovarianza
        with columns4[1]:
            # Calcular la autocovarianza para cada lag
            autocov = [autocovarianza(df_dif, lag) for lag in range(1, acov_lags+1)]
            
            # Graficar la autocovarianza
            plt.figure(figsize=(10, 6))
            plt.stem(range(1, acov_lags+1), autocov)
            plt.title(f"Autocovarianza de {option} con {acov_difs} diferencias")
            plt.xlabel("Lags")
            plt.ylabel("Autocovarianza")
            st.pyplot(plt)