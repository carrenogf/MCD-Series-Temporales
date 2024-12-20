import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as tsa
from pmdarima.arima.utils import ndiffs
from pmdarima.arima import ADFTest
import numpy as np



st.set_page_config(layout="wide")
st.title("Análisis de Series Temporales")  
    
# Load data
df = pd.read_excel("dataset/series_diarias.xlsx")
df = df.sort_values("FECHA",ascending=True)
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


# funciones
@st.dialog("Guardar columna nueva")
def guardar_columna(df):
    new_column = st.text_input("Ingrese el nombre de la columna")
    if st.button("guardar"):
        dataframe = pd.read_excel("dataset/series_diarias.xlsx")
        dataframe = dataframe.sort_values("FECHA")
        dataframe.set_index("FECHA", inplace=True)
        if new_column:
            if new_column not in df.columns:
                df[new_column] = df[option]
                df = df[[new_column]]
                dataframe = pd.merge(dataframe, df, left_index=True, right_index=True)
                dataframe.to_excel("dataset/series_diarias.xlsx")
                st.rerun()
            else:
                st.write("El nombre de la columna ya existe")



# filtrar serie y eliminar NAs
with columns0[0]:
    option = st.selectbox(
        "Elegir Serie",
        series
    )
    
if option:
    with columns0[1]: # ajuste logaritmico
        ajuste1 = st.radio("Ajuste 1", ["Ninguno", "Logarítmico"])
        
        if ajuste1 == "Logarítmico":
            df[option] = np.log(df[option])
        
    with columns0[2]: # ajuste por infalcion con uva o tipo de cambio
        ajuste2 = st.radio("Ajuste 2", ["Ninguno", "VALOR_UVA","TC_MINORISTA"])
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
    with columns0[4]:
        st.write("Guardar datos ajustados como columna nueva en el dataset")
        if st.button("Guadar columna"):
            guardar_columna(df)


        
        
    df = df[option] # filtra la serie desde lo seleccionado por el usuario
    df = df.dropna() # elimina los NAs

    # gráfico de la serie
    with st.container():
        fig = px.line(df, title=f"Serie Diaria de {option}")
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
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
            if period_default > int(len(df)/2):
                period_default = int(len(df)/2)
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
            acf_difs = st.number_input("acf_Diferencias", min_value=0, max_value=100, value=0)
            if acf_difs > 0:
                df_dif = df[option].diff(acf_difs).dropna()
                plot1 = px.line(df_dif, title=f"{option} con {acf_difs} diferencias")
                plot1.update_layout(legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ))
                st.plotly_chart(plot1)
        with columns3[1]:
            if acf_difs > 0:
                acf = plot_acf(df_dif,lags=acf_lags, title=f"ACF de {option} con {acf_difs} diferencias")
                st.pyplot(acf)
            else:
                acf = plot_acf(df[option],lags=acf_lags, title=f"ACF de {option}")
                st.pyplot(acf)
                
    with st.container():
        st.subheader("Autocorrelación Parcial (PACF)")
        columns4 = st.columns(2)
        with columns4[0]:
            pacf_lags = st.number_input("pacf_Lags", min_value=1, max_value=100, value=30)
            pacf_difs = st.number_input("pacf_Diferencias", min_value=0, max_value=100, value=0)
            if pacf_difs > 0:
                df_dif = df[option].diff(pacf_difs).dropna()
                plot2 = px.line(df_dif, title=f"{option} con  {pacf_difs} diferencias")
                plot2.update_layout(legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ))
                st.plotly_chart(plot2)
        with columns4[1]:
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
        columns5 = st.columns(2)

        # Configurar el número de desfases (lags) y diferencias
        with columns5[0]:
            acov_lags = st.number_input("acov lags", min_value=1, max_value=100, value=30)
            acov_difs = st.number_input("acov diferencias", min_value=0, max_value=100, value=0)
            
            # Aplicar las diferencias si es necesario
            if acov_difs > 0:
                df_dif = df[option].diff(acov_difs).dropna()
                plot = px.line(df_dif, title=f" {option} con {acov_difs} diferencias")
                plot.update_layout(legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ))
                st.plotly_chart(plot)
            else:
                df_dif = df[option]
        
        # Mostrar el gráfico de autocovarianza
        with columns5[1]:
            # Calcular la autocovarianza para cada lag
            autocov = [autocovarianza(df_dif, lag) for lag in range(1, acov_lags+1)]
            
            # Graficar la autocovarianza
            plt.figure(figsize=(10, 6))
            plt.stem(range(1, acov_lags+1), autocov)
            plt.title(f"Autocovarianza de {option} con {acov_difs} diferencias")
            plt.xlabel("Lags")
            plt.ylabel("Autocovarianza")
            st.pyplot(plt)
            
    with st.container():
        
        def print_test_afd(y):
            st.write("Augmented Dickey-Fuller unit root test.")
            resultado = tsa.adfuller(y)
            st.write(f"""Estadistico ADF: {resultado[0]}\n
            p-valor: {resultado[1]}\n
            valores criticos: {resultado[4]}
            """)


        def dickey_fuller_tests(train):
            # Test sin término independiente ni lineal ("None")
            result_none = tsa.adfuller(train, maxlag=None, regression='n', autolag='AIC', store=False, regresults=False)
            
            # Test con término independiente pero sin término lineal ("Drift")
            result_drift = tsa.adfuller(train, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
            
            # Test con ambos términos ("Trend")
            result_trend = tsa.adfuller(train, maxlag=None, regression='ct', autolag='AIC', store=False, regresults=False)
            
            # Mostrar resultados en Streamlit
            st.write("## Test de Raíces Unitarias - Dickey Fuller")
            
            # Sin término independiente ni lineal
            st.write("### Sin término independiente ni lineal ('None')")
            st.write(f"ADF Statistic: {result_none[0]}")
            st.write(f"p-value: {result_none[1]}")
            st.write(f"Lags used: {result_none[2]}")
            st.write(f"Number of observations: {result_none[3]}")
            st.write(f"Critical Values: {result_none[4]}")
            
            # Con término independiente pero sin término lineal ("Drift")
            st.write("### Con término independiente pero sin término lineal ('Drift')")
            st.write(f"ADF Statistic: {result_drift[0]}")
            st.write(f"p-value: {result_drift[1]}")
            st.write(f"Lags used: {result_drift[2]}")
            st.write(f"Number of observations: {result_drift[3]}")
            st.write(f"Critical Values: {result_drift[4]}")
            
            # Con ambos términos ("Trend")
            st.write("### Con ambos términos ('Trend')")
            st.write(f"ADF Statistic: {result_trend[0]}")
            st.write(f"p-value: {result_trend[1]}")
            st.write(f"Lags used: {result_trend[2]}")
            st.write(f"Number of observations: {result_trend[3]}")
            st.write(f"Critical Values: {result_trend[4]}")

        def test_stationarity(timeseries):
            # Rolling statistics
            rolmean = timeseries.rolling(12).mean()
            rolstd = timeseries.rolling(12).std()

            # Plot rolling statistics:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timeseries, color='blue', label='Original')
            ax.plot(rolmean, color='red', label='Rolling Mean')
            ax.plot(rolstd, color='black', label='Rolling Std')
            ax.legend(loc='best')
            ax.set_title('Rolling Mean & Standard Deviation')

            # Mostrar gráfico en Streamlit
            st.pyplot(fig)
            

        def estacionario(y):
            # Estimado de número de diferencias con ADF test: Dickey-Fuller
            n_adf = ndiffs(y, test='adf')

            # KPSS test (auto_arima default): Kwiatkowski-Phillips-Schmidt-Shin
            n_kpss = ndiffs(y, test='kpss')

            # PP test: Phillips-Perron
            n_pp = ndiffs(y, test='pp')

            # Mostrar resultados en Streamlit
            st.write('### Estimado de número de diferencias:')
            st.write(f"- ADF test: {n_adf}")
            st.write(f"- KPSS test: {n_kpss}")
            st.write(f"- PP test: {n_pp}")

            # Se debe realizar diferenciación con ADF Test
            adftest = ADFTest(alpha=0.05)
            should_diff, p_value = adftest.should_diff(y)
            
            st.write('### ¿Se debe realizar diferenciación según ADF Test?')
            st.write(f"- Should Diff: {should_diff}")
            st.write(f"- P-value: {p_value}")
            
            
        with st.expander("Test estadísticos"):
            print_test_afd(df[option])
            dickey_fuller_tests(df[option])
            test_stationarity(df[option])
            estacionario(df[option])
        