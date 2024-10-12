import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
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