from IPython.display import display, HTML
import plotly.express as px
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.stattools as tsa
from statsmodels.tsa.stattools import adfuller
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def display_col3(html1, html2, html3):
  
  html_str = """
  <div class="container-fluid" style="display: flex; flex-wrap: wrap;">
    <div class="col-md-4" style="padding: 2px; border: 1px solid #ddd;">
      """ + html1 + """
    </div>
    <div class="col-md-4" style="padding: 2px; border: 1px solid #ddd;">
      """ + html2 + """
    </div>
    <div class="col-md-4" style="padding: 2px; border: 1px solid #ddd;">
      """ + html3 + """
    </div>
  </div>
  """
  display(HTML(html_str))
  
  
def graficar_serie(serie, titulo="", xlabel="Tiempo", ylabel="Tasa", ax=None):
    """Grafica una serie en un eje específico."""
    serie.plot(ax=ax)  # Graficar la serie en el subplot dado

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend(loc='best')
    ax.grid(True, color='0.6', dashes=(5, 2, 1, 2))
    
    

# def graficar_boxplots(serie, titulo):  
#     # Convertir el índice en una columna si aún no lo has hecho
#     post_pandemia_copy = serie.copy()
#     post_pandemia_copy = post_pandemia_copy.reset_index()  # Convertir el índice en una columna llamada 'index'

#     # Asegurarse de que 'FECHA' sea de tipo datetime
#     post_pandemia_copy['FECHA'] = pd.to_datetime(post_pandemia_copy['FECHA'])

#     # Extraer el año de la columna 'FECHA'
#     post_pandemia_copy['año'] = post_pandemia_copy['FECHA'].dt.year  # Usar .dt.year para obtener el año

#     # Crear el boxplot por año
#     fig = px.box(post_pandemia_copy, x='año', y=f'{titulo}', title=f'Boxplot de {titulo}', color='año')

#     # Mostrar el gráfico
#     fig.show()


def boxplots(serie, titulo):
    post_pandemia_copy = serie.copy().reset_index()
    post_pandemia_copy['FECHA'] = pd.to_datetime(post_pandemia_copy['FECHA'])
    post_pandemia_copy['año'] = post_pandemia_copy['FECHA'].dt.year

    # Crear un boxplot con la serie dada
    fig = px.box(post_pandemia_copy, x='año', y=titulo, title=f'Boxplot de {titulo}', color='año')
    return fig

def graficar_boxplots(series, titulos):
    fig = make_subplots(rows=1, cols=3, subplot_titles=titulos)

    for i, (serie, titulo) in enumerate(zip(series, titulos), start=1):
        boxplot = boxplots(serie, titulo)
        for trace in boxplot.data:
            fig.add_trace(trace, row=1, col=i)

    fig.update_layout(height=400, width=1200, title_text="Boxplots Comparativos", showlegend=False)
    fig.show()


# def componentes(timeserie, periodo):
#   decomposition = seasonal_decompose(timeserie, model='additive', period=periodo)
#   fig = plt.figure()
#   fig = decomposition.plot()
#   fig.set_size_inches(15, 8)
#   return fig
def plotseasonal(res, axes, col):
    res.observed.plot(ax=axes[0, col], legend=False)
    res.trend.plot(ax=axes[1, col], legend=False)
    res.seasonal.plot(ax=axes[2, col], legend=False)
    res.resid.plot(ax=axes[3, col], legend=False)

def componentes(series, periodos, titulos):
    fig, axes = plt.subplots(ncols=len(series), nrows=4, figsize=(16, 12))

    component_names = ['Observed', 'Trend', 'Seasonal', 'Residual']

    for col, (serie, periodo, titulo) in enumerate(zip(series, periodos, titulos)):
        res = seasonal_decompose(serie, model='additive', period=periodo)
        plotseasonal(res, axes, col)
        
        # Añadir título a cada columna
        axes[0, col].set_title(titulo, fontsize=12)
        
        # Añadir leyendas a cada fila
        for row, name in enumerate(component_names):
            axes[row, col].set_ylabel(name, fontsize=10)
            axes[row, col].legend([name], loc='upper left')
        
        # Ajustar el eje x para mostrar las fechas correctamente
        if col == len(series) - 1:  # Solo para la última columna
            for row in range(4):
                axes[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


  
## Función para dibujar juntos FAS: autocovarianzas; FAC y FACP, autocorrelación y autocorrelación parcial
def autocov_autocorr(serie_r, nrol=75,serie_titulo=""):
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))

    plot_acf(serie_r, lags=nrol, ax=axes[0], color='blue', vlines_kwargs={"colors": 'blue'})
    axes[0].set_title(f'ACF (Autocorrelación) {serie_titulo}', fontsize=14)

    plot_pacf(serie_r, lags=nrol, ax=axes[1], color='green', vlines_kwargs={"colors": 'green'}, method='ywm')
    axes[1].set_title(f'PACF (Autocorrelación Parcial) {serie_titulo}', fontsize=14)

    axes[2].plot(tsa.acovf(serie_r, fft=False, nlag=nrol), color='red', label='AutoCov')
    axes[2].set_title(f'Autocovarianza {serie_titulo}', fontsize=14)
    axes[2].set_xlabel('Lag')

    plt.tight_layout()
    plt.show()
    

def test_dickeyfuller(series, nombres):
    # Lista para almacenar los resultados
    resultados = []

    for serie, nombre in zip(series, nombres):
        # Realizar las pruebas ADF
        adf_none = adfuller(serie, regression='n')
        adf_drift = adfuller(serie, regression='c')
        adf_trend = adfuller(serie, regression='ct')

        # Obtener niveles críticos
        crit_none = adf_none[4]
        crit_drift = adf_drift[4]
        crit_trend = adf_trend[4]

        # Verificar si se rechaza la hipótesis nula
        reject_none = adf_none[0] < crit_none['1%']
        reject_drift = adf_drift[0] < crit_drift['1%']
        reject_trend = adf_trend[0] < crit_trend['1%']

        # Guardar resultados para cada serie
        resultados.append({
            'Serie': nombre,
            'ADF (None)': adf_none[0], 'p-valor (None)': adf_none[1],
            'Críticos (None)': f"1%: {crit_none['1%'].round(3)}, 5%: {crit_none['5%'].round(3)}, 10%: {crit_none['10%'].round(3)}",
            'Rechaza H0 (None)': reject_none,

            'ADF (Drift)': adf_drift[0], 'p-valor (Drift)': adf_drift[1],
            'Críticos (Drift)': f"1%: {crit_drift['1%'].round(3)}, 5%: {crit_drift['5%'].round(3)}, 10%: {crit_drift['10%'].round(3)}",
            'Rechaza H0 (Drift)': reject_drift,

            'ADF (Trend)': adf_trend[0], 'p-valor (Trend)': adf_trend[1],
            'Críticos (Trend)': f"1%: {crit_trend['1%'].round(3)}, 5%: {crit_trend['5%'].round(3)}, 10%: {crit_trend['10%'].round(3)}",
            'Rechaza H0 (Trend)': reject_trend
        })

    # Convertir los resultados en DataFrame y transponerlo para mejor visualización
    df_resultados = pd.DataFrame(resultados).set_index('Serie').T
    return df_resultados

def Augmented_Dickey_Fuller_Test_func(series , column_name):
  print (f'Resultados de la prueba de Dickey-Fuller para columna: {column_name}')
  dftest = adfuller(series, autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','No Lags Used','Number of Observations Used'])
  for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
  print (dfoutput)
  if dftest[1] <= 0.05:
    print("Conclusion:====>")
    print("Rechazar la hipótesis nula")
    print("Los datos son estacionarios.")
  else:
    print("Conclusion:====>")
    print("No rechazar la hipótesis nula")
    print("Los datos no son estacionarios.")
    
def Augmented_Dickey_Fuller_Test_func2(series):
  resultados = {}
  for serie in series:
    dftest = adfuller(serie, autolag='AIC')
    resultados[serie.name] = {
        'Test Statistic': dftest[0],
        'p-value': dftest[1],
        'No Lags': dftest[2],
        'No Obs': dftest[3],
        'Crit Val (1%,5%,10%)': [dftest[4].get('1%').round(3), dftest[4].get('5%').round(3), dftest[4].get('10%').round(3)],
        'Estacionaria 5%': dftest[1] <= 0.05   
    }
  return pd.DataFrame(resultados).T