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
from pmdarima.arima import auto_arima, ndiffs, nsdiffs, ADFTest
import numpy as np
import statsmodels.api as sm

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



def multi_autocov_autocorr(dataframes, nrol=75, titulos=None):
  # Supongamos que tu arreglo de DataFrames se llama `dataframes`
  fig, axes = plt.subplots(nrows=len(dataframes), ncols=3, figsize=(18, 12))

  for i, df in enumerate(dataframes):
      # Extraemos la serie de valores
      serie = df
      
      
      # Gráfico de ACF
      sm.graphics.tsa.plot_acf(serie, lags=nrol,  ax=axes[i, 0], title=f"ACF - {df.name}")
      
      
      # Gráfico de PACF
      sm.graphics.tsa.plot_pacf(serie, ax=axes[i, 1], title=f"PACF - {df.name}")
      
      # Gráfico de Autocovarianza
      autocovarianza = tsa.acovf(serie, fft=False, nlag=nrol)
      axes[i, 2].plot(autocovarianza, marker='o', linestyle='--')
      axes[i, 2].set_title(f"Autocovarianza - {df.name}")
      axes[i, 2].set_xlabel("Lags")
      axes[i, 2].set_ylabel("Autocovarianza")

  # Ajustar el layout y la ubicación de las leyendas
  plt.tight_layout()
  plt.show()


## Función para dibujar juntos FAS: autocovarianzas; FAC y FACP, autocorrelación y autocorrelación parcial
def autocov_autocorr(serie_r, nrol=75,serie_titulo=""):
    fig, axes = plt.subplots(3, 1, figsize=(5, 5),dpi = 70)

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



def estacionario(y, name):
  print(name)
  # Estimado de número de diferencias con ADF test:Dickey-Fuller
  n_adf = ndiffs(y, test='adf')  # -> 0

  # KPSS test (auto_arima default): Kwiatkowski-Phillips-Schmidt-Shin
  n_kpss = ndiffs(y, test='kpss')  # -> 0

  # PP test: Phillips-Perron
  n_pp = ndiffs(y, test='pp')  # -> 0

  print('Estimado de número de diferencias con ADF test')
  print(n_adf)

  print('Estimado de número de diferencias con KPSS test')
  print(n_kpss)

  print('Estimado de número de diferencias con PP test')
  print(n_pp)

  print('Se debe realizar diferenciación (should_diff) ADF Test')
  adftest = ADFTest(alpha=0.05)
  print(adftest.should_diff(y))
  print("---------------------------------------------------------------------")
  
  
def estacionario2(series):
  result = {}
  for serie in series:
    n_adf = ndiffs(serie, test='adf')
    n_kpss = ndiffs(serie, test='kpss')
    n_pp = ndiffs(serie, test='pp')
    adftest = ADFTest(alpha=0.05)
    should_diff = adftest.should_diff(serie)
    result[serie.name] = {
        'ADF': n_adf,
        'KPSS': n_kpss,
        'PP': n_pp,
        'Should Diff ADFtest': should_diff
    }
  return pd.DataFrame(result).T


def test_stationarity(series):
    fig, axes = plt.subplots(ncols=len(series), nrows=1, figsize=(20, 5))
    i = 0
    for serie in series:
        # Determing rolling statistics
        rolmean = serie.rolling(25).mean()
        rolstd = serie.rolling(25).std()

        # Plot rolling statistics:
        
        ax = axes[i]
        ax.plot(serie, color='#75a2e0', label='Original')
        ax.plot(rolmean, color='#d12642', label='Media Móvil 25 periodos')
        ax.plot(rolstd, color='#111163', label='Desvio Estandar Movil 25 periodos')
        # Plot linear trend of rolling mean
        z = np.polyfit(range(len(rolmean.dropna())), rolmean.dropna(), 1)
        p = np.poly1d(z)
        ax.plot(rolmean.index, p(range(len(rolmean))), color='green', linestyle='--', label='Tendencia Media Móvil')
        ax.legend(loc='best')
        ax.set_title(f'{serie.name} \n Media Móvil & Desvio Estandar Movil')
        i += 1
    fig.show()
        