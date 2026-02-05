# Time Series Forecasting - S&P 500

Este notebook implementa e compara múltiplos modelos de previsão de séries temporais aplicados ao mercado de ações americano (S&P 500) usando a biblioteca sktime.

## 📋 Descrição

O projeto demonstra a aplicação e comparação de cinco diferentes modelos de previsão:

- 📊 **Naive Forecaster** - Baseline simples
- 🔄 **AutoARIMA** - Modelo autorregressivo integrado de médias móveis
- 📈 **AutoETS** - Suavização exponencial automática
- 🔗 **VAR** - Modelo vetorial autorregressivo (multivariado)
- 🎯 **Prophet** - Modelo de previsão desenvolvido pelo Facebook

## 🛠️ Requisitos

```bash
pip install sktime
pip install pmdarima
pip install statsmodels
pip install kagglehub
```

Bibliotecas utilizadas:
- pandas
- numpy
- matplotlib
- sktime
- statsmodels
- pmdarima
- kagglehub

## 📁 Dataset

**S&P 500 Finance Data**
- Fonte: Kaggle (`awadhi123/finance-data-sp-500`)
- Arquivo: `SP500.csv`
- Período: Dados históricos do índice S&P 500
- Frequência: Diária (dias úteis)

## 🔄 Fluxo de Trabalho

### 1. Carregamento e Preparação dos Dados

```python
# Carregar dataset do Kaggle
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "awadhi123/finance-data-sp-500",
    "SP500.csv"
)

# Converter coluna de data para datetime
df['Date'] = pd.to_datetime(df['Date'])

# Definir data como índice
df.set_index('Date', inplace=True)

# Extrair coluna de fechamento (variável alvo)
y = df['Close']
```

### 2. Tratamento de Dados Faltantes

```python
# Ajustar frequência para dias úteis e preencher fins de semana
y = y.asfreq('B').fillna(method='ffill')
```

> **Nota:** O mercado é fechado em fins de semana, então os valores são preenchidos repetindo o último valor disponível.

### 3. Análise de Decomposição

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(y.tail(500), model='additive', period=5)
result.plot()
```

A decomposição analisa:
- **Tendência** (Trend): Direção geral dos preços
- **Sazonalidade** (Seasonal): Padrões repetitivos
- **Resíduos** (Residual): Variações não explicadas

### 4. Divisão Treino/Teste

```python
y_train = y.iloc[:-30]  # Todos os dados exceto últimos 30 dias
y_test = y.iloc[-30:]    # Últimos 30 dias para teste
```

## 🤖 Modelos Implementados

### 1. Naive Forecaster

Modelo baseline que simplesmente repete o último valor observado.

```python
from sktime.forecasting.naive import NaiveForecaster

modelo = NaiveForecaster(strategy='last')
modelo.fit(y_train)
y_pred_naive = modelo.predict(fh=np.arange(1, 31))
```

**Características:**
- Simples e rápido
- Útil como baseline de comparação
- Assume que o futuro será igual ao presente

### 2. AutoARIMA

Encontra automaticamente os melhores parâmetros ARIMA (p, d, q).

```python
from sktime.forecasting.arima import AutoARIMA

modelo = AutoARIMA(sp=1)
modelo.fit(y_train.tail(500))
y_pred_arima = modelo.predict(fh=np.arange(1, 31))
```

**Características:**
- Automatiza seleção de parâmetros
- Captura tendências e padrões autorregressivos
- Bom para séries com tendência clara

### 3. AutoETS

Suavização exponencial com seleção automática de componentes (Error, Trend, Seasonal).

```python
from sktime.forecasting.ets import AutoETS

modelo = AutoETS(auto=True, sp=5)
modelo.fit(y_train.tail(500))
y_pred_ets = modelo.predict(fh=np.arange(1, 31))
```

**Características:**
- Suavização exponencial adaptativa
- Seleciona melhor combinação de componentes
- Eficiente para séries com sazonalidade

### 4. VAR (Vector Autoregression)

Modelo multivariado que considera múltiplas variáveis interdependentes.

```python
from sktime.forecasting.var import VAR

# Usa Close e Volume como variáveis
data = df[['Close', 'Volume']]
data = data.asfreq('B').ffill()

# Diferenciação para estacionariedade
data_diff = data.diff().dropna()

modelo = VAR(maxlags=15)
modelo.fit(treino_diff)
y_pred_var = modelo.predict(fh=np.arange(1, 31))
```

**Características:**
- Considera múltiplas variáveis simultaneamente
- Captura interdependências entre variáveis
- Requer diferenciação para estacionariedade

### 5. Prophet

Modelo desenvolvido pelo Facebook baseado em ajuste de curvas.

```python
from sktime.forecasting.fbprophet import Prophet

modelo = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    changepoint_prior_scale=0.5
)
modelo.fit(y_train)
y_pred_prophet = modelo.predict(fh=np.arange(1, 31))
```

**Características:**
- Robusto a dados faltantes
- Captura múltiplos níveis de sazonalidade
- Detecta mudanças de tendência automaticamente
- **Melhor performance geral** segundo as métricas

## 📊 Métricas de Avaliação

O notebook calcula cinco métricas principais para cada modelo:

```python
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_scaled_error
)
```

### Métricas Explicadas

| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| **MAPE** | Mean Absolute Percentage Error | Erro percentual médio - quanto menor, melhor |
| **MAE** | Mean Absolute Error | Erro médio em dólares - magnitude do erro |
| **RMSE** | Root Mean Squared Error | Penaliza erros grandes - sensível a outliers |
| **MSE** | Mean Squared Error | Erro quadrático médio |
| **MASE** | Mean Absolute Scaled Error | **< 1**: modelo melhor que naive<br>**> 1**: modelo pior que naive |

### Exemplo de Avaliação

```python
modelos = {
    'Naive': y_pred_naive,
    'AutoARIMA': y_pred_arima,
    'AutoETS': y_pred_ets,
    'VAR': y_pred_var,
    'Prophet': y_pred_prophet
}

for nome, pred in modelos.items():
    print(f"{nome}:")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
    print(f"MASE: {mean_absolute_scaled_error(y_test, pred, y_train=y_train):.4f}")
    print()
```

## 🎯 Resultados

Segundo as métricas de avaliação, o modelo **Prophet** demonstra performance superior em relação aos demais modelos testados, apresentando:
- Menor MAPE (erro percentual)
- Menor MAE (erro absoluto)
- MASE inferior a 1 (melhor que baseline)

## 📈 Visualização

Todas as previsões são visualizadas usando a função `plot_series` do sktime:

```python
from sktime.utils.plotting import plot_series

plot_series(
    y_train.tail(100),
    y_test,
    y_pred,
    labels=['treino', 'real', 'predição']
)
```

Cada gráfico mostra:
- **Azul**: Últimos 100 dias de treino
- **Laranja**: Valores reais (teste)
- **Verde**: Previsões do modelo

## 💡 Insights

### Tratamento de Dados
- Mercado fechado em fins de semana requer preenchimento forward-fill
- Frequência de dias úteis ('B') é essencial para dados financeiros

### Comparação de Modelos
- **Naive**: Útil como baseline, mas limitado
- **ARIMA/ETS**: Bons para tendências, mas podem não capturar volatilidade
- **VAR**: Considera volume, mas requer mais dados
- **Prophet**: Mais robusto a mudanças e sazonalidade

### Decomposição Temporal
- Período de 5 dias captura padrão semanal
- Resíduos indicam volatilidade do mercado
- Tendência mostra direção de longo prazo

## 🚀 Como Usar

1. **Configure o ambiente:**
```bash
pip install sktime pmdarima statsmodels kagglehub
```

2. **Execute o notebook:**
- Carregue os dados do Kaggle
- Prepare e visualize os dados
- Treine cada modelo
- Compare as métricas

3. **Escolha o melhor modelo:**
- Analise as métricas MASE, MAPE e MAE
- Considere o contexto do problema
- Valide com novos dados

## ⚠️ Limitações

- Modelos estatísticos podem não capturar eventos extremos
- Mercado financeiro é altamente volátil e imprevisível
- Dados passados não garantem performance futura
- Recomenda-se validação constante e retreinamento

## 🔗 Links Úteis

- [sktime Documentation](https://www.sktime.net/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [ARIMA Guide](https://otexts.com/fpp2/arima.html)
- [Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)

## 📄 Licença

Este notebook está disponível no GitHub: [Time-series](https://github.com/Pedroct06/Time-series)

---

**Nota:** Este projeto é apenas para fins educacionais. Não deve ser usado como base única para decisões de investimento.
