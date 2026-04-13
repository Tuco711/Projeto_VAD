import dash
from dash import html, dcc
import plotly.express as px
import plotly.express as px
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

# df = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
df = pd.read_csv('.\\datasets\\owid-covid-data.csv')

df_health = df[
[
'iso_code',
'continent',
'location',
'date',
'new_cases',
'total_cases', # acumulado
'new_deaths',
'total_deaths', # acumulado
'new_cases_smoothed',
'new_deaths_smoothed',
'total_cases_per_million',
'total_deaths_per_million',
# VACINAÇÃO
'total_vaccinations',
'people_vaccinated',
'people_fully_vaccinated',
]
]

# Escolha a metrica: 'cfr_pct' ou 'total_cases'
metric = 'total_cases'

# Ultimo registo por pais para um mapa comparavel
df_map = df_health.copy()
df_map['date'] = pd.to_datetime(df_map['date'])
df_map = df_map[
    df_map['continent'].notna()
    & (~df_map['location'].isin(['World', 'International', 'European Union']))
].copy()

df_map = (
    df_map.sort_values('date')
    .groupby('location', as_index=False)
    .tail(1)
)

# Calcula CFR (%)
df_map['cfr_pct'] = np.where(
    df_map['total_cases'] > 0,
    (df_map['total_deaths'] / df_map['total_cases']) * 100,
    np.nan
)

# Mantem apenas codigos ISO validos para paises
df_map = df_map[
    df_map['iso_code'].astype(str).str.len() == 3
].copy()

if metric == 'cfr_pct':
    title = 'Mapa Mundial - CFR (%) no Ultimo Registo por Pais'
    color_scale = 'YlOrRd'
    color_label = 'CFR (%)'
else:
    title = 'Mapa Mundial - Total de Casos no Ultimo Registo por Pais'
    color_scale = 'Blues'
    color_label = 'Total Cases'

fig = px.choropleth(
    df_map,
    locations='iso_code',
    color=metric,
    hover_name='location',
    hover_data={
        'date': True,
        'total_cases': ':,.0f',
        'total_deaths': ':,.0f',
        'cfr_pct': ':.2f'
    },
    color_continuous_scale=color_scale,
    title=title,
    labels={metric: color_label}
)

fig.update_layout(
    template='plotly_white',
    margin=dict(t=60, l=10, r=10, b=10)
)



app.layout = html.Div(children=[
    html.H1(children='CoVID-19 Dashboard', style={'textAlign': 'center'}),
    html.Div(children=[
        dcc.Graph(
            figure=fig)])])
     

if __name__ == '__main__':
    app.run(debug=True)