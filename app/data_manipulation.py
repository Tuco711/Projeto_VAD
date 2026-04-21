import pandas as pd
import plotly.io as pio
import numpy as np
from pathlib import Path
import plotly.express as px

# Global settings
pio.templates.default = "plotly_white"

FILEPATH = Path('.\\datasets\\owid-covid-data.csv')

def load_full_covid_data(filepath: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)

        # Criar variaveis de tempo
        df['date'] = pd.to_datetime(df['date'])

        # Criar as colunas de periodicidade
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp()

        # Criar Semestre (S1 para meses 1-6, S2 para meses 7-12)
        df['semester'] = np.where(df['date'].dt.month <= 6,
                                df['date'].dt.year.astype(str) + '-01-01',
                                df['date'].dt.year.astype(str) + '-07-01')
        df['semester'] = pd.to_datetime(df['semester'])

        return df
    
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {filepath}")
        return None

def load_health_covid_data(filepath: Path) -> pd.DataFrame:
    df = load_full_covid_data(filepath)
    if df is not None:
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
        return df_health
    else:
        return None
    
def prepare_map_data(df_health: pd.DataFrame) -> pd.DataFrame:
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

    return df_map

def generate_map_metadata(metric: str = 'cfr_pct') -> dict:
    if metric == 'cfr_pct':
        return {
            'title': 'Mapa Mundial - CFR (%) no Ultimo Registo por Pais',
            'color_scale': 'YlOrRd',
            'color_label': 'CFR (%)'
        }
    elif metric == 'total_cases':
        return {
            'title': 'Mapa Mundial - Total de Casos no Ultimo Registo por Pais',
            'color_scale': 'Blues',
            'color_label': 'Total Cases'
        }
    else:
        return {
            'title': 'Mapa Mundial - Métrica Desconecida',
            'color_scale': 'Greys',
            'color_label': metric
        }

def generate_map_figure(df_map: pd.DataFrame, metric: str):
    metadata = generate_map_metadata(metric)
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
    color_continuous_scale=metadata['color_scale'],
    title=metadata['title'],
    labels={metric: metadata.get('color_label', metric) },
    )

    fig.update_layout(
        template='plotly_white',
        margin=dict(t=60, l=10, r=10, b=10)
    )

    return fig