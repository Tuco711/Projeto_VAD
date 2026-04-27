import pandas as pd
import plotly.io as pio
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Global settings
pio.templates.default = "plotly_white"

FILEPATH = Path('.\\datasets\\owid-covid-data.csv')
GDP_FILEPATH = Path('.\\datasets\\GDP_data.csv')


def _extract_gdp_year_columns(df_gdp: pd.DataFrame) -> dict:
    year_columns = {}
    for column in df_gdp.columns:
        if len(column) >= 4 and column[:4].isdigit():
            year_columns[int(column[:4])] = column
    return dict(sorted(year_columns.items()))


def generate_gdp_alluvial_figure(filepath: Path = GDP_FILEPATH, top_n: int = 12, allowed_country_names=None):
    df_gdp = pd.read_csv(filepath)
    year_columns = _extract_gdp_year_columns(df_gdp)

    selected_years = [2019, 2022, 2024]
    missing_years = [year for year in selected_years if year not in year_columns]
    if missing_years:
        raise ValueError(
            'GDP_data.csv precisa conter as colunas de 2019, 2022 e 2024 para o diagrama alluvial.'
        )

    selected_columns = [
        'Country Name',
        'Country Code',
        *(year_columns[year] for year in selected_years),
    ]
    df_selected = df_gdp[selected_columns].copy()

    df_long = df_selected.melt(
        id_vars=['Country Name', 'Country Code'],
        var_name='year_column',
        value_name='gdp_usd',
    )

    column_to_year = {year_columns[year]: year for year in selected_years}
    df_long['year'] = df_long['year_column'].map(column_to_year)
    df_long['gdp_usd'] = pd.to_numeric(df_long['gdp_usd'], errors='coerce')
    df_long = df_long.dropna(subset=['gdp_usd'])
    df_long = df_long[df_long['gdp_usd'] > 0].copy()

    if allowed_country_names is not None:
        allowed_country_names = {str(name) for name in allowed_country_names if pd.notna(name)}
        df_long = df_long[df_long['Country Name'].isin(allowed_country_names)].copy()

    first_year = selected_years[0]
    middle_year = selected_years[1]
    latest_year = selected_years[-1]
    top_countries = (
        df_long[df_long['year'] == latest_year]
        .sort_values('gdp_usd', ascending=False)
        .head(top_n)['Country Code']
        .tolist()
    )

    df_long = df_long[df_long['Country Code'].isin(top_countries)].copy()

    country_order = df_long[df_long['year'] == latest_year].sort_values('gdp_usd', ascending=False)['Country Name'].tolist()

    node_keys = []
    node_labels = []
    node_colors = []
    node_x = []
    node_y = []
    palette = px.colors.sample_colorscale('Blues', [0.35 + 0.45 * i / max(len(country_order) - 1, 1) for i in range(len(country_order))])
    color_by_country = {country: palette[index] for index, country in enumerate(country_order)}

    year_positions = {
        first_year: 0.01,
        middle_year: 0.50,
        latest_year: 0.99,
    }
    y_positions = [0.02 + (0.96 * i / max(len(country_order) - 1, 1)) for i in range(len(country_order))]

    for year in selected_years:
        for index, country in enumerate(country_order):
            node_keys.append((year, country))
            node_labels.append(country if year == latest_year else '')
            node_colors.append(color_by_country[country])
            node_x.append(year_positions[year])
            node_y.append(y_positions[index])

    node_index = {key: index for index, key in enumerate(node_keys)}

    values_by_year = (
        df_long.pivot_table(
            index=['Country Code', 'Country Name'],
            columns='year',
            values='gdp_usd',
            aggfunc='first',
        )
        .reset_index()
    )

    values_by_year = values_by_year[values_by_year['Country Code'].isin(top_countries)].copy()

    sources = []
    targets = []
    values = []
    custom_text = []

    year_pairs = list(zip(selected_years[:-1], selected_years[1:]))
    for start_year, end_year in year_pairs:
        start_values = values_by_year[['Country Code', 'Country Name', start_year, end_year]].copy()
        for _, row in start_values.iterrows():
            country_name = row['Country Name']
            start_gdp = row[start_year]
            end_gdp = row[end_year]
            if pd.isna(start_gdp) or pd.isna(end_gdp):
                continue

            sources.append(node_index[(start_year, country_name)])
            targets.append(node_index[(end_year, country_name)])
            values.append(float(start_gdp))
            custom_text.append(f'{country_name}: {start_year} = {start_gdp:,.0f} USD, {end_year} = {end_gdp:,.0f} USD')

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=10,
                    thickness=12,
                    line=dict(color='rgba(30, 41, 59, 0.25)', width=0.6),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    align='left',
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color='rgba(37, 99, 235, 0.22)',
                    customdata=custom_text,
                    hovertemplate='%{customdata}<extra></extra>',
                ),
                valueformat=',d',
            )
        ]
    )

    layout_height = 700
    margin_top = 72
    margin_bottom = 8
    year_label_y = 1.02

    fig.update_layout(
        title=None,
        template='plotly_white',
        height=layout_height,
        margin=dict(t=margin_top, l=8, r=8, b=margin_bottom),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        annotations=[
            dict(x=year_positions[first_year], y=year_label_y, xref='paper', yref='paper', text=str(first_year), showarrow=False, font=dict(size=14, color='#0f172a')),
            dict(x=year_positions[middle_year], y=year_label_y, xref='paper', yref='paper', text=str(middle_year), showarrow=False, font=dict(size=14, color='#0f172a')),
            dict(x=year_positions[latest_year], y=year_label_y, xref='paper', yref='paper', text=str(latest_year), showarrow=False, font=dict(size=14, color='#0f172a')),
        ],
    )

    return fig

def load_full_covid_data(filepath: Path) -> pd.DataFrame | None:
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

def load_health_covid_data(filepath: Path) -> pd.DataFrame | None:
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

def generate_date_slider_marks(df_health: pd.DataFrame, num_marks: int = 6) -> dict:
    date_series = pd.to_datetime(df_health['date']).dropna()
    if date_series.empty:
        return {}

    start_date = date_series.min().normalize()
    end_date = date_series.max().normalize()
    marks = pd.date_range(start=start_date, end=end_date, periods=num_marks)

    return {
        int(mark.value // 10**6): mark.strftime('%b %Y')
        for mark in marks
    }

def build_map_base_data(df_health: pd.DataFrame) -> pd.DataFrame:
    df_map = df_health.copy()
    df_map['date'] = pd.to_datetime(df_map['date'])
    df_map = df_map[
        df_map['continent'].notna()
        & (~df_map['location'].isin(['World', 'International', 'European Union']))
        & (df_map['iso_code'].astype(str).str.len() == 3)
    ].copy()

    df_map['cfr_pct'] = np.where(
        df_map['total_cases'] > 0,
        (df_map['total_deaths'] / df_map['total_cases']) * 100,
        np.nan
    )

    return df_map.sort_values('date')
    
def prepare_map_data(df_health: pd.DataFrame, end_date=None, preprocessed: bool = False) -> pd.DataFrame:
    # Ultimo registo por pais para um mapa comparavel
    df_map = df_health.copy()
    df_map['date'] = pd.to_datetime(df_map['date'])
    if not preprocessed:
        df_map = df_map[
            df_map['continent'].notna()
            & (~df_map['location'].isin(['World', 'International', 'European Union']))
        ].copy()

        df_map = df_map.sort_values('date')

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_map = df_map[df_map['date'] <= end_date].copy()

    df_map = (
        df_map.groupby('location', as_index=False)
        .tail(1)
    )

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
    elif metric == 'total_deaths':
        return {
            'title': 'Mapa Mundial - Mortes Acumuladas no Ultimo Registo por Pais',
            'color_scale': 'Reds',
            'color_label': 'Mortes acumuladas'
        }
    else:
        return {
            'title': 'Mapa Mundial - Métrica Desconecida',
            'color_scale': 'Greys',
            'color_label': metric
        }

def generate_map_figure(df_map: pd.DataFrame, end_date=None, metric='total_deaths', color_max=None):
    metric = metric or 'total_deaths'
    metadata = generate_map_metadata(metric)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        if metric == 'cfr_pct':
            title = f"Mapa Mundial - CFR (%) até {end_date.strftime('%d/%m/%Y')}"
        elif metric == 'total_cases':
            title = f"Mapa Mundial - Total de Casos até {end_date.strftime('%d/%m/%Y')}"
        else:
            title = f"Mapa Mundial - Mortes Acumuladas até {end_date.strftime('%d/%m/%Y')}"
    else:
        if metric == 'cfr_pct':
            title = 'Mapa Mundial - CFR (%)'
        elif metric == 'total_cases':
            title = 'Mapa Mundial - Total de Casos'
        else:
            title = 'Mapa Mundial - Mortes Acumuladas'

    if color_max is None:
        color_max = df_map[metric].max()
    if pd.isna(color_max) or color_max <= 0:
        color_max = 1

    df_plot = df_map.copy()
    df_plot['date_display'] = pd.to_datetime(df_plot['date']).dt.strftime('%d/%m/%Y')

    fig = px.choropleth(
        df_plot,
        locations='iso_code',
        color=metric,
        hover_name='location',
        custom_data=['date_display', 'total_cases', 'total_deaths', 'cfr_pct'],
        color_continuous_scale=metadata['color_scale'],
        range_color=(0, color_max),
        title=title,
        labels={metric: metadata['color_label']},
    )

    if metric == 'cfr_pct':
        hover_template = (
            '<b>%{hovertext}</b><br>'
            'Data: %{customdata[0]}<br>'
            'Casos totais: %{customdata[1]:,.0f}<br>'
            'Mortes totais: %{customdata[2]:,.0f}<br>'
            'CFR: %{customdata[3]:.2f}%<extra></extra>'
        )
    else:
        hover_template = (
            '<b>%{hovertext}</b><br>'
            'Data: %{customdata[0]}<br>'
            'Mortes acumuladas: %{customdata[2]:,.0f}<br>'
            'Casos totais: %{customdata[1]:,.0f}<br>'
            'CFR: %{customdata[3]:.2f}%<extra></extra>'
        )

    fig.update_traces(hovertemplate=hover_template)

    fig.update_layout(
        template='plotly_white',
        margin=dict(t=60, l=10, r=10, b=10)
        ,paper_bgcolor='rgba(0,0,0,0)'
        ,plot_bgcolor='rgba(0,0,0,0)'
        ,coloraxis_colorbar=dict(
            title=metadata['color_label'],
            thickness=16,
            len=0.72,
            y=0.5,
            x=1.02,
            outlinewidth=0,
        )
        ,geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgba(148, 163, 184, 0.45)',
            showland=True,
            landcolor='rgba(241, 245, 249, 0.9)',
        )
    )

    return fig