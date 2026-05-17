import pandas as pd
import plotly.io as pio
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global settings
pio.templates.default = "plotly_white"

# Paleta de cores profissional e acessível
COLOR_PALETTE = {
    'primary': '#1e40af',      # Azul profundo
    'secondary': '#dc2626',    # Vermelho
    'accent': '#0d9488',       # Teal
    'neutral_light': '#f1f5f9',
    'neutral_dark': '#0f172a',
}

# Definir fonte padrão para toda a aplicação
FONT_FAMILY = 'Segoe UI, Roboto, Helvetica, Arial, sans-serif'

FILEPATH = Path('.\\datasets\\owid-covid-data.csv')
GDP_FILEPATH = Path('.\\datasets\\GDP_data.csv')


def _apply_style_template(fig, title_text=None, subtitle_text=None, height=450):
    """Aplica estilo consistente a todos os gráficos"""
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=dict(t=80 if subtitle_text else 70, l=30, r=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=11, color=COLOR_PALETTE['neutral_dark']),
        hovermode='x unified',
        uirevision='country-selection',
    )
    
    # Adicionar título e subtítulo se fornecidos
    if title_text or subtitle_text:
        annotations = []
        y_pos = 1.08
        
        if title_text:
            annotations.append(dict(
                text=f'<b>{title_text}</b>',
                xref='paper', yref='paper',
                x=0, y=y_pos,
                showarrow=False,
                font=dict(size=14, color=COLOR_PALETTE['neutral_dark']),
                xanchor='left'
            ))
            y_pos -= 0.06
        
        if subtitle_text:
            annotations.append(dict(
                text=subtitle_text,
                xref='paper', yref='paper',
                x=0, y=y_pos,
                showarrow=False,
                font=dict(size=10, color='#64748b'),
                xanchor='left'
            ))
        
        fig.update_layout(annotations=annotations)
    
    return fig


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
    palette = px.colors.sample_colorscale('Viridis', [0.35 + 0.45 * i / max(len(country_order) - 1, 1) for i in range(len(country_order))])
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

    layout_height = 520
    margin_top = 58
    margin_bottom = 8
    year_label_y = 1.02

    fig.update_layout(
        title=None,
        template='plotly_white',
        height=layout_height,
        margin=dict(t=margin_top, l=8, r=8, b=margin_bottom),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=10, color=COLOR_PALETTE['neutral_dark']),
        annotations=[
            dict(x=year_positions[first_year], y=year_label_y, xref='paper', yref='paper', text=str(first_year), showarrow=False, font=dict(size=14, color='#0f172a')),
            dict(x=year_positions[middle_year], y=year_label_y, xref='paper', yref='paper', text=str(middle_year), showarrow=False, font=dict(size=14, color='#0f172a')),
            dict(x=year_positions[latest_year], y=year_label_y, xref='paper', yref='paper', text=str(latest_year), showarrow=False, font=dict(size=14, color='#0f172a')),
        ],
    )

    return fig


def _resolve_country_selection(df_health: pd.DataFrame, selected_country_names=None, top_n: int = 6) -> list[str]:
    available_countries = [country for country in df_health['location'].dropna().drop_duplicates().tolist()]
    if selected_country_names:
        selected = [country for country in selected_country_names if country == 'World' or country in available_countries]
        if 'World' in selected:
            return ['World']
        if selected:
            return selected

    df_latest = prepare_map_data(df_health, preprocessed=True)
    ranking_column = 'total_deaths' if 'total_deaths' in df_latest.columns else 'total_cases'

    return (
        df_latest.sort_values(ranking_column, ascending=False)['location']
        .dropna()
        .drop_duplicates()
        .head(top_n)
        .tolist()
    )


def _collapse_to_world_when_all_countries_selected(
    df_health: pd.DataFrame,
    selected_country_names: list[str],
) -> list[str]:
    aggregate_locations = {'World', 'International', 'European Union'}
    available_countries = [
        country
        for country in df_health['location'].dropna().drop_duplicates().tolist()
        if country not in aggregate_locations
    ]

    if available_countries and set(selected_country_names) >= set(available_countries) and 'World' in df_health['location'].values:
        return ['World']

    return selected_country_names


def _prepare_covid_metrics(df_health: pd.DataFrame) -> pd.DataFrame:
    df = df_health.copy()
    if 'population' in df.columns:
        if 'new_cases_smoothed_per_million' not in df.columns and 'new_cases_smoothed' in df.columns:
            df['new_cases_smoothed_per_million'] = np.where(
                df['population'] > 0,
                (df['new_cases_smoothed'] / df['population']) * 1_000_000,
                np.nan,
            )
        if 'new_deaths_smoothed_per_million' not in df.columns and 'new_deaths_smoothed' in df.columns:
            df['new_deaths_smoothed_per_million'] = np.where(
                df['population'] > 0,
                (df['new_deaths_smoothed'] / df['population']) * 1_000_000,
                np.nan,
            )
        if 'total_vaccinations_per_hundred' not in df.columns and 'total_vaccinations' in df.columns:
            df['total_vaccinations_per_hundred'] = np.where(
                df['population'] > 0,
                (df['total_vaccinations'] / df['population']) * 100,
                np.nan,
            )
        if 'people_vaccinated_per_hundred' not in df.columns and 'people_vaccinated' in df.columns:
            df['people_vaccinated_per_hundred'] = np.where(
                df['population'] > 0,
                (df['people_vaccinated'] / df['population']) * 100,
                np.nan,
            )
        if 'people_fully_vaccinated_per_hundred' not in df.columns and 'people_fully_vaccinated' in df.columns:
            df['people_fully_vaccinated_per_hundred'] = np.where(
                df['population'] > 0,
                (df['people_fully_vaccinated'] / df['population']) * 100,
                np.nan,
            )

    return df


def generate_covid_evolution_figure(df_health: pd.DataFrame, selected_country_names=None):
    df = _prepare_covid_metrics(df_health)
    selected_country_names = _resolve_country_selection(df, selected_country_names, top_n=6)
    selected_country_names = _collapse_to_world_when_all_countries_selected(df, selected_country_names)

    required_columns = [
        'location',
        'date',
        'new_cases_smoothed_per_million',
        'new_deaths_smoothed_per_million',
        'people_fully_vaccinated_per_hundred',
    ]
    available_columns = [column for column in required_columns if column in df.columns]
    df_plot = df[df['location'].isin(selected_country_names)][available_columns].copy()
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    df_plot['month'] = df_plot['date'].dt.to_period('M').dt.to_timestamp()

    metric_specs = [
        ('new_cases_smoothed_per_million', 'Incidência de Casos', 'Número de novos casos confirmados por milhão de habitantes, suavizados em 7 dias', COLOR_PALETTE['primary']),
        ('new_deaths_smoothed_per_million', 'Mortalidade Atribuída', 'Número de mortes por COVID-19 por milhão de habitantes, suavizadas em 7 dias', COLOR_PALETTE['secondary']),
        ('people_fully_vaccinated_per_hundred', 'Cobertura de Vacinação', 'Percentual da população que completou o esquema primário de vacinação', COLOR_PALETTE['accent']),
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[f"<b>{spec[1]}</b>" for spec in metric_specs],
    )

    # Paleta de cores para os países (mais matizada)
    country_colors = [
        '#1e40af', '#dc2626', '#0d9488', '#ea580c', '#7c3aed', '#0891b2'
    ]
    color_map = {country: country_colors[index % len(country_colors)] for index, country in enumerate(selected_country_names)}

    for country in selected_country_names:
        country_df = df_plot[df_plot['location'] == country].sort_values('month')
        if country_df.empty:
            continue

        grouped = country_df.groupby('month', as_index=False)[[spec[0] for spec in metric_specs]].mean(numeric_only=True)

        for row_index, (metric_column, metric_name, metric_desc, _) in enumerate(metric_specs, start=1):
            fig.add_trace(
                go.Scatter(
                    x=grouped['month'],
                    y=grouped[metric_column],
                    mode='lines',
                    name=country,
                    legendgroup=country,
                    showlegend=(row_index == 1),
                    line=dict(color=color_map[country], width=2.5),
                    hovertemplate=f'<b>{country}</b><br>%{{x|%b %Y}}<br>%{{y:,.2f}}<extra></extra>',
                ),
                row=row_index,
                col=1,
            )

    fig.update_layout(
        template='plotly_white',
        height=720,
        margin=dict(t=120, l=30, r=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=11, color=COLOR_PALETTE['neutral_dark']),
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.04, 
            xanchor='left', 
            x=0, 
            title=None,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(200, 200, 200, 0.3)',
            borderwidth=1
        ),
        hovermode='x unified',
        uirevision='country-selection',
    )
    
    # Atualizar eixos Y com melhor formatação
    fig.update_yaxes(title_text='Casos/1M', row=1, col=1, tickformat=',d')
    fig.update_yaxes(title_text='Óbitos/1M', row=2, col=1, tickformat=',d')
    fig.update_yaxes(title_text='Cobertura %', row=3, col=1, tickformat='.1%')
    fig.update_xaxes(title_text='', row=3, col=1)
    
    # Adicionar descrições das métricas como anotações
    for i, (_, metric_name, metric_desc, _) in enumerate(metric_specs):
        fig.add_annotation(
            text=metric_desc,
            xref='paper', yref='paper',
            x=1.02, y=1 - (i * 0.33) - 0.05,
            showarrow=False,
            font=dict(size=9, color='#64748b'),
            xanchor='left',
            yanchor='top',
            align='left'
        )

    return fig


def generate_gdp_trend_figure(selected_country_names=None, filepath: Path = GDP_FILEPATH):
    df_gdp = pd.read_csv(filepath)
    year_columns = _extract_gdp_year_columns(df_gdp)
    selected_years = [year for year in range(2019, 2025) if year in year_columns]

    if not selected_years:
        raise ValueError('GDP_data.csv não contém colunas anuais compatíveis para 2019-2024.')

    if selected_country_names:
        df_filtered = df_gdp[df_gdp['Country Name'].isin(selected_country_names)].copy()
    else:
        latest_year = selected_years[-1]
        df_filtered = (
            df_gdp[['Country Name', 'Country Code', year_columns[latest_year]]]
            .dropna(subset=[year_columns[latest_year]])
            .sort_values(year_columns[latest_year], ascending=False)
            .head(6)
            .copy()
        )

    selected_columns = ['Country Name', 'Country Code', *(year_columns[year] for year in selected_years)]
    df_long = df_filtered[selected_columns].melt(
        id_vars=['Country Name', 'Country Code'],
        var_name='year_column',
        value_name='gdp_usd',
    )
    df_long['year'] = df_long['year_column'].map({year_columns[year]: year for year in selected_years})
    df_long['gdp_usd'] = pd.to_numeric(df_long['gdp_usd'], errors='coerce')
    df_long = df_long.dropna(subset=['gdp_usd'])

    # Paleta de cores melhorada
    country_colors = [
        '#1e40af', '#dc2626', '#0d9488', '#ea580c', '#7c3aed', '#0891b2'
    ]
    color_map = {country: country_colors[i % len(country_colors)] 
                 for i, country in enumerate(df_long['Country Name'].unique())}
    
    fig = go.Figure()
    
    for country in df_long['Country Name'].unique():
        country_data = df_long[df_long['Country Name'] == country].sort_values('year')
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['gdp_usd'],
            mode='lines+markers',
            name=country,
            line=dict(color=color_map[country], width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>PIB: $%{y:,.0f}<extra></extra>',
        ))

    fig.update_layout(
        template='plotly_white',
        height=450,
        margin=dict(t=80, l=50, r=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=11, color=COLOR_PALETTE['neutral_dark']),
        hovermode='x unified',
        uirevision='country-selection',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(200, 200, 200, 0.3)',
            borderwidth=1
        ),
        xaxis=dict(title='Ano', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(title='PIB (USD correntes)', tickformat=',.0f', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
    )

    return fig


def _compute_padding_for_marker_size(max_size_value, sizeref, x_data_range, y_data_range):
    """
    Compute padding in data units for axes based on marker size.
    Assumes sizemode='diameter' and standard plot dimensions.
    Returns (padding_x, padding_y).
    """
    if max_size_value <= 0 or sizeref <= 0:
        return 0, 0
    
    # Plotly formula: pixel_diameter = sqrt(2 * value / sizeref)
    marker_pixel_diameter = np.sqrt(2 * max_size_value / sizeref)
    marker_pixel_radius = marker_pixel_diameter / 2
    
    # Estimate padding: standard Plotly plot dimensions (height=450, margins ~100px → plot_height ~350px)
    # Typical aspect ratio ~ 1.3:1 → plot_width ~ 460px
    standard_plot_width = 460
    standard_plot_height = 350
    
    # Add 1.5x the marker radius for visual safety margin
    padding_px = marker_pixel_radius * 1.5
    
    # Convert to data units
    padding_x = (padding_px / standard_plot_width) * x_data_range if x_data_range > 0 else 0
    padding_y = (padding_px / standard_plot_height) * y_data_range if y_data_range > 0 else 0
    
    return padding_x, padding_y


def generate_gdp_mortality_scatter_figure(df_health: pd.DataFrame, selected_country_names=None, filepath: Path = GDP_FILEPATH):
    df = _prepare_covid_metrics(df_health)
    selected_country_names = _resolve_country_selection(df, selected_country_names, top_n=8)

    latest_columns = ['location', 'iso_code', 'continent', 'population', 'total_deaths_per_million']
    latest_columns = [column for column in latest_columns if column in df.columns]
    df_latest = prepare_map_data(df, preprocessed=True)[latest_columns].copy()
    df_gdp = pd.read_csv(filepath)
    year_columns = _extract_gdp_year_columns(df_gdp)
    if 2019 not in year_columns or 2024 not in year_columns:
        raise ValueError('GDP_data.csv precisa conter as colunas de 2019 e 2024 para calcular a variação do PIB.')

    df_gdp = df_gdp[['Country Name', 'Country Code', year_columns[2019], year_columns[2024]]].copy()
    gdp_2019 = pd.to_numeric(df_gdp[year_columns[2019]], errors='coerce')
    gdp_2024 = pd.to_numeric(df_gdp[year_columns[2024]], errors='coerce')
    df_gdp['gdp_growth_pct'] = np.where(gdp_2019 > 0, ((gdp_2024 - gdp_2019) / gdp_2019) * 100, np.nan)

    if selected_country_names == ['World']:
        df_merged = (
            df_latest[df_latest['location'] == 'World']
            .merge(df_gdp[df_gdp['Country Name'] == 'World'], left_on='location', right_on='Country Name', how='inner')
            .copy()
        )
        if not df_merged.empty:
            df_merged['continent'] = 'World'
            # If World total_deaths_per_million is NaN, compute from non-aggregate countries
            if pd.isna(df_merged.iloc[0].get('total_deaths_per_million')):
                non_agg = df_latest[~df_latest['location'].isin(['World', 'International', 'European Union'])].copy()
                if not non_agg.empty and 'total_deaths_per_million' in non_agg.columns:
                    # Get the weighted average by population
                    non_agg_clean = non_agg.dropna(subset=['total_deaths_per_million', 'population'])
                    if not non_agg_clean.empty:
                        weighted_avg = (non_agg_clean['total_deaths_per_million'] * non_agg_clean['population']).sum() / non_agg_clean['population'].sum()
                        df_merged.loc[df_merged.index[0], 'total_deaths_per_million'] = weighted_avg
    else:
        df_merged = df_latest.merge(df_gdp, left_on='iso_code', right_on='Country Code', how='inner')
        df_merged = df_merged[df_merged['location'].isin(selected_country_names)].copy()

    df_merged = df_merged.dropna(subset=['gdp_growth_pct', 'total_deaths_per_million'])

    # Prepare size series for markers (population in millions), cap extreme outliers
    if 'population' in df_merged.columns:
        size_series_gdp = (df_merged['population'] / 1_000_000).fillna(0).clip(lower=0)
        q = 0.95
        cap_value = size_series_gdp.quantile(q) if not size_series_gdp.empty else size_series_gdp.max()
        capped_max = min(size_series_gdp.max(), cap_value) if cap_value > 0 else size_series_gdp.max()
        desired_px_gdp = 15
        sizeref_gdp = 2 * capped_max / (desired_px_gdp ** 2) if capped_max > 0 else 1
        # use capped sizes for visual stability
        size_values_gdp = size_series_gdp.clip(upper=cap_value)
    else:
        size_values_gdp = pd.Series(dtype=float)
        sizeref_gdp = 1

    # Criar colormap por continente
    continent_colors = {
        'Africa': '#ef4444',
        'Americas': '#3b82f6',
        'Asia': '#10b981',
        'Europe': '#f59e0b',
        'Oceania': '#8b5cf6'
        , 'World': COLOR_PALETTE['primary']
    }
    df_merged['color'] = df_merged['continent'].map(continent_colors) if 'continent' in df_merged.columns else COLOR_PALETTE['primary']

    fig = go.Figure()
    
    for continent in (df_merged['continent'].unique() if 'continent' in df_merged.columns else [None]):
        df_continent = df_merged[df_merged['continent'] == continent] if continent else df_merged
        
        fig.add_trace(go.Scatter(
            x=df_continent['gdp_growth_pct'],
            y=df_continent['total_deaths_per_million'],
            mode='markers',
            name=continent or 'Dados',
            marker=dict(
                size=(size_values_gdp.loc[df_continent.index].values if 'population' in df_continent.columns else 10),
                color=continent_colors.get(continent, COLOR_PALETTE['primary']) if continent else COLOR_PALETTE['primary'],
                sizemode='diameter',
                sizeref=sizeref_gdp,
                line=dict(color='white', width=1.5),
                opacity=0.75
            ),
            text=df_continent['location'],
            hovertemplate='<b>%{text}</b><br>PIB crescimento: %{x:.1f}%<br>Mortes: %{y:,.0f}/1M<extra></extra>',
        ))

    # Compute padding based on largest marker size to ensure proper zoom (only if data exists)
    layout_update = dict(
        template='plotly_white',
        height=450,
        margin=dict(t=80, l=50, r=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=11, color=COLOR_PALETTE['neutral_dark']),
        hovermode='closest',
        uirevision='country-selection',
        xaxis=dict(
            title='Crescimento do PIB 2019-2024 (%)',
            zeroline=True,
            zerolinecolor='rgba(100, 100, 100, 0.2)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='Mortes acumuladas por milhão',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(200, 200, 200, 0.3)',
            borderwidth=1
        )
    )
    
    fig.update_layout(**layout_update)
    
    # Apply proportional zoom only if data exists
    if not df_merged.empty:
        x_range = df_merged['gdp_growth_pct'].max() - df_merged['gdp_growth_pct'].min()
        y_range = df_merged['total_deaths_per_million'].max() - df_merged['total_deaths_per_million'].min()
        
        # For single data point, provide reasonable padding instead of zero range
        if x_range == 0:
            x_center = df_merged['gdp_growth_pct'].iloc[0]
            x_range = 10  # Default range of ±5% for single point
        if y_range == 0:
            y_center = df_merged['total_deaths_per_million'].iloc[0]
            y_range = 500  # Default range of ±250 deaths/1M for single point
        
        pad_x, pad_y = _compute_padding_for_marker_size(capped_max, sizeref_gdp, x_range, y_range)
        x_min = df_merged['gdp_growth_pct'].min() - pad_x
        x_max = df_merged['gdp_growth_pct'].max() + pad_x
        y_min = df_merged['total_deaths_per_million'].min() - pad_y
        y_max = df_merged['total_deaths_per_million'].max() + pad_y
        
        # For single point, ensure minimum spacing
        if df_merged.shape[0] == 1:
            x_center = df_merged['gdp_growth_pct'].iloc[0]
            y_center = df_merged['total_deaths_per_million'].iloc[0]
            # Create symmetric range around the single point
            x_span = max(abs(x_center * 0.2), 5) if x_center != 0 else 5
            y_span = max(abs(y_center * 0.2), 250) if y_center != 0 else 250
            x_min = x_center - x_span
            x_max = x_center + x_span
            y_min = y_center - y_span
            y_max = y_center + y_span
        
        fig.update_layout(
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )

    return fig


def generate_age_mortality_figure(df_health: pd.DataFrame, selected_country_names=None):
    df = _prepare_covid_metrics(df_health)
    selected_country_names = _resolve_country_selection(df, selected_country_names, top_n=8)

    required_columns = ['location', 'continent', 'median_age', 'aged_65_older', 'total_deaths_per_million', 'gdp_per_capita']
    available_columns = [column for column in required_columns if column in df.columns]
    df_latest = prepare_map_data(df, preprocessed=True)[available_columns].copy()
    
    # For World selection, handle NaN total_deaths_per_million
    if selected_country_names == ['World']:
        world_row = df_latest[df_latest['location'] == 'World']
        if not world_row.empty and pd.isna(world_row.iloc[0].get('total_deaths_per_million')):
            # Compute World's total_deaths_per_million from non-aggregate countries
            non_agg = df_latest[~df_latest['location'].isin(['World', 'International', 'European Union'])].copy()
            if not non_agg.empty and 'total_deaths_per_million' in non_agg.columns:
                non_agg_clean = non_agg.dropna(subset=['total_deaths_per_million', 'population'] if 'population' in non_agg.columns else ['total_deaths_per_million'])
                if not non_agg_clean.empty:
                    if 'population' in non_agg_clean.columns:
                        weighted_avg = (non_agg_clean['total_deaths_per_million'] * non_agg_clean['population']).sum() / non_agg_clean['population'].sum()
                    else:
                        weighted_avg = non_agg_clean['total_deaths_per_million'].mean()
                    df_latest.loc[df_latest['location'] == 'World', 'total_deaths_per_million'] = weighted_avg
    
    df_latest = df_latest[df_latest['location'].isin(selected_country_names)].dropna(subset=['median_age', 'total_deaths_per_million'])

    gdp_series = pd.to_numeric(df_latest['gdp_per_capita'], errors='coerce') if 'gdp_per_capita' in df_latest.columns else pd.Series(dtype=float)
    gdp_fallback = gdp_series.dropna().median() if not gdp_series.dropna().empty else 50_000
    # Convert to a relative size (divide to keep numbers reasonable), fill NaNs and clip lower bound
    size_series_age = gdp_series.fillna(gdp_fallback).div(5_000) if not gdp_series.empty else pd.Series(dtype=float)
    size_series_age = size_series_age.clip(lower=0)
    q = 0.95
    cap_value_age = size_series_age.quantile(q) if not size_series_age.empty else size_series_age.max()
    capped_max_age = min(size_series_age.max(), cap_value_age) if cap_value_age > 0 else size_series_age.max()
    desired_px_age = 12
    sizeref = 2 * capped_max_age / (desired_px_age ** 2) if capped_max_age > 0 else 1
    size_values_age = size_series_age.clip(upper=cap_value_age)

    # Criar colormap por continente
    continent_colors = {
        'Africa': '#ef4444',
        'Americas': '#3b82f6',
        'Asia': '#10b981',
        'Europe': '#f59e0b',
        'Oceania': '#8b5cf6'
    }

    fig = go.Figure()
    
    # Get unique continents, handling NaN
    continents = [c for c in (df_latest['continent'].unique() if 'continent' in df_latest.columns else [None]) if pd.notna(c)]
    has_nan_continent = (df_latest['continent'].isna().any() if 'continent' in df_latest.columns else False)
    
    for continent in continents:
        df_continent = df_latest[df_latest['continent'] == continent]
        
        fig.add_trace(go.Scatter(
            x=df_continent['median_age'],
            y=df_continent['total_deaths_per_million'],
            mode='markers',
            name=continent or 'Dados',
                marker=dict(
                size=(size_values_age.loc[df_continent.index].values if 'gdp_per_capita' in df_continent.columns else 10),
                color=continent_colors.get(continent, COLOR_PALETTE['primary']) if continent else COLOR_PALETTE['primary'],
                sizemode='diameter',
                sizeref=sizeref,
                line=dict(color='white', width=1.5),
                opacity=0.75
            ),
            text=df_continent['location'],
            hovertemplate='<b>%{text}</b><br>Idade mediana: %{x:.1f}<br>Mortes: %{y:,.0f}/1M<extra></extra>',
        ))
    
    # Add trace for NaN continent (World)
    if has_nan_continent:
        df_world = df_latest[df_latest['continent'].isna()]
        fig.add_trace(go.Scatter(
            x=df_world['median_age'],
            y=df_world['total_deaths_per_million'],
            mode='markers',
            name='World',
                marker=dict(
                size=(size_values_age.loc[df_world.index].values if 'gdp_per_capita' in df_world.columns else 10),
                color=COLOR_PALETTE['primary'],
                sizemode='diameter',
                sizeref=sizeref,
                line=dict(color='white', width=1.5),
                opacity=0.75
            ),
            text=df_world['location'],
            hovertemplate='<b>%{text}</b><br>Idade mediana: %{x:.1f}<br>Mortes: %{y:,.0f}/1M<extra></extra>',
        ))

    # Compute padding based on largest marker size to ensure proper zoom (only if data exists)
    fig.update_layout(
        template='plotly_white',
        height=450,
        margin=dict(t=80, l=50, r=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=11, color=COLOR_PALETTE['neutral_dark']),
        hovermode='closest',
        uirevision='country-selection',
        xaxis=dict(
            title='Idade mediana (anos)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='Mortes acumuladas por milhão',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(200, 200, 200, 0.3)',
            borderwidth=1
        )
    )
    
    # Apply proportional zoom only if data exists
    if not df_latest.empty:
        x_range_age = df_latest['median_age'].max() - df_latest['median_age'].min()
        y_range_age = df_latest['total_deaths_per_million'].max() - df_latest['total_deaths_per_million'].min()
        
        # For single data point, provide reasonable padding instead of zero range
        if x_range_age == 0:
            x_range_age = 10  # Default range of ±5 years for single point
        if y_range_age == 0:
            y_range_age = 500  # Default range of ±250 deaths/1M for single point
        
        pad_x_age, pad_y_age = _compute_padding_for_marker_size(capped_max_age, sizeref, x_range_age, y_range_age)
        x_min_age = df_latest['median_age'].min() - pad_x_age
        x_max_age = df_latest['median_age'].max() + pad_x_age
        y_min_age = df_latest['total_deaths_per_million'].min() - pad_y_age
        y_max_age = df_latest['total_deaths_per_million'].max() + pad_y_age
        
        # For single point, ensure minimum spacing
        if df_latest.shape[0] == 1:
            x_center = df_latest['median_age'].iloc[0]
            y_center = df_latest['total_deaths_per_million'].iloc[0]
            # Create symmetric range around the single point
            x_span = max(abs(x_center * 0.2), 5) if x_center != 0 else 5
            y_span = max(abs(y_center * 0.2), 250) if y_center != 0 else 250
            x_min_age = x_center - x_span
            x_max_age = x_center + x_span
            y_min_age = y_center - y_span
            y_max_age = y_center + y_span
        
        fig.update_layout(
            xaxis=dict(range=[x_min_age, x_max_age]),
            yaxis=dict(range=[y_min_age, y_max_age])
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
        requested_columns = [
            'iso_code',
            'continent',
            'location',
            'date',
            'new_cases',
            'total_cases',
            'new_deaths',
            'total_deaths',
            'new_cases_smoothed',
            'new_deaths_smoothed',
            'new_cases_smoothed_per_million',
            'new_deaths_smoothed_per_million',
            'total_cases_per_million',
            'total_deaths_per_million',
            'population',
            'median_age',
            'aged_65_older',
            'gdp_per_capita',
            'total_vaccinations',
            'people_vaccinated',
            'people_fully_vaccinated',
            'total_vaccinations_per_hundred',
            'people_vaccinated_per_hundred',
            'people_fully_vaccinated_per_hundred',
        ]
        available_columns = [column for column in requested_columns if column in df.columns]
        df_health = df[available_columns].copy()
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
        int(mark.value // 10**6): mark.strftime('%b\n%Y')
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
            'title': 'Taxa de Letalidade (CFR) - % de óbitos entre casos confirmados',
            'color_scale': 'Reds',
            'color_label': 'CFR (%)'
        }
    elif metric == 'total_cases':
        return {
            'title': 'Casos Confirmados Cumulativos - Total de infecções registadas',
            'color_scale': 'Blues',
            'color_label': 'Total Cases'
        }
    elif metric == 'total_deaths':
        return {
            'title': 'Óbitos Cumulativos por COVID-19 - Mortes confirmadas',
            'color_scale': 'OrRd',
            'color_label': 'Mortes acumuladas'
        }
    else:
        return {
            'title': 'Mapa Mundial - Métrica Desconecida',
            'color_scale': 'Greys',
            'color_label': metric
        }

def generate_map_figure(df_map: pd.DataFrame, end_date=None, metric='total_deaths', color_max=None, selected_country_names=None):
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
    if 'cfr_pct' not in df_plot.columns and {'total_cases', 'total_deaths'}.issubset(df_plot.columns):
        df_plot['cfr_pct'] = np.where(
            df_plot['total_cases'] > 0,
            (df_plot['total_deaths'] / df_plot['total_cases']) * 100,
            np.nan,
        )

    if selected_country_names and 'World' in selected_country_names:
        selected_country_names = []

    selected_country_names = [country for country in (selected_country_names or []) if country in set(df_plot['location'].dropna())]

    if selected_country_names:
        base_df = df_plot[~df_plot['location'].isin(selected_country_names)].copy()
        selected_df = df_plot[df_plot['location'].isin(selected_country_names)].copy()
        fig = go.Figure()
        if not base_df.empty:
            fig.add_trace(
                go.Choropleth(
                    locations=base_df['iso_code'],
                    z=base_df[metric],
                    text=base_df['location'],
                    customdata=base_df[['date_display', 'total_cases', 'total_deaths', 'cfr_pct']],
                    colorscale='Greys',
                    zmin=0,
                    zmax=color_max,
                    showscale=False,
                    marker_line_color='rgba(148, 163, 184, 0.25)',
                    marker_line_width=0.2,
                    hovertemplate='<b>%{text}</b><br>Data: %{customdata[0]}<br>Casos totais: %{customdata[1]:,.0f}<br>Mortes totais: %{customdata[2]:,.0f}<br>CFR: %{customdata[3]:.2f}%<extra></extra>',
                )
            )
        fig.add_trace(
            go.Choropleth(
                locations=selected_df['iso_code'],
                z=selected_df[metric],
                text=selected_df['location'],
                customdata=selected_df[['date_display', 'total_cases', 'total_deaths', 'cfr_pct']],
                colorscale=metadata['color_scale'],
                zmin=0,
                zmax=color_max,
                showscale=True,
                colorbar=dict(
                    title=metadata['color_label'],
                    thickness=16,
                    len=0.72,
                    y=0.5,
                    x=1.02,
                    outlinewidth=0,
                ),
                marker_line_color='rgba(15, 23, 42, 0.55)',
                marker_line_width=0.8,
                hovertemplate='<b>%{text}</b><br>Data: %{customdata[0]}<br>Casos totais: %{customdata[1]:,.0f}<br>Mortes totais: %{customdata[2]:,.0f}<br>CFR: %{customdata[3]:.2f}%<extra></extra>',
            )
        )
    else:
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

    if not selected_country_names:
        fig.update_traces(hovertemplate=hover_template)

    fig.update_layout(
        template='plotly_white',
        height=620,
        margin=dict(t=60, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=10, color=COLOR_PALETTE['neutral_dark']),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgba(148, 163, 184, 0.45)',
            showland=True,
            landcolor='rgba(241, 245, 249, 0.9)',
        ),
        uirevision='country-selection'
    )

    return fig


def generate_vaccination_comparison_figure(df_health: pd.DataFrame, selected_country_names=None):
    """Cria um gráfico de comparação de vacinação (butterfly chart) entre países"""
    df = _prepare_covid_metrics(df_health)
    selected_country_names = _resolve_country_selection(df, selected_country_names, top_n=8)
    
    # Obter dados mais recentes de vacinação
    latest_data = prepare_map_data(df, preprocessed=True)
    df_comparison = latest_data[latest_data['location'].isin(selected_country_names)].copy()
    df_comparison = df_comparison[['location', 'people_fully_vaccinated_per_hundred', 'total_deaths_per_million']].dropna()
    df_comparison = df_comparison.sort_values('people_fully_vaccinated_per_hundred', ascending=True)
    
    # Criar butterfly chart
    fig = go.Figure()
    
    # Barra para vacinação
    fig.add_trace(go.Bar(
        y=df_comparison['location'],
        x=df_comparison['people_fully_vaccinated_per_hundred'],
        orientation='h',
        name='Cobertura de Vacinação (%)',
        marker_color=COLOR_PALETTE['accent'],
        hovertemplate='<b>%{y}</b><br>Vacinados: %{x:.1f}%<extra></extra>',
    ))
    
    # Barra para mortalidade (invertida visualmente)
    fig.add_trace(go.Bar(
        y=df_comparison['location'],
        x=-df_comparison['total_deaths_per_million'],
        orientation='h',
        name='Mortes/1M (invertido)',
        marker_color=COLOR_PALETTE['secondary'],
        hovertemplate='<b>%{y}</b><br>Mortes: %{x:.0f}/1M<extra></extra>',
    ))
    
    fig.update_layout(
        barmode='relative',
        template='plotly_white',
        height=400,
        margin=dict(t=80, l=120, r=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=11, color=COLOR_PALETTE['neutral_dark']),
        hovermode='y unified',
        xaxis=dict(
            title='Vacinação (%) / Mortes/1M',
            zeroline=True,
            zerolinecolor='rgba(100, 100, 100, 0.2)',
        ),
        yaxis=dict(title=''),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    
    return fig