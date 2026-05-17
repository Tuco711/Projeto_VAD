import pandas as pd
import plotly.io as pio
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global settings
pio.templates.default = "plotly_white"


FILEPATH = Path('.\\datasets\\owid-covid-data.csv')
GDP_FILEPATH = Path('.\\datasets\\GDP_data.csv')


def _apply_style_template(fig, title_text=None, subtitle_text=None, height=450):
    """Minimal layout helper: keeps sizes but removes visual styling.
    Visual presentation is handled in assets/style.css.
    """
    fig.update_layout(
        height=height,
        margin=dict(t=80 if subtitle_text else 70, l=30, r=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#0f172a'),
    )

    if title_text or subtitle_text:
        annotations = []
        y_pos = 1.08
        if title_text:
            annotations.append(dict(text=f'<b>{title_text}</b>', xref='paper', yref='paper', x=0, y=y_pos, showarrow=False, xanchor='left'))
            y_pos -= 0.06
        if subtitle_text:
            annotations.append(dict(text=subtitle_text, xref='paper', yref='paper', x=0, y=y_pos, showarrow=False, xanchor='left'))
        fig.update_layout(annotations=annotations)

    return fig


def _extract_gdp_year_columns(df_gdp: pd.DataFrame) -> dict:
    year_columns = {}
    for column in df_gdp.columns:
        if len(column) >= 4 and column[:4].isdigit():
            year_columns[int(column[:4])] = column
    return dict(sorted(year_columns.items()))


def gdp_alluvial_figure(filepath: Path = GDP_FILEPATH, top_n: int = 12, allowed_country_names=None):
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
        height=layout_height,
        margin=dict(t=margin_top, l=8, r=8, b=margin_bottom),
        annotations=[
            dict(x=year_positions[first_year], y=year_label_y, xref='paper', yref='paper', text=str(first_year), showarrow=False),
            dict(x=year_positions[middle_year], y=year_label_y, xref='paper', yref='paper', text=str(middle_year), showarrow=False),
            dict(x=year_positions[latest_year], y=year_label_y, xref='paper', yref='paper', text=str(latest_year), showarrow=False),
        ],
    )

    fig = _apply_style_template(fig)

    return fig


def _resolve_country_selection(df_health: pd.DataFrame, selected_country_names=None, top_n: int = 6) -> list[str]:
    available_countries = [country for country in df_health['location'].dropna().drop_duplicates().tolist()]
    if selected_country_names:
        selected = [country for country in selected_country_names if country in available_countries]
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


def covid_evolution_figure(df_health: pd.DataFrame, selected_country_names=None):
    df = _prepare_covid_metrics(df_health)
    selected_country_names = _resolve_country_selection(df, selected_country_names, top_n=6)

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
        ('new_cases_smoothed_per_million', 'Incidence of Cases', 'Number of new confirmed cases per million inhabitants, smoothed over 7 days', 'cases'),
        ('new_deaths_smoothed_per_million', 'Attributable Mortality', 'Number of deaths from COVID-19 per million inhabitants, smoothed over 7 days', 'deaths'),
        ('people_fully_vaccinated_per_hundred', 'Vaccination Coverage', 'Percentage of the population that has completed the primary vaccination scheme', 'vaccines'),
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
        height=720,
        margin=dict(t=120, l=30, r=20, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='left', x=0),
    )
    
    # Atualizar eixos Y com melhor formatação
    fig.update_yaxes(title_text='Cases/1M', row=1, col=1, tickformat=',d')
    fig.update_yaxes(title_text='Deaths/1M', row=2, col=1, tickformat=',d')
    fig.update_yaxes(title_text='Coverage %', row=3, col=1, tickformat='.1%')
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

    fig = _apply_style_template(fig)

    return fig


def gdp_trend_figure(selected_country_names=None, filepath: Path = GDP_FILEPATH):
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
        height=450,
        margin=dict(t=80, l=50, r=20, b=30),
        xaxis=dict(title='Year', showgrid=True),
        yaxis=dict(title='GDP (USD current)', tickformat=',.0f', showgrid=True),
    )

    return fig


def gdp_mortality_scatter_figure(df_health: pd.DataFrame, selected_country_names=None, filepath: Path = GDP_FILEPATH):
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

    df_merged = df_latest.merge(df_gdp, left_on='iso_code', right_on='Country Code', how='inner')
    df_merged = df_merged[df_merged['location'].isin(selected_country_names)].copy()
    df_merged = df_merged.dropna(subset=['gdp_growth_pct', 'total_deaths_per_million'])

    # Criar colormap por continente
    continents = df_merged['continent'].unique() if 'continent' in df_merged.columns else []
    continent_colors = {
        'Africa': '#ef4444',
        'Americas': '#3b82f6',
        'Asia': '#10b981',
        'Europe': '#f59e0b',
        'Oceania': '#8b5cf6'
    }
    df_merged['color'] = df_merged['continent'].map(continent_colors) if 'continent' in df_merged.columns else '#2563eb'

    fig = go.Figure()

    # Robust marker sizing: map population to a bounded visual range
    if 'population' in df_merged.columns and not df_merged['population'].isna().all():
        pop_max = df_merged['population'].max()
        pop_min = df_merged['population'].min()
        pop_max = pop_max if pop_max > 0 else 1
    else:
        pop_max = 1

    for continent in (df_merged['continent'].unique() if 'continent' in df_merged.columns else [None]):
        df_continent = df_merged[df_merged['continent'] == continent] if continent else df_merged

        # scale sizes between 8 and 48 for readability
        if 'population' in df_continent.columns and not df_continent['population'].isna().all():
            sizes = ((df_continent['population'] / pop_max) * 40).fillna(0) + 8
        else:
            sizes = 10

        fig.add_trace(go.Scatter(
            x=df_continent['gdp_growth_pct'],
            y=df_continent['total_deaths_per_million'],
            mode='markers',
            name=continent or 'Dados',
            marker=dict(
                size=sizes,
                color='#2563eb',
                line=dict(color='white', width=1.25),
                opacity=0.85
            ),
            text=df_continent['location'],
            hovertemplate='<b>%{text}</b><br>Crescimento do PIB: %{x:.1f}%<br>Mortes: %{y:,.0f} por milhão<extra></extra>',
        ))

    # Add a linear trend line to help interpret the relationship
    mask = df_merged['gdp_growth_pct'].notna() & df_merged['total_deaths_per_million'].notna()
    if mask.sum() >= 2:
        x_vals = pd.to_numeric(df_merged.loc[mask, 'gdp_growth_pct'], errors='coerce').to_numpy(dtype=float)
        y_vals = pd.to_numeric(df_merged.loc[mask, 'total_deaths_per_million'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]
        try:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Tendência (linear)', line=dict(dash='dash', color='#111827'), hoverinfo='skip'))
        except Exception:
            pass

    # Add a small legend/annotation explaining point size
    fig.update_layout(
        title='GDP vs Mortality — GDP Growth (2019–2024) vs Deaths per million',
        height=480,
        margin=dict(t=100, l=60, r=20, b=40),
        xaxis=dict(title='GDP Growth 2019–2024 (%)', zeroline=True, showgrid=True),
        yaxis=dict(title='Deaths per million (cumulative)', showgrid=True),
        legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='right', x=0.99),
        annotations=[dict(x=1, y=-0.12, xref='paper', yref='paper', showarrow=False, text='Dot size ~ country population (larger dot = larger population)')],
    )

    fig = _apply_style_template(fig)

    return fig


def age_mortality_figure(df_health: pd.DataFrame, selected_country_names=None):
    df = _prepare_covid_metrics(df_health)
    selected_country_names = _resolve_country_selection(df, selected_country_names, top_n=8)

    required_columns = ['location', 'continent', 'median_age', 'aged_65_older', 'total_deaths_per_million', 'gdp_per_capita']
    available_columns = [column for column in required_columns if column in df.columns]
    df_latest = prepare_map_data(df, preprocessed=True)[available_columns].copy()
    df_latest = df_latest[df_latest['location'].isin(selected_country_names)].dropna(subset=['median_age', 'total_deaths_per_million'])

    # Criar colormap por continente
    continent_colors = {
        'Africa': '#ef4444',
        'Americas': '#3b82f6',
        'Asia': '#10b981',
        'Europe': '#f59e0b',
        'Oceania': '#8b5cf6'
    }

    fig = go.Figure()
    
    for continent in (df_latest['continent'].unique() if 'continent' in df_latest.columns else [None]):
        df_continent = df_latest[df_latest['continent'] == continent] if continent else df_latest
        
        fig.add_trace(go.Scatter(
            x=df_continent['median_age'],
            y=df_continent['total_deaths_per_million'],
            mode='markers',
            name=continent or 'Dados',
            marker=dict(
                size=df_continent['gdp_per_capita'] / 5000 if 'gdp_per_capita' in df_continent.columns else 10,
                color=continent_colors.get(continent, '#2563eb') if continent else '#2563eb',
                sizemode='diameter',
                sizeref=2 * (max(df_latest['gdp_per_capita'] / 5000) if 'gdp_per_capita' in df_latest.columns else 10) / (44 ** 2),
                line=dict(color='white', width=1.5),
                opacity=0.75
            ),
            text=df_continent['location'],
            hovertemplate='<b>%{text}</b><br>Idade mediana: %{x:.1f}<br>Mortes: %{y:,.0f}/1M<extra></extra>',
        ))

    fig.update_layout(
        height=450,
        margin=dict(t=80, l=50, r=20, b=30),
        xaxis=dict(title='Median Age (years)', showgrid=True),
        yaxis=dict(title='Cumulative Deaths per Million', showgrid=True),
        legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='right', x=0.99),
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

def date_slider_marks(df_health: pd.DataFrame, num_marks: int = 6) -> dict:
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

def map_metadata(metric: str = 'cfr_pct') -> dict:
    if metric == 'cfr_pct':
        return {
            'title': 'Case Fatality Rate (CFR) - % of deaths among confirmed cases',
            'color_scale': 'Reds',
            'color_label': 'CFR (%)'
        }
    elif metric == 'total_cases':
        return {
            'title': 'Cumulative Confirmed Cases - Total of infections recorded',
            'color_scale': 'Blues',
            'color_label': 'Total Cases'
        }
    elif metric == 'total_deaths':
        return {
            'title': 'Cumulative Deaths from CoVID-19 - Confirmed deaths',
            'color_scale': 'OrRd',
            'color_label': 'Cumulative Deaths'
        }
    else:
        return {
            'title': 'World Map - Unknown Metric',
            'color_scale': 'Greys',
            'color_label': metric
        }

def map_figure(df_map: pd.DataFrame, end_date=None, metric='total_deaths', color_max=None, selected_country_names=None):
    metric = metric or 'total_deaths'
    metadata = map_metadata(metric)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        if metric == 'cfr_pct':
            title = f"World Map - CFR (%) until {end_date.strftime('%d/%m/%Y')}"
        elif metric == 'total_cases':
            title = f"World Map - Total Cases until {end_date.strftime('%d/%m/%Y')}"
        else:
            title = f"World Map - Cumulative Deaths until {end_date.strftime('%d/%m/%Y')}"
    else:
        if metric == 'cfr_pct':
            title = 'World Map - CFR (%)'
        elif metric == 'total_cases':
            title = 'World Map - Total Cases'
        else:
            title = 'World Map - Cumulative Deaths'

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
                    hovertemplate='<b>%{text}</b><br>Data: %{customdata[0]}<br>Total Cases: %{customdata[1]:,.0f}<br>Total Deaths: %{customdata[2]:,.0f}<br>CFR: %{customdata[3]:.2f}%<extra></extra>',
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
                hovertemplate='<b>%{text}</b><br>Data: %{customdata[0]}<br>Total Cases: %{customdata[1]:,.0f}<br>Total Deaths: %{customdata[2]:,.0f}<br>CFR: %{customdata[3]:.2f}%<extra></extra>',
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
            'Date: %{customdata[0]}<br>'
            'Total Cases: %{customdata[1]:,.0f}<br>'
            'Total Deaths: %{customdata[2]:,.0f}<br>'
            'CFR: %{customdata[3]:.2f}%<extra></extra>'
        )
    else:
        hover_template = (
            '<b>%{hovertext}</b><br>'
            'Date: %{customdata[0]}<br>'
            'Total Cases: %{customdata[1]:,.0f}<br>'
            'Total Deaths: %{customdata[2]:,.0f}<br>'
            'CFR: %{customdata[3]:.2f}%<extra></extra>'
        )

    if not selected_country_names:
        fig.update_traces(hovertemplate=hover_template)

    fig.update_layout(
        height=540,
        margin=dict(t=20, l=10, r=10, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgba(148, 163, 184, 0.45)',
            showland=True,
            landcolor='rgba(241, 245, 249, 0.9)',
        ),
    )

    return fig


def vaccination_comparison_figure(df_health: pd.DataFrame, selected_country_names=None):
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
        name='Vaccination Coverage (%)',
        marker_color='steelblue',
        hovertemplate='<b>%{y}</b><br>Vaccinated: %{x:.1f}%<extra></extra>',
    ))
    
    # Barra para mortalidade (invertida visualmente)
    fig.add_trace(go.Bar(
        y=df_comparison['location'],
        x=-df_comparison['total_deaths_per_million'],
        orientation='h',
        name='Deaths/1M (inverted)',
        marker_color='indianred',
        hovertemplate='<b>%{y}</b><br>Deaths: %{x:.0f}/1M<extra></extra>',
    ))
    
    fig.update_layout(
        barmode='relative',
        height=400,
        margin=dict(t=80, l=120, r=20, b=20),
        xaxis=dict(title='Vaccination Coverage (%) / Deaths/1M', zeroline=True),
        yaxis=dict(title=''),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    
    fig = _apply_style_template(fig)

    return fig
