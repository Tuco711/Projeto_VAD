import dash
import pandas as pd
import numpy as np
from dash import html, dcc, ctx
try:
    from .data_manipulation import (
        load_health_covid_data,
        FILEPATH,
        build_map_base_data,
        prepare_map_data,
        date_slider_marks,
        map_figure,
        gdp_alluvial_figure,
        covid_evolution_figure,
        gdp_trend_figure,
        gdp_mortality_scatter_figure,
        age_mortality_figure,
    )
except ImportError:
    from data_manipulation import (
        load_health_covid_data,
        FILEPATH,
        build_map_base_data,
        prepare_map_data,
        date_slider_marks,
        map_figure,
        gdp_alluvial_figure,
        covid_evolution_figure,
        gdp_trend_figure,
        gdp_mortality_scatter_figure,
        age_mortality_figure,
    )

app = dash.Dash(__name__)


#* GERAÇÃO DO MAPA
df_health = load_health_covid_data(FILEPATH)
if df_health is None:
    raise FileNotFoundError(f'Não foi possível carregar os dados de {FILEPATH}')
assert df_health is not None
df_health_data: pd.DataFrame = df_health

date_marks = date_slider_marks(df_health, num_marks=3)
df_map_base = build_map_base_data(df_health)
min_date = df_health['date'].min()
max_date = df_health['date'].max()
default_date = max_date
max_total_deaths = df_health['total_deaths'].max()
max_cfr_pct = df_map_base['cfr_pct'].max()
country_values = sorted(df_map_base['location'].dropna().unique().tolist())
selection_values = set(country_values) | {'World'}
country_options = [{'label': 'World', 'value': 'World'}] + [
    {'label': location, 'value': location}
    for location in country_values
]
default_countries = ['World']

MIN_DATE_MS = int(min_date.value // 10**6)
MAX_DATE_MS = int(max_date.value // 10**6)
DEFAULT_DATE_MS = int(default_date.value // 10**6)
DAY_IN_MS = 24 * 60 * 60 * 1000
SPEED_STEPS = [
    ('1x', 1500),
    ('2x', 750),
    ('4x', 375),
]


def _normalize_selected_countries(selected_countries):
    # Coerce single string into a list
    if isinstance(selected_countries, str):
        selected_countries = [selected_countries]

    # If it's not an iterable of country names, fall back to defaults
    if not selected_countries or not hasattr(selected_countries, '__iter__'):
        return default_countries

    # Normalize items to str and filter by available values
    normalized = [str(country) for country in selected_countries if pd.notna(country) and str(country) in selection_values]
    if 'World' in normalized:
        return ['World']
    return normalized or default_countries


def _extract_country_from_click(click_data):
    if not click_data:
        return None

    points = click_data.get('points') or []
    if not points:
        return None

    point = points[0]
    for key in ('hovertext', 'text', 'location'):
        value = point.get(key)
        if value:
            return value

    customdata = point.get('customdata')
    if isinstance(customdata, (list, tuple)):
        for value in customdata:
            if isinstance(value, str) and value:
                return value

    return None


def _mean_latest_valid_metric(df: pd.DataFrame, selected_countries, metric: str):
    if metric not in df.columns:
        return np.nan

    df_metric = df[[col for col in ['location', 'date', metric] if col in df.columns]].copy()
    if 'location' not in df_metric.columns or 'date' not in df_metric.columns:
        return np.nan

    df_metric = df_metric[df_metric['location'].isin(selected_countries)].copy()
    if df_metric.empty:
        return np.nan

    df_metric['date'] = pd.to_datetime(df_metric['date'])
    df_metric = df_metric.sort_values(['location', 'date'])

    latest_valid_values = []
    for _, country_df in df_metric.groupby('location', sort=False):
        valid_values = country_df[metric].dropna()
        if not valid_values.empty:
            latest_valid_values.append(valid_values.iloc[-1])

    if not latest_valid_values:
        return np.nan

    return float(np.mean(latest_valid_values))

@app.callback(
    dash.dependencies.Output('country-selection-store', 'data'),
    [
        dash.dependencies.Input('country-selector', 'value'),
        dash.dependencies.Input('map-graph', 'clickData'),
        dash.dependencies.Input('gdp-alluvial-graph', 'clickData'),
        dash.dependencies.Input('covid-evolution-graph', 'clickData'),
        dash.dependencies.Input('gdp-trend-graph', 'clickData'),
        dash.dependencies.Input('gdp-mortality-scatter-graph', 'clickData'),
        dash.dependencies.Input('age-mortality-graph', 'clickData'),
    ],
    [
        dash.dependencies.State('country-selection-store', 'data'),
    ],
)
def sync_country_selection(dropdown_value, map_click, alluvial_click, covid_click, gdp_click, scatter_click, age_click, current_selection):
    triggered_id = ctx.triggered_id
    current_selection = _normalize_selected_countries(current_selection)

    if triggered_id == 'country-selector':
        return _normalize_selected_countries(dropdown_value)

    clicked_country = _extract_country_from_click(
        map_click or alluvial_click or covid_click or gdp_click or scatter_click or age_click
    )
    if clicked_country in selection_values:
        return [clicked_country]

    return current_selection





@app.callback(
    dash.dependencies.Output('map-graph', 'figure'),
    dash.dependencies.Output('gdp-alluvial-graph', 'figure'),
    dash.dependencies.Output('covid-evolution-graph', 'figure'),
    dash.dependencies.Output('gdp-trend-graph', 'figure'),
    dash.dependencies.Output('gdp-mortality-scatter-graph', 'figure'),
    dash.dependencies.Output('age-mortality-graph', 'figure'),
    dash.dependencies.Output('selection-summary', 'children'),
    dash.dependencies.Output('selected-count-kpi', 'children'),
    dash.dependencies.Output('mortality-kpi', 'children'),
    dash.dependencies.Output('vaccination-kpi', 'children'),
    [
        dash.dependencies.Input('country-selection-store', 'data'),
        dash.dependencies.Input('date-slider', 'value'),
    ],
)
def update_dashboard_views(selected_countries, end_date_value):
    selected_countries = _normalize_selected_countries(selected_countries)
    end_date = default_date if end_date_value is None else pd.to_datetime(end_date_value, unit='ms')
    df_map = prepare_map_data(df_map_base, end_date, preprocessed=True)
    # Map metric is fixed to Total Deaths only
    metric_value = 'total_deaths'
    color_max = max_total_deaths

    map_selected_countries = [] if selected_countries == ['World'] else selected_countries
    world_only_selection = ['World'] if selected_countries == ['World'] else selected_countries

    latest_data = prepare_map_data(df_health_data, preprocessed=True)
    selected_latest = latest_data[latest_data['location'].isin(selected_countries)].copy()
    selected_count = len(selected_countries)

    # Compute deaths per million. If World selected, prefer explicit World row,
    # otherwise compute population-weighted aggregate from non-aggregate countries.
    total_deaths_million = np.nan
    if selected_countries == ['World']:
        # Prefer World row if present
        world_row = latest_data[latest_data['location'] == 'World'] if 'location' in latest_data.columns else latest_data.iloc[0:0]
        if not world_row.empty and 'total_deaths_per_million' in world_row.columns and pd.notna(world_row.iloc[0].get('total_deaths_per_million')):
            total_deaths_million = float(world_row.iloc[0]['total_deaths_per_million'])
        else:
            non_agg = latest_data[~latest_data['location'].isin(['World', 'International', 'European Union'])].copy()
            # If raw total_deaths available, compute (sum deaths / sum pop) * 1e6
            if 'total_deaths' in non_agg.columns and 'population' in non_agg.columns and not non_agg.empty:
                total_deaths_sum = non_agg['total_deaths'].sum(min_count=1)
                pop_sum = non_agg['population'].sum(min_count=1)
                if pop_sum and not pd.isna(total_deaths_sum):
                    total_deaths_million = float((total_deaths_sum / pop_sum) * 1_000_000)
            # Fallback: if only per-million available, compute population-weighted average
            elif 'total_deaths_per_million' in non_agg.columns and 'population' in non_agg.columns and not non_agg.empty:
                weighted = (non_agg['total_deaths_per_million'] * non_agg['population']).sum(min_count=1)
                pop_sum = non_agg['population'].sum(min_count=1)
                if pop_sum and not pd.isna(weighted):
                    total_deaths_million = float(weighted / pop_sum)
    else:
        # For non-world selections, use simple mean of per-country per-million values
        if 'total_deaths_per_million' in selected_latest.columns and not selected_latest.empty:
            total_deaths_million = selected_latest['total_deaths_per_million'].mean()

    vaccination_rate = _mean_latest_valid_metric(
        df_health_data,
        selected_countries,
        'people_fully_vaccinated_per_hundred',
    )

    if pd.isna(total_deaths_million):
        total_deaths_text = 'n/d'
    else:
        total_deaths_text = f'{total_deaths_million:,.1f}'

    if pd.isna(vaccination_rate):
        vaccination_text = 'n/d'
    else:
        vaccination_text = f'{vaccination_rate:,.1f}%'

    summary_text = html.Div(
        children=[
            html.Span('Active Selection: ', className='selection-summary-label'),
            html.Span(', '.join(selected_countries[:4]), className='selection-summary-countries'),
            html.Span('' if len(selected_countries) <= 4 else f' +{len(selected_countries) - 4}', className='selection-summary-more'),
        ]
    )

    selected_count_text = 'World' if selected_countries == ['World'] else f'{selected_count} countries'

    return (
        map_figure(df_map, end_date, metric=metric_value, color_max=color_max, selected_country_names=map_selected_countries),
        gdp_alluvial_figure(allowed_country_names=world_only_selection),
        covid_evolution_figure(df_health_data, world_only_selection),
        gdp_trend_figure(world_only_selection),
        gdp_mortality_scatter_figure(df_health_data, world_only_selection),
        age_mortality_figure(df_health_data, world_only_selection),
        summary_text,
        f'{selected_count} countries',
        f'{total_deaths_text} deaths/million',
        f'{vaccination_text} fully vaccinated',
    )


app.layout = html.Div(
    children=[
        dcc.Store(id='country-selection-store', data=default_countries),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div('The Pandemic Lens', className='dashboard-title'),
                        html.Div(
                            'Health, economy and demographic trends during the CoVID-19 pandemic',
                            className='dashboard-subtitle',
                        ),
                    ],
                    className='dashboard-hero-copy',
                ),
            ],
            className='dashboard-hero',
        ),
        # Left collapsible sidebar (appears on hover)
        html.Div(
            id='left-sidebar',
            children=[
                html.Div(
                    className='left-sidebar-inner',
                    children=[
                        html.Button(
                            children=[
                                html.Span(className='hamburger-line'),
                                html.Span(className='hamburger-line'),
                                html.Span(className='hamburger-line'),
                            ],
                            id='sidebar-toggle',
                            n_clicks=0,
                            className='left-sidebar-icon',
                            title='Abrir filtros',
                        ),
                        html.Div(
                            className='left-panel sidebar-contents',
                            children=[
                                html.Div('Filters', className='control-card-title'),
                                dcc.Dropdown(
                                    id='country-selector',
                                    options=country_options,
                                    value=default_countries,
                                    multi=True,
                                    clearable=False,
                                    placeholder='Select one or more countries',
                                    className='country-selector',
                                ),
                                dcc.Slider(
                                    id='date-slider',
                                    min=MIN_DATE_MS,
                                    max=MAX_DATE_MS,
                                    value=DEFAULT_DATE_MS,
                                    step=DAY_IN_MS,
                                    marks=date_marks,
                                    updatemode='mouseup',
                                    allow_direct_input=False, 
                                    tooltip={
                                        "always_visible": False,
                                        "placement": "bottom",
                                        "transform": "formatDate",
                                        "template": "{value}",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Main content area with map and charts
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div('Tracking human fatalities', className='panel-title'),
                                html.Div(id='selection-summary', className='selection-summary'),
                            ],
                            className='panel-header',
                        ),
                        dcc.Graph(
                            id='map-graph',
                            figure=map_figure(
                                prepare_map_data(df_map_base, default_date, preprocessed=True),
                                default_date,
                                metric='total_deaths',
                                color_max=max_total_deaths,
                                selected_country_names=default_countries,
                            ),
                            className='dashboard-graph dashboard-graph--map',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='dashboard-card dashboard-card--main',
                ),
            ],
            className='dashboard-stage',
        ),

        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div('Context and Indicators', className='panel-title'),
                        html.Div('Summary of indicators for the selected countries.', className='panel-caption'),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div('Selected Countries', className='kpi-title'),
                                        html.Div(id='selected-count-kpi', className='kpi-value'),
                                    ],
                                    className='kpi-card',
                                ),
                                html.Div(
                                    children=[
                                        html.Div('Average Deaths per Million', className='kpi-title'),
                                        html.Div(id='mortality-kpi', className='kpi-value'),
                                    ],
                                    className='kpi-card',
                                ),
                                html.Div(
                                    children=[
                                        html.Div('Average Complete Vaccination Rate', className='kpi-title'),
                                        html.Div(id='vaccination-kpi', className='kpi-value'),
                                    ],
                                    className='kpi-card',
                                ),
                            ],
                            className='kpi-blocks',
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('Economic Disruption during the Pandemic', className='panel-title'),
                        dcc.Graph(
                            id='gdp-alluvial-graph',
                            figure=gdp_alluvial_figure(allowed_country_names=default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('The Progression of the Pandemic', className='panel-title'),
                        dcc.Graph(
                            id='covid-evolution-graph',
                            figure=covid_evolution_figure(df_health_data, default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('Economic Performance under Pandemic Pressure', className='panel-title'),
                        dcc.Graph(
                            id='gdp-trend-graph',
                            figure=gdp_trend_figure(default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('Did stronger economies experience lower mortality?', className='panel-title'),
                        dcc.Graph(
                            id='gdp-mortality-scatter-graph',
                            figure=gdp_mortality_scatter_figure(df_health_data, default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('How ageing influenced CoVID-19 mortality?', className='panel-title'),
                        dcc.Graph(
                            id='age-mortality-graph',
                            figure=age_mortality_figure(df_health_data, default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
            ],
            className='dashboard-grid',
        ),
    ],
    className='dashboard-shell',
)

app.layout = html.Div(app.layout.children, className='dashboard-page')     

if __name__ == '__main__':
    app.run(debug=True)
