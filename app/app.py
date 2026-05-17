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
        generate_date_slider_marks,
        generate_map_figure,
        generate_gdp_alluvial_figure,
        generate_covid_evolution_figure,
        generate_gdp_trend_figure,
        generate_gdp_mortality_scatter_figure,
        generate_age_mortality_figure,
    )
except ImportError:
    from data_manipulation import (
        load_health_covid_data,
        FILEPATH,
        build_map_base_data,
        prepare_map_data,
        generate_date_slider_marks,
        generate_map_figure,
        generate_gdp_alluvial_figure,
        generate_covid_evolution_figure,
        generate_gdp_trend_figure,
        generate_gdp_mortality_scatter_figure,
        generate_age_mortality_figure,
    )

app = dash.Dash(__name__)


#* GERAÇÃO DO MAPA
df_health = load_health_covid_data(FILEPATH)
if df_health is None:
    raise FileNotFoundError(f'Não foi possível carregar os dados de {FILEPATH}')
assert df_health is not None
df_health_data: pd.DataFrame = df_health

date_marks = generate_date_slider_marks(df_health, num_marks=3)
df_map_base = build_map_base_data(df_health)
min_date = df_health['date'].min()
max_date = df_health['date'].max()
default_date = max_date
max_total_deaths = df_health['total_deaths'].max()
max_cfr_pct = df_map_base['cfr_pct'].max()
country_values = sorted(df_map_base['location'].dropna().unique().tolist())
country_options = [
    {'label': location, 'value': location}
    for location in country_values
]
default_countries = (
    prepare_map_data(df_map_base, preprocessed=True)
    .sort_values('total_deaths', ascending=False)['location']
    .dropna()
    .drop_duplicates()
    .head(6)
    .tolist()
)

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
    normalized = [str(country) for country in selected_countries if pd.notna(country) and str(country) in country_values]
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
    if clicked_country in country_values:
        return [clicked_country]

    return current_selection


@app.callback(
    dash.dependencies.Output('country-selector', 'value'),
    [
        dash.dependencies.Input('country-selection-store', 'data'),
    ],
)
def mirror_country_selection_to_dropdown(selected_countries):
    return _normalize_selected_countries(selected_countries)


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
        dash.dependencies.Input('metric-dropdown', 'value'),
    ],
)
def update_dashboard_views(selected_countries, end_date_value, metric_value):
    selected_countries = _normalize_selected_countries(selected_countries)
    end_date = default_date if end_date_value is None else pd.to_datetime(end_date_value, unit='ms')
    df_map = prepare_map_data(df_map_base, end_date, preprocessed=True)
    metric_value = metric_value or 'total_deaths'
    color_max = max_total_deaths if metric_value == 'total_deaths' else max_cfr_pct

    latest_data = prepare_map_data(df_health_data, preprocessed=True)
    selected_latest = latest_data[latest_data['location'].isin(selected_countries)].copy()
    selected_count = len(selected_countries)
    total_deaths_million = selected_latest['total_deaths_per_million'].mean()
    # Calcular vacinação completa se a coluna existe, senão retorna NaN
    if 'people_fully_vaccinated_per_hundred' in selected_latest.columns:
        vaccination_rate = selected_latest['people_fully_vaccinated_per_hundred'].mean()
    else:
        vaccination_rate = np.nan

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
            html.Span('Seleção ativa: ', className='selection-summary-label'),
            html.Span(', '.join(selected_countries[:4]), className='selection-summary-countries'),
            html.Span('' if len(selected_countries) <= 4 else f' +{len(selected_countries) - 4}', className='selection-summary-more'),
        ]
    )

    return (
        generate_map_figure(df_map, end_date, metric=metric_value, color_max=color_max, selected_country_names=selected_countries),
        generate_gdp_alluvial_figure(allowed_country_names=selected_countries),
        generate_covid_evolution_figure(df_health_data, selected_countries),
        generate_gdp_trend_figure(selected_countries),
        generate_gdp_mortality_scatter_figure(df_health_data, selected_countries),
        generate_age_mortality_figure(df_health_data, selected_countries),
        summary_text,
        f'{selected_count} países',
        f'{total_deaths_text} mortes/milhão',
        f'{vaccination_text} totalmente vacinados',
    )


app.layout = html.Div(
    children=[
        dcc.Store(id='country-selection-store', data=default_countries),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div('CoVID-19 Dashboard', className='dashboard-title'),
                        html.Div(
                            'Veja como a pandemia de CoVID-19 evoluiu, como afetou o PIB e aspetos demográficos a nível mundial.',
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
                                html.Div('País e Data', className='control-card-title'),
                                dcc.Dropdown(
                                    id='metric-dropdown',
                                    options=[
                                        {'label': 'Mortes acumuladas', 'value': 'total_deaths'},
                                        {'label': 'Case Fatality Ratio (%)', 'value': 'cfr_pct'},
                                    ],
                                    value='total_deaths',
                                    clearable=False,
                                    className='metric-dropdown',
                                ),
                                dcc.Dropdown(
                                    id='country-selector',
                                    options=country_options,
                                    value=default_countries,
                                    multi=True,
                                    clearable=False,
                                    placeholder='Selecione um ou mais países',
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
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div('Mapa Global', className='panel-title'),
                                html.Div(id='selection-summary', className='selection-summary'),
                            ],
                            className='panel-header',
                        ),
                        dcc.Graph(
                            id='map-graph',
                            figure=generate_map_figure(
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
                        html.Div('Contexto e indicadores', className='panel-title'),
                        html.Div('Seleção ativa e resumo da situação.', className='panel-caption'),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div('Países selecionados', className='kpi-title'),
                                        html.Div(id='selected-count-kpi', className='kpi-value'),
                                    ],
                                    className='kpi-card',
                                ),
                                html.Div(
                                    children=[
                                        html.Div('Mortes médias por milhão', className='kpi-title'),
                                        html.Div(id='mortality-kpi', className='kpi-value'),
                                    ],
                                    className='kpi-card',
                                ),
                                html.Div(
                                    children=[
                                        html.Div('Vacinação completa média', className='kpi-title'),
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
                        html.Div('Evolução do PIB por país', className='panel-title'),
                        dcc.Graph(
                            id='gdp-alluvial-graph',
                            figure=generate_gdp_alluvial_figure(allowed_country_names=default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('Evolução pandémica', className='panel-title'),
                        dcc.Graph(
                            id='covid-evolution-graph',
                            figure=generate_covid_evolution_figure(df_health_data, default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('PIB e resiliência económica', className='panel-title'),
                        dcc.Graph(
                            id='gdp-trend-graph',
                            figure=generate_gdp_trend_figure(default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('PIB vs mortalidade', className='panel-title'),
                        dcc.Graph(
                            id='gdp-mortality-scatter-graph',
                            figure=generate_gdp_mortality_scatter_figure(df_health_data, default_countries),
                            className='report-graph',
                            config={'displayModeBar': False, 'responsive': True},
                        ),
                    ],
                    className='chart-card',
                ),
                html.Div(
                    children=[
                        html.Div('Envelhecimento e mortalidade', className='panel-title'),
                        dcc.Graph(
                            id='age-mortality-graph',
                            figure=generate_age_mortality_figure(df_health_data, default_countries),
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
