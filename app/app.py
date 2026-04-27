import dash
import pandas as pd
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
    )

app = dash.Dash(__name__)


#* GERAÇÃO DO MAPA
df_health = load_health_covid_data(FILEPATH)
if df_health is None:
    raise FileNotFoundError(f'Não foi possível carregar os dados de {FILEPATH}')

date_marks = generate_date_slider_marks(df_health)
df_map_base = build_map_base_data(df_health)
min_date = df_health['date'].min()
max_date = df_health['date'].max()
default_date = max_date
max_total_deaths = df_health['total_deaths'].max()
max_cfr_pct = df_map_base['cfr_pct'].max()

MIN_DATE_MS = int(min_date.value // 10**6)
MAX_DATE_MS = int(max_date.value // 10**6)
DEFAULT_DATE_MS = int(default_date.value // 10**6)
DAY_IN_MS = 24 * 60 * 60 * 1000
SPEED_STEPS = [
    ('1x', 1500),
    ('2x', 750),
    ('4x', 375),
]

@app.callback(
    dash.dependencies.Output('map-graph', 'figure'),
    dash.dependencies.Output('date-indicator', 'children'),
    [
        dash.dependencies.Input('date-slider', 'value'),
        dash.dependencies.Input('metric-dropdown', 'value'),
    ]
)
def update_map(end_date_value, metric_value):
    end_date = default_date if end_date_value is None else pd.to_datetime(end_date_value, unit='ms')
    df_map = prepare_map_data(df_map_base, end_date, preprocessed=True)
    metric_value = metric_value or 'total_deaths'
    color_max = max_total_deaths if metric_value == 'total_deaths' else max_cfr_pct
    figure = generate_map_figure(df_map, end_date, metric=metric_value, color_max=color_max)
    indicator = f"Data selecionada: {pd.to_datetime(end_date).strftime('%d/%m/%Y')}"
    return figure, indicator


@app.callback(
    dash.dependencies.Output('timeline-playing', 'data'),
    dash.dependencies.Output('timeline-toggle', 'children'),
    dash.dependencies.Output('timeline-speed', 'children'),
    dash.dependencies.Output('timeline-interval', 'disabled'),
    dash.dependencies.Output('timeline-interval', 'interval'),
    dash.dependencies.Output('date-slider', 'value'),
    dash.dependencies.Output('timeline-speed-index', 'data'),
    [
        dash.dependencies.Input('timeline-toggle', 'n_clicks'),
        dash.dependencies.Input('timeline-speed', 'n_clicks'),
        dash.dependencies.Input('timeline-interval', 'n_intervals'),
    ],
    [
        dash.dependencies.State('timeline-playing', 'data'),
        dash.dependencies.State('date-slider', 'value'),
        dash.dependencies.State('timeline-speed-index', 'data'),
    ],
)
def control_timeline(toggle_clicks, speed_clicks, interval_ticks, is_playing, slider_value, speed_index):
    triggered_id = ctx.triggered_id
    is_playing = bool(is_playing)
    current_value = DEFAULT_DATE_MS if slider_value is None else int(slider_value)

    try:
        speed_index = int(speed_index)
    except (TypeError, ValueError):
        speed_index = 0

    if speed_index < 0 or speed_index >= len(SPEED_STEPS):
        speed_index = 0

    speed_label, speed_interval = SPEED_STEPS[speed_index]

    if triggered_id == 'timeline-toggle':
        is_playing = not is_playing
        if is_playing and current_value >= MAX_DATE_MS:
            current_value = MIN_DATE_MS
        return is_playing, ('Pause' if is_playing else 'Play'), f'Velocidade: {speed_label}', (not is_playing), speed_interval, current_value, speed_index

    if triggered_id == 'timeline-speed':
        next_speed_index = (speed_index + 1) % len(SPEED_STEPS)
        next_speed_label, next_speed_interval = SPEED_STEPS[next_speed_index]
        return is_playing, ('Pause' if is_playing else 'Play'), f'Velocidade: {next_speed_label}', (not is_playing), next_speed_interval, current_value, next_speed_index

    if triggered_id == 'timeline-interval' and is_playing:
        next_value = min(current_value + DAY_IN_MS, MAX_DATE_MS)
        if next_value >= MAX_DATE_MS:
            return False, 'Play', f'Velocidade: {speed_label}', True, speed_interval, MAX_DATE_MS, speed_index

        return True, 'Pause', f'Velocidade: {speed_label}', False, speed_interval, next_value, speed_index

    return is_playing, ('Pause' if is_playing else 'Play'), f'Velocidade: {speed_label}', (not is_playing), speed_interval, current_value, speed_index
app.layout = html.Div(
    children=[
        html.H1(
            children='CoVID-19 Dashboard',
            className='dashboard-title',
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Button('Play', id='timeline-toggle', n_clicks=0, className='timeline-button'),
                                html.Button('Velocidade: 1x', id='timeline-speed', n_clicks=0, className='timeline-button'),
                            ],
                            className='timeline-buttons',
                        ),
                        dcc.Dropdown(
                            id='metric-dropdown',
                            options=[
                                {'label': 'Mortes acumuladas', 'value': 'total_deaths'},
                                {'label': 'CFR (%)', 'value': 'cfr_pct'},
                            ],
                            value='total_deaths',
                            clearable=False,
                        ),
                        html.Div(
                            id='date-indicator',
                            children=f"Data selecionada: {pd.to_datetime(DEFAULT_DATE_MS, unit='ms').strftime('%d/%m/%Y')}",
                            className='date-indicator',
                        ),
                    ],
                    className='timeline-topbar',
                ),
                dcc.Slider(
                    id='date-slider',
                    min=MIN_DATE_MS,
                    max=MAX_DATE_MS,
                    value=DEFAULT_DATE_MS,
                    step=DAY_IN_MS,
                    marks=date_marks,
                    updatemode='mouseup',
                    # 1. Isso desativa a caixinha (tooltip) com o número gigante ao arrastar
                    tooltip={"always_visible": False, "placement": "bottom"}, 
                ),
                dcc.Interval(
                    id='timeline-interval',
                    interval=1500,
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Store(id='timeline-playing', data=False),
                dcc.Store(id='timeline-speed-index', data=0),
            ],
            className='dashboard-card',
        ),
        dcc.Graph(
            id='map-graph',
            figure=generate_map_figure(
                prepare_map_data(df_map_base, default_date, preprocessed=True),
                default_date,
                metric='total_deaths',
                color_max=max_total_deaths,
            ),
            className='dashboard-graph',
            style={'width': '100%', 'height': '760px'},
            config={'displayModeBar': False, 'responsive': True},
        ),
        html.Div(
            children=[
                html.Div('Evolução do PIB por país', className='gdp-alluvial-title'),
                dcc.Graph(
                    id='gdp-alluvial-graph',
                    figure=generate_gdp_alluvial_figure(allowed_country_names=df_map_base['location'].unique()),
                    className='gdp-alluvial-graph',
                    config={'displayModeBar': False, 'responsive': True},
                ),
            ],
            className='gdp-alluvial-panel',
        ),
    ],
    className='dashboard-shell',
)

app.layout = html.Div(app.layout.children, className='dashboard-page')
     

if __name__ == '__main__':
    app.run(debug=True)