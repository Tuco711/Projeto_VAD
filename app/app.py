import dash
from dash import html, dcc
try:
    from .data_manipulation import (
        load_health_covid_data,
        FILEPATH,
        prepare_map_data,
        generate_map_metadata,
        generate_map_figure,
    )
except ImportError:
    from data_manipulation import (
        load_health_covid_data,
        FILEPATH,
        prepare_map_data,
        generate_map_metadata,
        generate_map_figure,
    )

app = dash.Dash(__name__)


#* GERAÇÃO DO MAPA
df_health = load_health_covid_data(FILEPATH)
df_map = prepare_map_data(df_health)

@app.callback(
    dash.dependencies.Output('map-graph', 'figure'),
    [dash.dependencies.Input('metric-dropdown', 'value')]
)
def update_map(metric):
    return generate_map_figure(df_map, metric)

app.layout = html.Div(children=[
    html.H1(children='CoVID-19 Dashboard', style={'textAlign': 'center'}),
    html.Div(children=[
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'CFR (%)', 'value': 'cfr_pct'},
                {'label': 'Total Cases', 'value': 'total_cases'}
            ],
            value='cfr_pct'
        ),
        dcc.Graph(
            figure=generate_map_figure(df_map, 'cfr_pct'),
            id='map-graph'
        ),])])
     

if __name__ == '__main__':
    app.run(debug=True)