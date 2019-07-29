import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pickle
import pandas as pd
import plotly.graph_objs as go

dist = pickle.load(open('./osm_multiplex/data/lstm_dist.pickle', 'rb'))
collected = pickle.load(open('./osm_multiplex/data/lstm_collected_target.pickle', 'rb'))
predicted = pickle.load(open('./osm_multiplex/data/lstm_predicted_target.pickle', 'rb'))

thresholds = {}
for location, results in dist.items():
    thresholds[location] = results['threshold']
    del results['threshold']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Label('Location'),
        dcc.Dropdown(
            id="location_dropdown",
            options=[
                {'label': location, 'value': location} for location, _ in dist.items()
            ],
            value=list(dist.keys())[0]
        )
    ]),

    html.Div([
        dcc.Graph(
            id='dist_threshold'
        )
    ]),
    html.Div([
        dcc.Graph(
            id='collected_predicted'
        )
    ])
])

@app.callback(
    [Output('dist_threshold', 'figure')],
    [Input('location_dropdown', 'value')]
)
def update_location(location):
    x_week=[week for week in dist[location].keys()]
    dist_graph = {
        'data':[
        go.Scatter(
            x=x_week,
            y=[result[1] for result in dist[location].values()],
            mode='lines',
            name='Predicted Distance'
        ),
        go.Scatter(
            x=x_week,
            y=[thresholds[location]] * len(x_week),
            name='Threshold'
        )
        ],
        'layout':{
            'xaxis':{'type':'category'}
        }
    }
    return [dist_graph]

@app.callback(
    [Output('collected_predicted', 'figure')],
    [Input('dist_threshold', 'clickData'),
     Input('location_dropdown', 'value')]
)
def update_week(click, location):
    if click is None:
        week = 0
    else:
        week = click['points'][0]['pointIndex']
    x_week = list(range(len(collected[location][week])))

    week_graph = {
        'data':[
            go.Scatter(
                x=x_week,
                y=collected[location][week],
                mode='lines',
                name='Collected Difference'
            ),
            go.Scatter(
                x=x_week,
                y=predicted[location][week],
                mode='lines',
                name='Predicted Difference'
            )
        ]
    }
    return [week_graph]
    

if __name__ == '__main__':
    app.run_server(debug=True)