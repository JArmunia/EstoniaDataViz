from dash import Dash, dcc, html, Input, Output, callback, State, ctx
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from pyproj import CRS

# Cargar los datos
weather_gdf = gpd.read_parquet("weather_summary.parquet")
energy_prices = pd.read_parquet("energy_prices.parquet")

weather_gdf = weather_gdf.merge(energy_prices, on="datetime")

# Asegurarse de que gdf está en WGS84
if weather_gdf.crs is None or weather_gdf.crs.to_string() != 'EPSG:4326':
    weather_gdf = weather_gdf.set_crs('EPSG:4326', allow_override=True)
    weather_gdf = weather_gdf.to_crs('EPSG:4326')

# Cargar el shapefile de Estonia
ee_gdf = gpd.read_file("shapefiles/ee.shp")
train = pd.read_parquet("train.parquet")

ee_gdf = ee_gdf.merge(train, left_on='name', right_on='county')

# Asegurarse de que ee_gdf está en WGS84
if ee_gdf.crs is None or ee_gdf.crs.to_string() != 'EPSG:4326':
    ee_gdf = ee_gdf.to_crs('EPSG:4326')

# Proyección adecuada para Estonia (Estonian Coordinate System of 1997)
estonia_crs = CRS.from_epsg(3301)

# Calcular el centroide en la proyección adecuada
ee_gdf_projected = ee_gdf.to_crs(estonia_crs)
centroid = ee_gdf_projected.geometry.centroid.to_crs('EPSG:4326')
center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

# Convertir el GeoDataFrame a GeoJSON
ee_geojson = json.loads(ee_gdf[["name", "geometry"]].drop_duplicates().to_json())

# Calcular el rango de temperaturas para todo el conjunto de datos
temp_min = weather_gdf['temperature'].min()
temp_max = weather_gdf['temperature'].max()

# Crear una escala de colores de azul a rojo para la temperatura
temp_color_scale = [
    (0, "#74d5ed"),  (0.5, "#ffe172"),  (1, "#ff574d")
]

# Configurar la aplicación Dash
app = Dash(__name__)
server = app.server

# Obtener la lista de fechas únicas
dates = weather_gdf['datetime'].dt.date.unique()
date_marks = {i: date.strftime('%Y-%m-%d') for i, date in enumerate(dates)}

weather_metrics = ['temperature', 'rain', 'cloudcover_total', 'snowfall', 'electricity_price_per_mwh', 'gas_price_per_mwh']

app.layout = html.Div([
    html.H1(id='title', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(id='map-graph', style={'height': '80vh'}),
        html.Div([
            dcc.Graph(id='time-series-graph', style={'height': '80vh', 'display': 'none'}),
            html.Button('Volver al mapa', id='return-button', n_clicks=0, style={'display': 'none'}),
            html.Div([
                dcc.Checklist(
                    id='weather-metrics-checklist',
                    options=[{'label': metric, 'value': metric} for metric in weather_metrics],
                    value=['temperature'],
                    inline=True
                )
            ], id='weather-metrics-container', style={'display': 'none'})
        ])
    ]),
    html.Div([
        dcc.Slider(
            id='date-slider',
            min=0,
            max=len(dates) - 1,
            value=0,
            marks=date_marks,
            step=None
        )
    ], id='slider-container', style={'width': '80%', 'margin': 'auto', 'padding': '20px'}),
    html.Div([
        dcc.RadioItems(
            id='energy-type-selector',
            options=[
                {'label': 'Generación', 'value': 0},
                {'label': 'Consumo', 'value': 1}
            ],
            value=0,
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        )
    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px'}),
    dcc.Store(id='clicked-county')
], style={'height': '100vh', 'width': '100vw', 'margin': 0, 'padding': 0})

@callback(
    Output('clicked-county', 'data'),
    [Input('map-graph', 'clickData'),
     Input('return-button', 'n_clicks')],
    State('clicked-county', 'data')
)
def store_clicked_county(clickData, n_clicks, current_county):
    ctx_msg = ctx.triggered_id
    if ctx_msg == 'return-button':
        return None
    elif clickData is not None:
        return clickData['points'][0]['location']
    return current_county

@callback(
    [Output('map-graph', 'style'),
     Output('time-series-graph', 'style'),
     Output('time-series-graph', 'figure'),
     Output('return-button', 'style'),
     Output('slider-container', 'style'),
     Output('weather-metrics-container', 'style')],
    [Input('clicked-county', 'data'),
     Input('energy-type-selector', 'value'),
     Input('return-button', 'n_clicks'),
     Input('weather-metrics-checklist', 'value')],
    [State('map-graph', 'style'),
     State('time-series-graph', 'style')]
)
def update_graphs(clicked_county, energy_type, n_clicks, selected_metrics, map_style, ts_style):
    ctx_msg = ctx.triggered_id
    if ctx_msg == 'return-button' or clicked_county is None:
        return {'height': '80vh', 'display': 'block'}, {'height': '80vh', 'display': 'none'}, go.Figure(), {'display': 'none'}, {'width': '80%', 'margin': 'auto', 'padding': '20px'}, {'display': 'none'}
    
    # Filtrar datos para el condado seleccionado
    county_data = ee_gdf[(ee_gdf['name'] == clicked_county) & (ee_gdf['is_consumption'] == energy_type)]
    weather_data = weather_gdf[weather_gdf['geometry'].apply(lambda point: point.within(county_data.geometry.iloc[0]))]
    weather_data = weather_data.groupby('datetime')[weather_metrics].mean().reset_index()
    
    # Crear figura de series temporales
    fig = go.Figure()
    # Añadir generación y consumo
    energy_type_label = "Consumo" if energy_type else "Generación"
    fig.add_trace(go.Scatter(x=county_data['datetime'], y=county_data['target'],
                             mode='lines', name=energy_type_label, yaxis='y2'))
    # Añadir métricas meteorológicas seleccionadas
    for metric in selected_metrics:
        fig.add_trace(go.Scatter(x=weather_data['datetime'], y=weather_data[metric],
                                 mode='lines', name=metric, yaxis='y'))
    
    
    
    fig.update_layout(
        title=f'Series Temporales para {clicked_county}',
        xaxis_title='Fecha',
        yaxis_title='Valores de métricas meteorológicas',
        yaxis2=dict(title=f'{energy_type_label} de Energía (MWh)', overlaying='y', side='right'),
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x unified'
    )
    
    return {'height': '80vh', 'display': 'none'}, {'height': '80vh', 'display': 'block'}, fig, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}

@callback(
    [Output('map-graph', 'figure'),
     Output('title', 'children')],
    [Input('date-slider', 'value'),
     Input('energy-type-selector', 'value')]
)
def update_map(selected_date_index, energy_type):
    selected_date = dates[selected_date_index]
    filtered_df = weather_gdf[weather_gdf['datetime'].dt.date == selected_date]

    filtered_ee_gdf = ee_gdf[(ee_gdf['datetime'].dt.date == selected_date) & 
                             (ee_gdf['is_consumption'] == energy_type)]
    
    # Si no hay datos para la fecha seleccionada, usa todos los datos del tipo de energía seleccionado
    if filtered_ee_gdf.empty:
        filtered_ee_gdf = ee_gdf[ee_gdf['is_consumption'] == energy_type]
    
    # Calcular el rango de energía
    energy_min = filtered_ee_gdf.target.min()
    energy_max = filtered_ee_gdf.target.max()
    
    # Normalizar los valores de energía
    filtered_ee_gdf['energy_normalized'] = (filtered_ee_gdf.target - energy_min) / (energy_max - energy_min)

    energy_type_label = "Consumo" if energy_type else "Generación"

    # Crear el mapa coroplético
    fig = px.choropleth_mapbox(filtered_ee_gdf, 
                               geojson=ee_geojson, 
                               locations='name', 
                               featureidkey="properties.name",
                               color='energy_normalized',  # Usar el valor normalizado para el color
                               color_continuous_scale=temp_color_scale,
                               range_color=[0, 1],  # Rango normalizado
                               mapbox_style="open-street-map",
                               zoom=6, 
                               center={"lat": center_lat, "lon": center_lon},
                               opacity=0.75,
                               labels={'target': f'{energy_type_label} de Energía (MWh)',
                                       'name': 'Condado'},
                               hover_data={'name': True, 
                                           'target': ':.2f'})  # Mostrar target con 2 decimales

    # Añadir el scatter plot encima con transparencia
    scatter_trace = px.scatter_mapbox(filtered_df,
                                      lat=filtered_df.geometry.y,
                                      lon=filtered_df.geometry.x,
                                      hover_name='weather_summary',
                                      color=(filtered_df.temperature-temp_min) / (temp_max-temp_min),
                                      text='weather_summary',
                                      size=np.array([10]*len(filtered_df)),
                                      color_continuous_scale=temp_color_scale,
                                      range_color=[0, 1],
                                      size_max=20,
                                      opacity=0.75,
                                      ).data[0]

    fig.add_trace(scatter_trace)
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            zoom=6,
            center={"lat": center_lat, "lon": center_lon}
        ),
        coloraxis_colorbar=dict(
            title=f"{energy_type_label} de Energía (MWh)",
            len=0.75,
            yanchor="top",
            y=0.99,
            x=0.99,
            xanchor="right",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=[f"{energy_min:.2f}", 
                      f"{energy_min + 0.25*(energy_max-energy_min):.2f}", 
                      f"{energy_min + 0.5*(energy_max-energy_min):.2f}", 
                      f"{energy_min + 0.75*(energy_max-energy_min):.2f}", 
                      f"{energy_max:.2f}"]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800
    )
    
    # Mantener los colores de los puntos pero ocultar su barra de color
    fig.update_traces(marker=dict(showscale=False), selector=dict(type='scattermapbox'))
    
    # Crear el título dinámico
    title = f"{energy_type_label} de Energía en Estonia - {selected_date.strftime('%Y-%m-%d')}"
    
    return fig, title
    
# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)