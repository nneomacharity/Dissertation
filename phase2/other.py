

import dash                              
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
from datetime import date
from dash_extensions import Lottie       
import dash_bootstrap_components as dbc  
import plotly.express as px                                  
import calendar
from wordcloud import WordCloud 
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#household files 
from base import df, interface


#block 6 - other analysis - line chart
#Other Analysis  *****************************************************

#Creating dropdowns for X-axis and Y-axis selection
x_dropdown = dcc.Dropdown(
    id='x-axis-dropdown',
    options=[{'label': col, 'value': col} for col in df.columns],
    value=df.columns[0]  # Setting a default column as the initial value
)

y_dropdown = dcc.Dropdown(
    id='y-axis-dropdown',
    options=[{'label': col, 'value': col} for col in df.columns],
    value=df.columns[1]  # Setting another default column as the initial value
)

# Creating dropdown for chart type selection
chart_type_dropdown = dcc.Dropdown(
    id='chart-type-dropdown',
    options=[
        {'label': 'Bar Chart', 'value': 'bar'},
        {'label': 'Scatter Plot', 'value': 'scatter'},
        {'label': 'line Chart', 'value': 'line'},
        {'label': 'Pie Chart', 'value': 'pie'},
    ],
    value='bar'  # Setting a default chart type as the initial vgraphical representation
)

# Callback to update plots for any other column comparisons
@interface.callback(
    Output('chart', 'figure'),
    [
        Input('x-axis-dropdown', 'value'),
        Input('y-axis-dropdown', 'value'),
        Input('chart-type-dropdown', 'value')
    ]
)
def other_analysis(x_column, y_column, chart_type):
    if chart_type == 'bar':
        figure = go.Figure(data=[go.Bar(x=df[x_column], y=df[y_column])])
    elif chart_type == 'scatter':
        figure = go.Figure(data=[go.Scatter(x=df[x_column], y=df[y_column], mode='markers')])
    # Add more conditions for other chart types

    # Customize the layout of the chart (optional)
    figure.update_layout(
        title=f"{y_column} vs {x_column}",
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    
    return figure