import dash
from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Group
from dash.exceptions import PreventUpdate
import tweepy
import pycountry
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import pandas as pd
import os
import csv
import io
from datetime import datetime as dt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')



# Building the interface of the welcome page using Dash
Dashboard = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.PULSE],
                      meta_tags=[{'name': 'viewport',
                                  'content': 'width=device-width, initial-scale=1.0'}]
                      )

# Adding a background image to the welcome page
Dashboard.layout = html.Div(
    style={
        'background-image': 'url(https://images.unsplash.com/photo-1683560044376-6ac69ce791c5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1375&q=80)',
        'background-size': 'cover',
        'height': '100vh',
    },
    children=[
        dcc.Store(id='server-side-store', storage_type='session'),
        dbc.Container([
            dbc.Row(dbc.Col(html.H1("Interactive Analytics of Twitter in Real Time", style={"textAlign": "center", 'color':"white"}), width=40 )),
            html.Hr(),  # Adding a horizontal line as a divisor

            dbc.Row(
                dbc.Col(
                    [
                        html.H3("Hi! I'm Lemonade AI, let's make a juice!", style={"textAlign": "center", 'color':"white"}),  # Adding a welcome address
                        html.P("Click the button below to hand me some lemons:", style={"textAlign": "center", 'color':"white"}),
                        dbc.Button("Start", id="start-button", color="primary", className="mr-2",
                                   style={"display": "block", "margin": "auto"}),
                    ],
                    width=12,
                    className="mt-2",
                    align="center"  # Aligning the content vertically within the column
                ),
                align="center",  # Aligning the row vertically within the container
                className="mt-2"  # Adding a top margin to the row
            ),

            dbc.Row(
                dbc.Col(id="input-section", width=6, className="mt-4", style={"margin": "auto"}),
                className="justify-content-center"
            ),

            dbc.Row(
                dbc.Col(id="date-section", className="mt-4", style={"margin": "auto"}),
                className="justify-content-center"
            ),

            dbc.Row(
                dbc.Col(id="retrieve-section", className="mt-4", style={"margin": "auto"}),
                className="justify-content-center"
            ),

            dbc.Row(
                dbc.Col(id="output-section", className="mt-4", style={"margin": "auto"}),
                className="justify-content-center"
            ),

            dbc.Row(
                dbc.Col(id="info-section", className="mt-4", style={"margin": "auto"}),
                className="justify-content-center"
            ),
            dbc.Row(
                dbc.Col(id="cleaning-tweets-section", className="mt-4", style={"margin": "auto"}),
                className="justify-content-center"
            ),
            dbc.Row(
                dbc.Col(id="cleaning-options-section", className="mt-4", style={"margin": "auto"}),
                className="justify-content-center"
            ),
            dbc.Row(
            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id="dashboard-button-section", className="mt-4", style={"margin": "auto"})
            ),
            className="justify-content-center"
        ),

        ], style={"max-width": "500px", "margin": "auto"})
    ]
)