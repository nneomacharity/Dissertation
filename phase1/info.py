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



#household files

from interface import Dashboard 


# Callback to  show a tweet info button


@Dashboard.callback(
    Output("info-section", "children"),
    [Input("show-tweets-button", "n_clicks")]
)
def show_tweets_info(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        # Loading the saved DataFrame
        df = pd.read_csv ('retrieved_tweets.csv')
        
        # Preparing the info of the DataFrame
        info = f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}"
        
        # Adding data types of each column
        info += "\n\n" + df.dtypes.to_string()  # data types of each column

        # Adding count of NaN values for each column
        nan_counts = df.isnull().sum()
        for column, count in nan_counts.items():
            info += f"NaN count in {column}: {count}\n"
        
        # Creating a preformatted text block to display the info
        info_block = dcc.Markdown(f"```\n{info}\n```", style={'whiteSpace': 'pre-line'})
        
        return info_block
    else:
        return None

