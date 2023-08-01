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


# Defining the Tweet class
class Tweet:
    def __init__(self, tweet_id, text, likes, retweets, comments, shares, created_at):
        self.tweet_id = tweet_id
        self.text = text
        self.likes = likes
        self.retweets = retweets
        self.comments = comments
        self.shares = shares
        self.created_at = created_at

# Generating a list of countries using pycountry
countries = [{'label': country.name, 'value': country.alpha_2} for country in pycountry.countries]

#Geocoding the Countries
def do_geocode(address):
    geolocator = Nominatim(user_agent="myGeocoder")
    try:
        return geolocator.geocode(address)
    except GeocoderTimedOut:
        return do_geocode(address)



geocoder = do_geocode

# Connecting to the Twitter API using credentials provided from my developer account
consumer_key = 'PuQc9u7k0aAWBeC5iuvA50phL'
consumer_secret = 'gHnv2OsSMaYM0TAjVhksWcw7JER0f5Ur1aGJnXje5WQS5iUJm1'
access_key = '924681622288510980-AkQbiEXVKODsfIECQgQPMNjworq8dvG'
access_secret = 'CuIqjmoDO2UisyZHDi8nZqb8d0expwedM6eWQGGsCb7oO'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

theapi = tweepy.API(auth)



# Building the interface of the welcome page using Dash
Dashboard = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.PULSE],
                      meta_tags=[{'name': 'viewport',
                                  'content': 'width=device-width, initial-scale=1.0'}]
                    )

#calling all the other parts of the codes
from scraping import show_input
from scraping import show_date
from scraping import retrieve_tweets
from scraping import retrieve_and_store_tweets
from scraping import display_output
from info import show_tweets_info
from preprocess import clean_tweets_button
from preprocess import show_cleaning_options
from preprocess import clean_tweets
from preprocess import clean_and_display_tweets

if __name__ == "__main__":
    Dashboard.run_server(debug=True)
