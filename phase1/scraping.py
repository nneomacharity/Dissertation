import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
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
from datetime import datetime as dt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')



#household files

from interface import theapi, Dashboard,countries, geocoder, Tweet

# Adding a start button to commence activity
@Dashboard.callback(Output("input-section", "children"), [Input("start-button", "n_clicks")], [State('input-section', 'children')])
def show_input(n_clicks, children):
    if n_clicks is not None and n_clicks > 0:
        return [
            dcc.Input(id="input-keyword", placeholder="Enter a keyword", type="text", className="mb-2",
                      style={"width": "100%"}),
            dcc.Dropdown(id='country-dropdown', options=countries, placeholder='Select a country')
        ]
    else:
        return children

# Adding the dropdown of countries in the world for the user to choose from
@Dashboard.callback(
    Output("date-section", "children"),
    Output("date-section", "style"),
    [Input("country-dropdown", "value")],
    [State("date-section", "children")]
)
# Adding a date range or calender/ a request for the number of tweets to be retieved
def show_date(value, children):
    if value is not None:
        return (
            [
                dbc.Row([
                    dbc.Col([
                        dcc.DatePickerRange(id='date-picker-start', className="mb-2",  start_date_placeholder_text='Tweets from',
                        end_date_placeholder_text='Till'),
                    ], width=12, className="m-auto")
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Input(id='tweets-input', type='number', placeholder='Number of tweets', min=0, step=1),
                    ], width=6, className="m-auto")
                ]),
            ],
            {},  # making the date-section visible as it removes the 'display: none' style
        )
    else:
        return children, {"display": "none"}
    

# Adding a retrieve button for twitter scraping to be initiated
@Dashboard.callback(Output("retrieve-section", "children"), Output("retrieve-section", "style"),
                    [Input("tweets-input", "value")], [State("retrieve-section", "children")])
def retrieve_tweets(value, children):
    if value is not None and value > 0:
        return( 
            [dbc.Button("Retreive Tweets", id="retrieve-button", color="primary", className="mr-2", style={"display": "block", "margin": "auto"}),
            html.P(
                "N.B: You may have to modify the API logins to retrieve large quantity of tweets.",
                style={"fontSize": "13px", "color": "white", 'textAlign': 'center'}),
             dbc.Button("Show Tweets Info", id="show-tweets-button", color="primary", className="mr-2", 
                style={"display": "block", "margin": "auto"})
            
            ],
            {}
        )
    else:
        return children, {"display": "none"}

# Callback to start fetching the tweets
@Dashboard.callback(Output('server-side-store', 'data'),
                    [Input('retrieve-button', 'n_clicks')],
                    [State('input-keyword', 'value'),
                     State('country-dropdown', 'value'),
                     State('date-picker-start', 'start_date'),
                     State('date-picker-start', 'end_date'),
                     State('tweets-input', 'value'),
                     State('server-side-store', 'data')])
def retrieve_and_store_tweets(n_clicks, keyword, country, start_date, end_date, tweets_number, data):
    if n_clicks and keyword and country and start_date and end_date and tweets_number:
        data = data or {}
        data['status'] = 'loading'
        location = geocoder(country)
        latitude = location.latitude
        longitude = location.longitude
        
        query = f"{keyword} geocode:{latitude},{longitude},1000km"
        tweets = []
        for tweet in tweepy.Cursor(theapi.search_tweets, q=keyword, lang="en", since=start_date, until=end_date, tweet_mode='extended').items(tweets_number):
            tweet_text = tweet.full_text
            tweet_id = tweet.id
            likes = tweet.favorite_count
            retweets = tweet.retweet_count
            comments = tweet.reply_count
            shares = tweet.quote_count
            created_at = tweet.created_at.strftime("%Y-%m-%d %H:%M:%S")
                
            new_tweet = Tweet(tweet_id, tweet_text, likes, retweets, comments, shares, created_at)
            tweets.append(new_tweet)

        # Saving the retrieved tweets into a csv file
        file_path = os.path.join('C:/Users/DELL/Desktop/Dissertation/', 'retrieved_tweets.csv')
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'tweet', 'country', 'keyword'])
            writer.writeheader()
            writer.writerows(tweets)
        data['status'] = 'done'
        return data
    else:
        raise PreventUpdate

# Callback to handle and display spinner and messages
@Dashboard.callback(Output('output-section', 'children'),
                    [Input('server-side-store', 'modified_timestamp')],
                    [State('server-side-store', 'data')])
def display_output(ts, data):
    if ts is None:
        raise PreventUpdate

    data = data or {}
    status = data.get('status')

    if status == 'loading':
        return dbc.Spinner(color="primary")
    elif status == 'done':
        return html.P("Tweets retrieved and saved.")
    else:
        return None