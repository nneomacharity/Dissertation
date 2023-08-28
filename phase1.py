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
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import word_tokenize
import nltk
import subprocess
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

# Connecting to the Twitter API using credentials provided from my developer account
consumer_key = 'PuQc9u7k0aAWBeC5iuvA50phL'
consumer_secret = 'gHnv2OsSMaYM0TAjVhksWcw7JER0f5Ur1aGJnXje5WQS5iUJm1'
access_key = '924681622288510980-AkQbiEXVKODsfIECQgQPMNjworq8dvG'
access_secret = 'CuIqjmoDO2UisyZHDi8nZqb8d0expwedM6eWQGGsCb7oO'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

theapi = tweepy.API(auth)

# Building the interface of the welcome page using Dash
Dashboard = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CERULEAN],
                      meta_tags=[{'name': 'viewport',
                                  'content': 'width=device-width, initial-scale=1.0'}]
                      )

# Adding a background image to the welcome page
Dashboard.layout = html.Div(
    style={
        'background-image': 'url(https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2071&q=80)',
        'background-size': 'cover',
        'background-repeat': 'no-repeat',
        'background-attachment': 'fixed', #This makes sure the background image doesn't scroll */
        'min-height': '100vh',
    },
    children=[
        dcc.Store(id='server-side-store', storage_type='session'),
        dbc.Container([
            dbc.Row(dbc.Col(html.H1("Retrieval Section For Generating Tweets Insights", style={"textAlign": "center", 'color':"black", 'width':"100"}), width=12 )),
            html.Hr(),  # Adding a horizontal line as a divisor

            dbc.Row(
                dbc.Col(
                    [
                        html.H3("Hello!", style={"textAlign": "center", 'color':"black"}),  # Adding a welcome address
                        html.P("Click the button below to start:", style={"textAlign": "center", 'color':"black"}),
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
            
                dbc.Col(id="date-section", width=8, className="mt-4", style={"margin": "auto"}),
                
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
            dcc.Loading(
                id="loading-icon",
                children=[html.Div(id="info-section")],
                type="default"
            ),
            dbc.Row(
                dbc.Col(id="info-sectionn", className="mt-4", style={"margin": "auto"}),
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
                        end_date_placeholder_text='Tweets Till'),
                    ], width=16, className="m-auto")
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Input(id='tweets-input', type='number', placeholder='Number of tweets', min=0, step=1),
                    ], width=6, className="m-auto")
                ]),
            ],
            {},  # this will make the date-section visible as it removes the 'display: none' style
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
                style={"fontSize": "15px", "color": "black", 'textAlign': 'center','font-weight': 'bold'}),
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
#function for tweet retieval
def retrieve_and_store_tweets(n_clicks, keyword, country, start_date, end_date, tweets_number, data):
    if n_clicks and keyword and country and start_date and end_date and tweets_number:
        data = data or {}
        data['status'] = 'loading'
        location = do_geocode(country)
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
# Callback to show a tweet info button
@Dashboard.callback(
    Output("info-sectionn", "children"),
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

        
        # Creating a preformatted text block to display the info
        info_block = dcc.Markdown(f"```\n{info}\n```", style={'whiteSpace': 'pre-line'})
        
        return info_block
    else:
        return None


#preprocessing stage

@Dashboard.callback(
    Output("cleaning-tweets-section", "children"),
    [Input("show-tweets-button", "n_clicks")]
)
def clean_tweets_button(n_clicks):
    if n_clicks is not None and  n_clicks > 0:
        return [
            dbc.Button("Preprocess Tweets", id="clean-tweets-button", color="primary", className="mr-2", style={"display": "block", "margin": "auto"}),
        ]
    else:
        return None

# Showing the cleaning options and  the'Preprocess' button
@Dashboard.callback(
    Output("cleaning-options-section", "children"),
    [Input('clean-tweets-button', 'n_clicks')]
)
def show_cleaning_options(n_clicks):
    if n_clicks and n_clicks > 0:
        return [
            html.P("Choose one or more of these options:", style={'color': 'black', 'font-weight': 'bold'}),
            dcc.Checklist(
                id='cleaning-options',
                options=[
                    {'label': 'Convert tweet to lowercase', 'value': 'lowercase'}, # indicating the possible 
                    {'label': 'Remove punctuations', 'value': 'punctuations'},
                    {'label': 'Remove stopwords', 'value': 'stopwords'},
                    {'label': 'Remove digits', 'value': 'digits'},
                    {'label': 'Remove URLs', 'value': 'urls'},
                    {'label': 'Remove numerical values', 'value': 'numerical_values'},
                    {'label': 'Remove non-alphabetic characters', 'value': 'non_alphabetic_characters'},
                    {'label': 'Remove empty rows', 'value': 'missing_data'},
                ],
                value=[],
                style={'color': 'white', 'font-weight': 'bold'},
            ),
            dbc.Button("Start Preprocessing", id="preprocess-button", color="primary", className="mr-2", style={"display": "block", "margin": "auto"})
        ]
    else:
        return None
stop = stopwords.words('english')
def clean_tweets(df, options):
    # Check if 'Text' column exists
    if 'Text' not in df.columns:
        raise ValueError("The expected 'Text' column is missing in the dataframe.")
    
    # Convert tweet to lowercase
    if 'lowercase' in options:
        df['Text'] = df['Text'].str.lower()
    
    # Remove punctuations
    if 'punctuations' in options:
        df['Text'] = df['Text'].str.replace('[^\w\s]', '', regex=True)
    
    # Remove stopwords
    if 'stopwords' in options:
        df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    # Remove URLs
    if 'urls' in options:
        df['Text'] = df['Text'].replace(to_replace=r'http\S+', value='', regex=True).replace(to_replace=r'www\S+', value='', regex=True)
    
    # Remove digits and numerical values (combined both checks to remove redundancy)
    if 'digits' in options or 'numerical_values' in options:
        df['Text'] = df['Text'].str.replace('\d+', '', regex=True)
    
    # Remove non-alphabetic characters
    if 'non_alphabetic_characters' in options:
        df['Text'] = df['Text'].str.replace('[^a-zA-Z]', ' ', regex=True)
    
    # Remove empty rows or rows with missing 'Text' data
    if 'missing_data' in options:
        df.dropna(subset=['Text'], inplace=True)
    
    return df



# Clean tweets and show 'See Dashboard' button
@Dashboard.callback(
    Output("dashboard-button-section", "children"),
    [Input("preprocess-button", "n_clicks")],
    [State("cleaning-options", "value")]
)
def clean_and_display_tweets(n_clicks, cleaning_options):
    if n_clicks and n_clicks > 0:
        # Loading the tweets data
        df = pd.read_csv('retrieved_tweets.csv')

        # Cleaning the tweets
        cleaned_df = clean_tweets(df, cleaning_options)

        # Saving the cleaned tweets to a CSV file
        cleaned_df.to_csv('cleaned_tweets.csv', index=False)

        # Displaying the "See Dashboard" button
        return [
            dbc.Spinner([
        html.A(
            dbc.Button("See Dashboard", color="primary", className="mr-2", style={"display": "block", "margin": "auto"}),
            href="http://localhost:8002", target="_blank"
        )
    ], color="primary", type="grow"),
        ]
    else:
        return None

if __name__ == "__main__":
    Dashboard.run_server(debug=False)




#new things to add
#cleaning option for filtering out the keyword used for retrieval