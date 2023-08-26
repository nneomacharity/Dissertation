# installing and all the required libaries needed

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
import matplotlib.pyplot as plt
from collections import Counter
import requests
import os
from transformers import pipeline

# using a theme  from bootstthemes by Ann: https://hellodash.pythonanywhere.com/theme_explorer
#picking a purple colour theme also known as PULSE
interface = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE]) 


#inputting animations for the headers all generated from lottie - https://lottiefiles.com/
likes = "https://assets10.lottiefiles.com/datafiles/hvAaKBDVLhuV5Wl/data.json"
viewers = "https://assets7.lottiefiles.com/packages/lf20_i48xonfk.json"
retweets = "https://assets5.lottiefiles.com/packages/lf20_iF9sFw.json"
comments = "https://assets9.lottiefiles.com/packages/lf20_1joxr8cy.json"
shares = "https://assets10.lottiefiles.com/packages/lf20_OBNxe4.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))



#reading the cleaned tweets csv file
df = pd.read_csv(('cleaned_tweets.csv'))


#Building the layout and interface
interface.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardImg(src='/assets/Xlogo2.png') # 150px by 45px
            ],className='mb-2'),
            dbc.Card([
                dbc.CardBody([
                    dbc.CardLink("Go Back", target="_blank",
                                 href="link to the csv file"
                    )
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([

                ])
            ], color="info"),
        ], width=8),
    ],className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=likes)),
                dbc.CardBody([
                    html.H5('Total Likes'),
                    html.H6(id='total_likes',  children=likes_text, style={'color': 'blue'}),
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=viewers)),
                dbc.CardBody([
                    html.H5('Total Viewers'),
                    html.H6(id='views', children=viewers_text, style={'color': 'blue'})
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=retweets)),
                dbc.CardBody([
                    html.H5('Total Retweets'),
                    html.H6(id='retweet', children=retweet_text, style={'color': 'blue'})
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=comments)),
                dbc.CardBody([
                    html.H5('Total Comments'),
                    html.H6(id='comments', children=comment_text, style={'color': 'blue'})
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=shares)),
                dbc.CardBody([
                    html.H5('Total Shares'),
                    html.H6(id='reshare', children=reshare_text, style={'color': 'blue'})
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),
    ],className='mb-2'),
    
#Laying Out the Graphs
#1st filtering
#Scatter Plot  for Sentiments Analysis

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Sentiment Analysis'),
                    html.H6('Select a tweet', style={'color': 'blue'}),
                        dcc.Dropdown(
                            id='tweet-selector',
                            options=[{'label': f'Tweet {i}', 'value': i} for i in range(len(df))],
                            value=random.randint(0, len(df) - 1),  # Initialize with a random tweet                          
                    ),
                        dcc.Graph(id='Scatter-plot', figure={}),
                ])
            ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Percentages of Sentiments'),
                    dcc.Graph(id='pie-chart', figure={}),
                ])
            ]),
        ], width=6),

        
        ],width=6),

     dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Pick a sentiment class'),
#attach a drowpdow of the names of each sentiments
                ])
            ]),
        ], width=6),
    ],className='mb-2'),


 #Barchart for top keywords in 'selected' sentiments
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Top Keywords in XXX Sentiments'),
                    html.H6('Slide through to pick the number of top words you desire', style={'color': 'blue'}),
                           dcc.Slider(
                                id='top-words-slider',
                                min=0,
                                max=50,
                                step=1,
                                value=10,
                                marks={i: str(i) for i in range(0, 51, 10)},
                                tooltip={'placement': 'bottom'}
                            ),
                            dcc.Graph(id='bar-chart-top', figure={}),
                ])
            ]),
        ], width=6),
    ],className='mb-2'),
 

#Wordcloud for hashtags used in 'selected' sentiment
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('HASHTAG ANALYSIS'),
                    dcc.Graph(id='wordcloud', figure={}),
                ])
            ]),
        ],), #width=6),
    ],className='mb-2'),

#list of tweets of 'selected sentiments' and their metrics to measure performance
dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H5('INDIVIDUAL TWEET INSIGHTS'),
                html.H6('Select a tweet', style={'color': 'blue'}),
                    dcc.Dropdown(
                        id='tweet-selector-bar-chart',
                        options=[{'label': tweet, 'value': tweet} for tweet in df['Text']],
                        value=df['Text'][0],  # Set the initial value to the first tweet in the dataset
                    ),
                    dcc.Graph(id='bar-chart', figure={}),
                ])
            ]),
        ], width=6),


#3rd filtering
     dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Input a keyword'),
#attach an input box for keyword
                ])
            ]),
        ], width=6),
    ],className='mb-2'),

#filter out tweets having the keyword, save to a new csv file and then display the following


 #Barchart for top keywords in tweets having xxx keyword' from 'selected sentiments'
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Top Words surrounding xxs xearch'),
                    html.H6('Slide through to pick the number of top words you desire', style={'color': 'blue'}),
                           dcc.Slider(
                                id='top-words-slider',
                                min=0,
                                max=50,
                                step=1,
                                value=10,
                                marks={i: str(i) for i in range(0, 51, 10)},
                                tooltip={'placement': 'bottom'}
                            ),
                            dcc.Graph(id='bar-chart-top', figure={}),
                ])
            ]),
        ], width=6),
    ],className='mb-2'),
 

#Wordcloud for hashtags used in tweets having xxx from 'selected sentiment'
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Hashtags'),
                    dcc.Graph(id='wordcloud', figure={}),
                ])
            ]),
        ],), #width=6),
    ],className='mb-2'),

#list of tweets of carrying 'xxx word' and their metrics to measure performance
dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H5('INDIVIDUAL TWEET INSIGHTS'),
                html.H6('Select a tweet', style={'color': 'blue'}),
                    dcc.Dropdown(
                        id='tweet-selector-bar-chart',
                        options=[{'label': tweet, 'value': tweet} for tweet in df['Text']],
                        value=df['Text'][0],  # Set the initial value to the first tweet in the dataset
                    ),
                    dcc.Graph(id='bar-chart', figure={}),
                ])
            ]),
        ], width=6),


#chatgpt integration
    dbc.Row([
        dbc.Col([
               dbc.Card([
                    dbc.CardBody([
                        html.H5("Type something"),
                        html.Div([
                            dcc.Textarea(id='chat-history', readOnly=True),
                            dcc.Input(id='user-input', type='text'),
                            html.Button('Send', id='send-button', n_clicks=0)
                        ])
                    ])
                ])
        ], width=6),
    ],className='mb-2'),
], fluid=True)

if __name__=='__main__':
    interface.run_server(debug=False, port=8002)


