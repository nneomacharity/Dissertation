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

#inputting animations for the headers all generated from lottie - https://lottiefiles.com/
likes = "https://assets10.lottiefiles.com/datafiles/hvAaKBDVLhuV5Wl/data.json"
viewers = "https://assets7.lottiefiles.com/packages/lf20_i48xonfk.json"
retweets = "https://assets5.lottiefiles.com/packages/lf20_iF9sFw.json"
comments = "https://assets9.lottiefiles.com/packages/lf20_1joxr8cy.json"
shares = "https://assets10.lottiefiles.com/packages/lf20_OBNxe4.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))


#household files 
from base import df, interface
from top import likes_text, retweet_text, comment_text, reshare_text, viewers_text
from other import x_dropdown, y_dropdown, chart_type_dropdown

#Building the layout and interface
interface.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardImg(src='/assets/twitter-logo2.png') # 150px by 45px
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
                    dcc.DatePickerSingle(
                        id='beginning_from',
                        date=date(2020, 1, 1),
                        className='ml-5'
                    ),
                    dcc.DatePickerSingle(
                        id='to_end_of',
                        date=date(2023, 8, 31),
                        className='mb-2 ml-2'
                    ),
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
    dbc.Row([
        dbc.Col([           #Scatter Plot  for similarity Analysis
            dbc.Card([
                dbc.CardBody([
                        dcc.Dropdown(
                            id='tweet-selector',
                            options=[{'label': f'Tweet {i}', 'value': i} for i in range(len(df))],
                            value=random.randint(0, len(df) - 1),  # Initialize with a random tweet                          
                    ),
                        dcc.Graph(id='Scatter-plot', figure={}),
                ])
            ]),
        ],width=6),
        dbc.Col([            #Line chart for Content Analysis 
            dbc.Card([
                dbc.CardBody([
                    html.H6('Slide through to pick the number of top words you desire'),
                           dcc.Slider(
                                id='top-words-slider',
                                min=0,
                                max=50,
                                step=1,
                                value=10,
                                marks={i: str(i) for i in range(0, 51, 10)},
                                tooltip={'placement': 'bottom'}
                            ),
                            dcc.Graph(id='line-chart', figure={}),
                ])
            ]),
        ], width=6),
    ],className='mb-2'),  
    dbc.Row([
        dbc.Col([           #Pie chart for Sentiment Analysis
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='pie-chart', figure={}),
                ])
            ]),
        ], width=6),
        dbc.Col([           #Word chart for 
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='wordcloud', figure={}),
                ])
            ]),
        ],), #width=6),
    ],className='mb-2'),
dbc.Row([                   #Bar chart for specific tweet analysis
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                dcc.Dropdown(
                    id='tweet-selector-bar-chart',
                    options=[{'label': tweet, 'value': tweet} for tweet in df['Text']],
                    value=df['Text'][0],  # Set the initial value to the first tweet in the dataset
                ),
                dcc.Graph(id='bar-chart', figure={}),
            ])
        ]),
    ], width=6),
        dbc.Col([               #Different charts for every other analysis
               dbc.Card([
                    dbc.CardBody([
                        html.H6("Any Other Analysis"),
                        html.Div([
                            html.Label('Select X-axis:'),
                            x_dropdown,
                        ]),
                        html.Div([
                            html.Label('Select Y-axis:'),
                            y_dropdown,
                        ]),
                        html.Div([
                            html.Label('Select Chart Type:'),
                            chart_type_dropdown,
                        ]),
                        dcc.Graph(id='chart')
                    ])
                ])
        ], width=6),
    ],className='mb-2'),
], fluid=True)
