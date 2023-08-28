# installing and all the required libaries needed

import dash                              
from dash import html
from dash import dcc
from dash.dependencies import Output, Input, State
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
import openai

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

#Top Section------------------------------

#Likes
# Checking if the 'LikesCount' column exists in the DataFrame
if 'LikeCount' in df.columns:
    # Calculating the sum of likes
    total_likes = df['LikeCount'].sum()
    # Setting the text to display the total likes
    likes_text = str(total_likes)
else:
    # Setting the text to display 'Data Unavailable'
    likes_text = 'Data Unavailable'

#Retweets
# Checking if the 'RetweetCount' column exists in the DataFrame
if 'RetweetCount' in df.columns:
    # Calculating the sum of retweets
    total_retweet = df['RetweetCount'].sum()
    # Setting the text to display the total retweet
    retweet_text = str(total_retweet)
else:
    # Setting the text to display 'Data Unavailable'
    retweet_text = 'Data Unavailable'

#Reshare block
# Checking if the 'RetweetCount' column exists in the DataFrame
if 'ReshareCount' in df.columns:
    # Calculating the sum of shares
    total_reshare = df['ShareCount'].sum()
    # Setting the text to display the total reshare
    reshare_text = str(total_reshare)
else:
    # Setting the text to display 'Data Unavailable'
    reshare_text = 'Data Unavailable'

#Comments block
# Checking if the 'CommentCount' column exists in the DataFrame
if 'ReplyCount' in df.columns:
    # Calculating the sum of comments
    total_comment = df['ReplyCount'].sum()
    # Setting the text to display the total commments
    comment_text = str(total_comment)
else:
    # Setting the text to display 'Data Unavailable'
    comment_text = 'Data Unavailable'

#Viewers block
# Checking if the 'ViewersCount' column exists in the DataFrame
if 'ViewersCount' in df.columns:
    # Calculating the sum of viewers
    total_viewers = df['ViewersCount'].sum()
    # Setting the text to display the total commments
    viewers_text = str(total_viewers)
else:
    # Setting the text to display 'Data Unavailable'
    viewers_text = 'Data Unavailable'




#chat-gpt integration
@interface.callback(
    Output('chat-history', 'value'),
    [Input('send-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('chat-history', 'value')]
)
def update_chat(n, user_message, chat_history):
    if not n or not user_message:
        # No button clicks or empty message, don't update
        return chat_history

    # Interact with OpenAI and get a response
    response_message = "Sorry, I couldn't understand that."  # Default response
    
    try:
        response = openai.Completion.create(engine="davinci", prompt=user_message, max_tokens=100)
        response_message = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error with OpenAI: {e}")

    # Now, append the user's message and the bot's response to the chat history
    updated_history = (chat_history or "") + f"\nUser: {user_message}\nBot: {response_message}"

    return updated_history
print("Callback triggered")   
# Setting up OpenAI API
openai.api_key = "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"


#Building the layout and interface
interface.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.CardLink("Go Back", target="_blank",
                                 href="http://127.0.0.1:8050"
                    )
                ])
            ]),
        ], width=1),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1('Interactive Analysis of Tweets', style={'color': 'black', 'textAlign': 'center'}),
                ])
            ], color="info"),
        ], width=11),
    ],className='mb-2 mt-2'),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardImg(src='/assets/Xlogo.png') # 150px by 45px
            ]),
        ], width=2),


         dbc.Col([   
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=likes)),
                dbc.CardBody([
                    html.H5('Total Likes'),
                    html.H6(id='total_likes', children=likes_text, style={'color': 'blue'}),
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=viewers)),
                dbc.CardBody([
                    html.H5('Total Viewers'),
                    html.H6(id='views', children=viewers_text, style={'color': 'blue'})
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=retweets)),
                dbc.CardBody([
                    html.H5('Total Retweets'),
                    html.H6(id='retweet', children=retweet_text, style={'color': 'blue'})
                ], style={'textAlign': 'center'})
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
    ], className='mb-2'),



    
#Laying Out the Graphs
#1st filtering
#Scatter Plot  for Sentiments Analysis

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Degrees of Sentiments In the Data'),
                    dcc.Graph(id='Scatter-plot', figure={}),
                ])
            ]),
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Clusters of Sentiments'),
                    dcc.Graph(id='pie-chart', figure={}),
                ])
            ]),
        ], width=6),
    ], className='mb-2'),

#new row
    dbc.Row([
        dbc.Col([], width=3),  # Empty column to create space on the left
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Pick a sentiment class'),
                    # Attach a dropdown of the names of each sentiment
                ])
            ]),
        ], width=6, align='center'),  # Center the column content
        
        dbc.Col([], width=3),  # Empty column to create space on the right
    ], className='mb-2'),

     dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Top Keywords in Selected Sentiment'),
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
                    dcc.Graph(id='first-bar-chart', figure={}),
                ])
            ]),
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Hashtags Used in Selected Sentiment'),
                    dcc.Graph(id='first-wordcloud', figure={}),
                ])
            ]),
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Tweets insight from selected sentiment'),
                    html.H6('Select a tweet', style={'color': 'blue'}),
                    dcc.Dropdown(
                        id='tweet-selector-bar-chart',
                        options=[{'label': tweet, 'value': tweet} for tweet in df['Text']],
                        value=df['Text'][0],  # Set the initial value to the first tweet in the dataset
                    ),
                    dcc.Graph(id='second-bar-chart', figure={}),
                ])
            ]),
        ], width=4),
    ], className='mb-2'),


#3rd filtering
    dbc.Row([
        dbc.Col([], width=3),  # Empty column to create space on the left
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Input a keyword'),
#attach an input box for keyword
                ])
            ]),
        ], width=6),
         dbc.Col([], width=3),  # Empty column to create space on the right
    ],className='mb-2'),

#filter out tweets having the keyword, save to a new csv file and then display the following


 #Barchart for top keywords in tweets having xxx keyword' from 'selected sentiments'
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Top Words surrounding the Keywor typed In'),
                    html.H6('Slide through to pick the number of top words you desire', style={'color': 'blue'}),
                    dcc.Slider(
                        id='top-words-sliders',
                        min=0,
                        max=50,
                        step=1,
                        value=10,
                        marks={i: str(i) for i in range(0, 51, 10)},
                        tooltip={'placement': 'bottom'}
                    ),
                    dcc.Graph(id='third-bar-chart', figure={}),
                ])
            ]),
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Hashtags used in Tweets having the Keyword'),
                    dcc.Graph(id='second-wordcloud', figure={}),
                ])
            ]),
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Evaluation Metrics of Tweets'),
                    html.H6('Select a tweet', style={'color': 'blue'}),
                    dcc.Dropdown(
                        id='tweet-selector-bar-charts',
                        options=[{'label': tweet, 'value': tweet} for tweet in df['Text']],
                        value=df['Text'][0],  # Set the initial value to the first tweet in the dataset
                    ),
                    dcc.Graph(id='fourth-bar-chart', figure={}),
                ])
            ]),
        ], width=4),
    ], className='mb-2'),

#chatgpt integration

    dbc.Row([
        dbc.Col([], width=2),  # Empty column to create space on the left

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("ChatGPT", style={'textAlign': 'center', 'margin-bottom': '20px'}),  # Increase the size for prominence and add margin for spacing
                    
                    html.Div("Ask a Question", style={'color': 'blue', 'textAlign': 'center', 'margin-bottom': '10px'}),  # Using a div to allow for more styling flexibility
                    
                    dcc.Textarea(
                        id='chat-history',
                        readOnly=True,
                        style={"width": "100%", "height": "250px", "margin-bottom": '10px'}
                    ),
                    
                    html.Div([
                        dcc.Input(
                            id='user-input',
                            type='text',
                            placeholder='Type your message here...',
                            style={"width": "80%", "display": "inline-block"}  # Adjusted to 80% width for a little spacing
                        ),
                        html.Button('Send', id='send-button', n_clicks=0, style={"width": "18%", "display": "inline-block", "margin-left": "2%"})
                    ], style={"width": "100%", "textAlign": "center"})  # Center align the input and button
                ])
            ]),
        ], width=8, className='mx-auto my-auto'),

        dbc.Col([], width=2),  # Empty column to create space on the right
    ], className='h-500'),
])
if __name__=='__main__':
    interface.run_server(debug=False, port=8002)

