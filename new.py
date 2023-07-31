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

#Generating the basic codes for each of the graphical chart backend****

#content  analysis *****************************************************
def content_analysis(df, column_name, num_topics=10):
    # Count the occurrences of each word in the column
    word_counts = df[column_name].str.split(expand=True).stack().value_counts()
    
    # Extract the most common words as trending topics
    trending_topics = word_counts.head(num_topics).index.tolist()
    return trending_topics

# Define the callback to update the bar chart for content analysis
@interface.callback(
    Output('line-chart', 'figure'),
    [Input('top-words-slider', 'value')]
)
def update_line_chart(num_topics):
    # Get the top trending topics based on the slider value
    trending_topics = content_analysis(df, 'Text', num_topics)

    # Getting the corresponding frequency of each trending topic
    topic_counts = df['Text'].str.split(expand=True).stack().value_counts()
    topic_counts = topic_counts[trending_topics]

    # Creating the line chart
    fig = go.Figure(data=[go.Line(x=trending_topics, y=topic_counts)])

    fig.update_layout(
        title="Top Trending Topics, Keywords and Hashtags",
        xaxis_title="Topic",
        yaxis_title="Frequency",
        height=400
    )
    return fig



#Other Sections------------------------------

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
    total_reshare = df['ShareCount'].sum(
        
    )
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





#working on the layout and interface
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




 #Line chart for Content Analysis 
    dbc.Row([
        dbc.Col([
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
], fluid=True)

if __name__=='__main__':
    interface.run_server(debug=False, port=8002)



