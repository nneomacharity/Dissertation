# installing  all the required libaries needed

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

import plotly.express as px

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
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
df = pd.read_csv('cleaned_tweets.csv')

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


#-------------------------------------------------------------
#PLEASE PLACE ALL MAIN PYTHON CODES HERE
#-------------------------------------------------------------

# Function to get sentiment
def get_sentiment(tweet):
    sentiment = sia.polarity_scores(tweet)
    if sentiment['pos'] > 0.1 and sentiment['neg'] > 0.1:
        return 'Mixed'
    elif sentiment['compound'] > 0.05:
        return 'Positive'
    elif sentiment['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Add sentiment to the DataFrame
df['Sentiment'] = df['Text'].apply(get_sentiment)
#get the number of unique sentiment - pos, neg, neutal, mixed
unique_sentiments = df['Sentiment'].unique()

#-----------------------------------------------------------
#CALL BACKS AND FUNCTIONS
#-----------------------------------------------------------

#PIE CHART------------------------------------------
# Callback to update the pie chart figure & horizontal bar chart
@interface.callback(
    Output('sentiment-pie', 'figure'),
    Input('sentiment-pie', 'clickData')  # Add any relevant input here
)
def update_sentiment_pie(click_data):
    # Count the number of tweets in each sentiment category
    sentiment_counts = df['Sentiment'].value_counts()
    fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title='Sentiment Counts',
        hover_data=[sentiment_counts.index, sentiment_counts.values],
        labels={'percent': '%'}
    )
    fig.update_traces(textinfo='percent+label')
    return fig


#------------------------------------------------------------------------------
#BASED ON SENTIMENT TYPE-------------------------------------------------------
#------------------------------------------------------------------------------
@interface.callback(
    Output('first-bar-chart', 'figure'),
    [Input('sentiment-dropdown', 'value'),
    Input('top-words-sentiment-slider', 'value')]
)
# Function to create horizontal bar chart
def update_horizontal_bar(selected_sentiment, slider_input):

    filtered_df = df[df['Sentiment'] == selected_sentiment]

    # Plot most frequent words
    words = ' '.join(filtered_df['Text']).split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    sentiment_top_words, sentiment_frequencies = zip(*sorted_word_counts[:int(slider_input)])

    fig = px.bar(
        x=sentiment_frequencies,
        y=sentiment_top_words,
        orientation='h',
        color=sentiment_top_words,
        text=sentiment_frequencies,
        labels={'x': 'Counts', 'y': 'Attributes'},
        title="Add a title"
    )
    fig.update_layout(yaxis_categoryorder='total ascending')
    fig.update_traces(textposition='inside')
    
    return fig

@interface.callback(
    Output('first-wordcloud', 'figure'),
    Input('sentiment-dropdown', 'value')  # Add any relevant input here
)
# Function to create word cloud
def update_sentiment_word_cloud(selected_sentiment):

    all_hashtag = ' '.join(df[df['Sentiment'] == selected_sentiment]['hashtag'])
    wordcloud = WordCloud(width=1000, height=800, background_color='white',
                          colormap='viridis', contour_color='steelblue').generate(all_hashtag)
    
    # Convert wordcloud image to Plotly figure
    wordcloud_fig = px.imshow(wordcloud.to_array(), binary_string=True)
    wordcloud_fig.update_layout(title="add a title", xaxis={'visible': False}, yaxis={'visible': False})
    
    return wordcloud_fig


# Callback to update tweet selector dropdown options and value

@interface.callback(
    [Output('tweet-selector-bar-chart', 'options'), Output('tweet-selector-bar-chart', 'value')],
    Input('sentiment-dropdown', 'value')
)
def update_tweet_selector_options(selected_sentiment):
    filtered_tweets = df[df['Sentiment'] == selected_sentiment]['Text']
    tweet_options = [{'label': tweet, 'value': index} for index, tweet in filtered_tweets.items()]
    return tweet_options, tweet_options[0]['value']


@interface.callback(
    Output('second-bar-chart', 'figure'),
    Input('tweet-selector-bar-chart', 'value')  # Add any relevant input here
)
# Function to create horizontal bar chart
def update_second_horizontal_bar(selected_row_index):
    # print("SELECTED TWEETERS",selected_row_index)

    # filtered_df = df[df['Sentiment'] == selected_sentiment]
    selected_comment = df.iloc[selected_row_index]
    selected_comment['RetweetCount'] = 0
    selected_comment['QuoteCount'] = 0


    attributes = ['ReplyCount', 'RetweetCount', 'LikeCount', 'QuoteCount']
    values = [selected_comment[attr] for attr in attributes]

    
    # filtered_df.to_csv('sentiment_filtered.csv', index = False)

    # Plot most frequent words
    
    # words = selected_tweet.split()
    # word_counts = Counter(words)

    # sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    # sentiment_top_words, sentiment_frequencies = zip(*sorted_word_counts)

    fig = px.bar(
        x=attributes,
        y=values,
        # color=attributes,
        # text=attributes,
        # labels={'x': 'Counts', 'y': 'Attributes'},
        # title="Add a title"
    )
    fig.update_layout(yaxis_categoryorder='total ascending')
    # fig.update_xaxes(showticklabels=False)  # Hide x-axis tickers (labels)
    fig.update_traces(textposition='inside')
    
    return fig


# Define callback to validate and process the input
@interface.callback(
    Output('validation-output', 'children'),
    Input('keyword-input', 'value')
)
def validate_input(keyword):
    if keyword is None:
        return None
    if re.match(r'^[a-zA-Z\s]*$', keyword):
        # Call your function with the keyword for further processing
        return keyword
    else:
        return 'Invalid input: Enter only words (no numbers or symbols)'

#---------------------------------------------------------
#BASED ON KEYWORD
#----------------------------------------------------------
@interface.callback(
    Output('third-bar-chart', 'figure'),
    [Input('sentiment-dropdown', 'value'),
    Input('validation-output', 'children'),
    Input('top-keyword-sliders', 'value')]  # Add any relevant input here
)
# Function to create horizontal bar chart
def update_keyword_horizontal_bar(selected_sentiment, keyword, slider_input):

    filtered_df = df[df['Sentiment'] == selected_sentiment]

    # Filter the DataFrame based on the search text
    filtered_df = filtered_df[filtered_df['Text'].str.contains(keyword, case=False)]

    # Plot most frequent words
    words = ' '.join(filtered_df['Text']).split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    sentiment_top_words, sentiment_frequencies = zip(*sorted_word_counts[:int(slider_input)])

    fig = px.bar(
        x=sentiment_frequencies,
        y=sentiment_top_words,
        orientation='h',
        color=sentiment_top_words,
        text=sentiment_frequencies,
        labels={'x': 'Counts', 'y': 'Attributes'},
        title="Add a title"
    )
    fig.update_layout(yaxis_categoryorder='total ascending')
    fig.update_traces(textposition='inside')
    
    return fig


@interface.callback(
    Output('second-wordcloud', 'figure'),
    [Input('sentiment-dropdown', 'value'),
    Input('validation-output', 'children')]
)
# Function to create word cloud
def update_sentiment_word_cloud(selected_sentiment, keyword):

    filtered_df = df[df['Sentiment'] == selected_sentiment]

    # Filter the DataFrame based on the search text
    filtered_df = filtered_df[filtered_df['Text'].str.contains(keyword, case=False)]

    all_hashtag = ' '.join(filtered_df['hashtag'])
    wordcloud = WordCloud(width=1000, height=800, background_color='white',
                          colormap='viridis', contour_color='steelblue').generate(all_hashtag)
    
    # Convert wordcloud image to Plotly figure
    wordcloud_fig = px.imshow(wordcloud.to_array(), binary_string=True)
    wordcloud_fig.update_layout(title="add a title", xaxis={'visible': False}, yaxis={'visible': False})
    
    return wordcloud_fig


@interface.callback(
    [Output('tweet-keyword-selector-bar-chart', 'options'), Output('tweet-keyword-selector-bar-chart', 'value')],
    [Input('sentiment-dropdown', 'value'),
    Input('validation-output', 'children')]
)
def update_tweet_selector_options(selected_sentiment, keyword):

    filtered_df = df[df['Sentiment'] == selected_sentiment]

    # Filter the DataFrame based on the search text
    filtered_df = filtered_df[filtered_df['Text'].str.contains(keyword, case=False)]

    tweet_options = [{'label': tweet, 'value': index} for index, tweet in filtered_df["Text"].items()]
    return tweet_options, tweet_options[0]['value']


@interface.callback(
    Output('fourth-bar-chart', 'figure'),
    Input('tweet-keyword-selector-bar-chart', 'value')  # Add any relevant input here
)
# Function to create horizontal bar chart
def update_fourth_horizontal_bar(selected_row_index):
    # words = selected_tweet.split()
    # word_counts = Counter(words)

    selected_comment = df.iloc[selected_row_index]

    selected_comment['RetweetCount'] = 0
    selected_comment['QuoteCount'] = 0
    attributes = ['ReplyCount', 'RetweetCount', 'LikeCount', 'QuoteCount']
    values = [selected_comment[attr] for attr in attributes]

    # sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    # sentiment_top_words, sentiment_frequencies = zip(*sorted_word_counts)

    fig = px.bar(
        x=attributes,
        y=values,
        # color=attributes,
        # text=attributes,
        # labels={'x': 'Counts', 'y': 'Attributes'},
        # title="Add a title"
    )
    fig.update_layout(yaxis_categoryorder='total ascending')
    # fig.update_xaxes(showticklabels=False)  # Hide x-axis tickers (labels)
    fig.update_traces(textposition='inside')
    
    return fig



#----------------------------------------------------------
#----------------------------------------------------------


#chat-gpt integration
@interface.callback(
    Output('chat-history', 'value'),  # Update the chat history textarea
    Input('send-button', 'n_clicks'),
    State('user-input', 'value'),
    State('chat-history', 'value')  # Maintain the existing chat history
)
def generate_response(n_clicks, user_message, chat_history):
    if n_clicks is None:
        return dash.no_update  # No update until the button is clicked

    if user_message:
        conversation_history = f'{chat_history}\nUser: {user_message}\nAI:'
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose an appropriate engine
            prompt=conversation_history,
            max_tokens=50  # Adjust the response length as needed
        )
        ai_reply = response.choices[0].text.strip()
        conversation_history += f' {ai_reply}\n'
        return conversation_history

    return chat_history  # If user_message is empty, return the current chat history


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
                    dcc.Graph(id='sentiment-pie')
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
                    dcc.Dropdown(
                id='sentiment-dropdown',
                options=[{'label': sentiment, 'value': sentiment} for sentiment in unique_sentiments],
                value=unique_sentiments[0]  # Set a default value
            )
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
                        id='top-words-sentiment-slider',
                        min=5,
                        max=20,
                        step=2,
                        value=5,
                        marks={i: str(i) for i in range(0, 21, 5)},
                        tooltip={'placement': 'bottom'}
                    ),
                    dcc.Graph(id='first-bar-chart'),
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
                        options=[],  # Placeholder for options, will be populated by the callback
                        value="",    # Placeholder for value, will be set by the callback
                    ),
                    dcc.Graph(id='second-bar-chart', figure={}),
                ])
            ]),
        ], width=4),
    ], className='mb-2'),


# 3rd filtering
    dbc.Row([
        dbc.Col([], width=3),  # Empty column to create space on the left
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Input a keyword'),
                    dcc.Input(id='keyword-input', type='text', placeholder='Enter a keyword'),
                    html.Div(id='validation-output', style={'color': 'red'}),
                    dbc.Button('Search', id='search-filter-button', color='primary')
                ])
            ]),
        ], width=6),
         dbc.Col([], width=3),  # Empty column to create space on the right
    ], className='mb-2'),

#filter out tweets having the keyword, save to a new csv file and then display the following


 #Barchart for top keywords in tweets having xxx keyword' from 'selected sentiments'
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Top Words surrounding the Keyword typed In'),
                    html.H6('Slide through to pick the number of top words you desire', style={'color': 'blue'}),
                    dcc.Slider(
                        id='top-keyword-sliders',
                        min=5,
                        max=20,
                        step=2,
                        value=5,
                        marks={i: str(i) for i in range(0, 21, 5)},
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
                        id='tweet-keyword-selector-bar-chart',
                        options=[],  # Placeholder for options, will be populated by the callback
                        value='',    # Placeholder for value, will be set by the callback
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
                        html.H5("ChatBot", style={'textalign': 'center'}),
                        html.H6("Ask a Question", style={'color': 'blue'}),
                        dcc.Textarea(id='chat-history', readOnly=True),
                        html.Div([
                            dcc.Input(id='user-input', type='text'),
                            html.Button('Send', id='send-button', n_clicks=0)
                        ])
                    ])
                ]),
        ], width=8,
        className='mx-auto my-auto'),

        dbc.Col([], width=2),  # Empty column to create space on the left
    ],className='h-500'),



], fluid=True)



# Tweets insight from selected sentiment
#tweet-selector-bar-chart

# @interface.callback(
#     Output('second-bar-chart', 'figure'),
#     [Input('tweet-selector-bar-chart', 'value')]  # Add any relevant input here
# )

# def update_bar_chart(selected_text):

#Run the app
if __name__=='__main__':
    interface.run_server(debug=False, port=8090)


