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

#household files 
from base import df, interface



#block 5 - Specific tweet analysis - bar chart

def tweet_analysis(selected_tweet):
    # Filter the dataset to get the row corresponding to the selected tweet
    tweet_row = df[df['Text'] == selected_tweet]

    # Extract the number of likes, retweets, shares, etc. from the row
    likes = tweet_row['LikeCount'].values[0]
    comments = tweet_row['ReplyCount'].values[0]
    retweets = tweet_row['RetweetCount'].values[0]
    shares = tweet_row['ShareCount'].values[0]
    hashtags = tweet_row['hastag_counts'].values[0]

    # Create a bar chart with the extracted data
    fig = go.Figure(data=[
        go.Bar(name='Likes', x=['Likes'], y=[likes]),
        go.Bar(name='Retweets', x=['Retweets'], y=[retweets]),
        go.Bar(name='Shares', x=['Shares'], y=[shares]),
        go.Bar(name='Comments', x=['Comments'], y=[comments]),
        go.Bar(name='Hashtags', x=['Hashtags'], y=[hashtags]),
    ])
 
    # Update the layout of the chart as needed
    fig.update_layout(title=f"Specicific Tweet Analysis: {selected_tweet}")

    return fig

#Specific Tweet Analysis  *****************************************************

@interface.callback(
    Output('bar-chart', 'figure'),
    [Input('tweet-selector-bar-chart', 'value')]
)

def tweet_analysis(selected_tweet):
    available_columns = ['RetweetCount', 'LikeCount', 'ShareCount', 'ReplyCount', 'hastag_counts']  

    data = []
    for col in available_columns:
        if col in df:
            # Plot the bar for the available column
            data.append(go.Bar(
                x=[col],
                y=[df.loc[df['Tweet'] == selected_tweet, col].iloc[0]],
                name=col.capitalize()  # Show column name as the legend
            ))
        else:
            # If the column is not available, then the program should add an empty trace
            data.append(go.Bar(
                x=[col],
                y=[0],
                name=col.capitalize()  # Show column name as the legend
            ))

    return {
        'data': data,
        'layout': {
            'title': f'Analysis for Each Tweet: {selected_tweet}',
            'barmode': 'group',
        }
    }
