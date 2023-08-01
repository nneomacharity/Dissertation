#Libraries used
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

#block 3 - sentiment analysis - pie chart
#Sentiment analysis *****************************************************

def sentiment_analysis(df):
    # Initializing and usingthe VADER SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Performing sentiment analysis for each tweet and categorize them into 5 sentiment groups
    sentiments = []
    for text in df['Text']:
        score = analyzer.polarity_scores(text)
        compound_score = score['compound']

        if compound_score >= 0.8:
            sentiment = 'Highly Positive'
        elif 0.5 <= compound_score < 0.8:
            sentiment = 'Positive'
        elif -0.5 <= compound_score < 0.5:
            sentiment = 'Neutral'
        elif -0.8 <= compound_score < -0.5:
            sentiment = 'Negative'
        else:
            sentiment = 'Highly Negative'
        sentiments.append(sentiment)

    # Counting the occurrences of each sentiment category
    sentiment_counts = {sentiment: sentiments.count(sentiment) for sentiment in set(sentiments)}


    # Preparing data for the 3D pie chart
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    # Creating the 3D pie chart figure
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

    # Updating the layout for better visualization
    fig.update_layout(title='Sentiment Analysis',
                      margin=dict(l=0, r=0, t=30, b=0),
                      scene=dict(
                          aspectmode="cube",
                          camera=dict(
                              eye=dict(x=1.2, y=1.2, z=1.2)
                          )
                      )
                      )
    return fig

# Callback to update the pie chart plot for sentiment analysis
@interface.callback(
    Output('pie-chart', 'figure'),
    [Input('tweet-selector', 'value')]  # Assuming you have a tweet selector input
)
def update_pie_chart():
    fig = sentiment_analysis(df['Text'])
    return fig