import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
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

# Create a Dash app
app = dash.Dash(__name__)

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

# Load the data
df = pd.read_csv('cleaned_tweets.csv')

# Add sentiment to the DataFrame
df['Sentiment'] = df['Text'].apply(get_sentiment)



# Count the number of tweets in each sentiment category
sentiment_counts = df['Sentiment'].value_counts()

# Layout of the Dash app
app.layout = html.Div([
    dcc.Graph(id='sentiment-pie')
])

# Callback to update the pie chart figure
@app.callback(
    Output('sentiment-pie', 'figure'),
    Input('sentiment-pie', 'clickData')  # Add any relevant input here
)
def update_sentiment_pie(click_data):
    fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title='Sentiment Counts',
        hover_data=[sentiment_counts.index, sentiment_counts.values],
        labels={'percent': '%'}
    )
    fig.update_traces(textinfo='percent+label')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
