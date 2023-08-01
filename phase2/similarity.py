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


#block 1 - similarity analysis analysis - scattered plot
#Similarity  analysis *****************************************************
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Text'].astype('U').values)

#Computing the Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix)

#Clustering Using K-Means
num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)
# Function for Finding Similar Tweets
def find_similar_tweets(tweet_index, num_similar=5):
    row = cosine_sim[tweet_index]
    similar_indices = np.argsort(row)[-num_similar-1:-1][::-1]
    return similar_indices, row[similar_indices]

# Callback to update the scatter plot for similar tweets
@interface.callback(
    Output('scatter-plot', 'figure'),
    [Input('tweet-selector', 'value')]
)
def update_scatter_plot(tweet_index):
    # TruncatedSVD for 2D visualization
    tsvd = TruncatedSVD(n_components=2, random_state=42)
    reduced_tfidf = tsvd.fit_transform(tfidf_matrix)

    # Get similar tweets
    similar_indices, _ = find_similar_tweets(tweet_index)

    # Scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reduced_tfidf[similar_indices, 0],
        y=reduced_tfidf[similar_indices, 1],
        mode='markers',
        marker=dict(color='blue', size=10),
        name='Similar Tweets',
    ))

    fig.add_trace(go.Scatter(
        x=[reduced_tfidf[tweet_index, 0]],
        y=[reduced_tfidf[tweet_index, 1]],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Selected Tweet',
    ))

    fig.update_layout(
        title='Similarity Analysis: Similar Tweets Visualization',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        height=400
    )

    return fig