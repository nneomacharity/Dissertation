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
#household files 
from base import df, interface

#block 2 - content analysis - line chart
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
