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




df = pd.read_csv(('cleaned_tweets.csv'))
interface = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE]) 

import layout
import top
import similarity
import content
import sentiment
import specific
import other

if __name__=='__main__':
    interface.run_server(debug=False, port=8002)
