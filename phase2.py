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
    Output('bar-chart', 'figure'),
    [Input('top-words-slider', 'value')]
)
def update_bar_chart(num_topics):
    # Get the top trending topics based on the slider value
    trending_topics = content_analysis(df, 'Text', num_topics)

    # Getting the corresponding frequency of each trending topic
    topic_counts = df['Text'].str.split(expand=True).stack().value_counts()
    topic_counts = topic_counts[trending_topics]

    # Creating the bar chart
    fig = go.Figure(data=[go.Bar(x=trending_topics, y=topic_counts)])

    fig.update_layout(
        title="Top Trending Topics, Keywords and Hashtags",
        xaxis_title="Topic",
        yaxis_title="Frequency",
        height=400
    )
    return fig





'''
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

'''



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
            'title': f'Analysis for Tweet: {selected_tweet}',
            'barmode': 'group',
        }
    }



#Other Analysis  *****************************************************

#Creating dropdowns for X-axis and Y-axis selection
x_dropdown = dcc.Dropdown(
    id='x-axis-dropdown',
    options=[{'label': col, 'value': col} for col in df.columns],
    value=df.columns[0]  # Setting a default column as the initial value
)

y_dropdown = dcc.Dropdown(
    id='y-axis-dropdown',
    options=[{'label': col, 'value': col} for col in df.columns],
    value=df.columns[1]  # Setting another default column as the initial value
)

# Creating dropdown for chart type selection
chart_type_dropdown = dcc.Dropdown(
    id='chart-type-dropdown',
    options=[
        {'label': 'Bar Chart', 'value': 'bar'},
        {'label': 'Scatter Plot', 'value': 'scatter'},
        {'label': 'line Chart', 'value': 'line'},
        {'label': 'Pie Chart', 'value': 'pie'},
    ],
    value='bar'  # Setting a default chart type as the initial value
)

# Callback to update plots for any other column comparisons
@interface.callback(
    Output('chart', 'figure'),
    [
        Input('x-axis-dropdown', 'value'),
        Input('y-axis-dropdown', 'value'),
        Input('chart-type-dropdown', 'value')
    ]
)
def other_analysis(x_column, y_column, chart_type):
    if chart_type == 'bar':
        figure = go.Figure(data=[go.Bar(x=df[x_column], y=df[y_column])])
    elif chart_type == 'scatter':
        figure = go.Figure(data=[go.Scatter(x=df[x_column], y=df[y_column], mode='markers')])
    # Add more conditions for other chart types

    # Customize the layout of the chart (optional)
    figure.update_layout(
        title=f"{y_column} vs {x_column}",
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    
    return figure




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
    
#Laying Out the Graphs
#Scatter Plot  for similarity Analysis

    dbc.Row([
        dbc.Col([
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

 

 #Line chart for Content Analysis 
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


#Pie chart for   
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='pie-chart', figure={}),
                ])
            ]),
        ], width=6),

#Word chart for 
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='wordcloud', figure={}),
                ])
            ]),
        ],), #width=6),
    ],className='mb-2'),

#Bar chart for tweet analysis
dbc.Row([
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

#Different charts for every other analysis
        dbc.Col([
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

if __name__=='__main__':
    interface.run_server(debug=False, port=8002)



''''
@interface.callback(
    [Output('total-likes', 'children'),
     Output('selected-tweet-likes', 'children')],
    [Input('tweet-selector', 'value')]
)
def likes(selected_tweet):
    # Assuming 'LikeCount' column contains the number of likes for each tweet
    if 'LikeCount' in df.columns:
        total_likes = df['LikeCount'].sum()
        if selected_tweet is not None:
            likes_for_selected_tweet = df.iloc[selected_tweet]['LikeCount']
        else:
            likes_for_selected_tweet = 0  # Default to 0 if no tweet is selected
        return str(total_likes), str(likes_for_selected_tweet)
    else:
        return 'Data Unavailable', 'Data Unavailable'
'''