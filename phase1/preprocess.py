
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Group
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import subprocess
nltk.download('stopwords')



#household files

from interface import theapi, Dashboard 

#preprocessing stage

@Dashboard.callback(
    Output("cleaning-tweets-section", "children"),
    [Input("show-tweets-button", "n_clicks")]
)
def clean_tweets_button(n_clicks):
    if n_clicks is not None and  n_clicks > 0:
        return [
            dbc.Button("Preprocess Tweets", id="clean-tweets-button", color="primary", className="mr-2", style={"display": "block", "margin": "auto"}),
        ]
    else:
        return None

# Show cleaning options and 'Preprocess' button
@Dashboard.callback(
    Output("cleaning-options-section", "children"),
    [Input('clean-tweets-button', 'n_clicks')]
)
def show_cleaning_options(n_clicks):
    if n_clicks and n_clicks > 0:
        return [
            html.P("Choose one or more of these options:"),
            dcc.Checklist(
                id='cleaning-options',
                options=[
                    {'label': 'Convert tweet to lowercase', 'value': 'lowercase'},
                    {'label': 'Remove punctuations', 'value': 'punctuations'},
                    {'label': 'Remove stopwords', 'value': 'stopwords'},
                    {'label': 'Remove digits', 'value': 'digits'},
                    {'label': 'Remove URLs', 'value': 'urls'},
                    {'label': 'Remove numerical values', 'value': 'numerical_values'},
                    {'label': 'Remove non-alphabetic characters', 'value': 'non_alphabetic_characters'},
                    {'label': 'Remove missing data', 'value': 'missing_data'},
                ],
                value=[]
            ),
            dbc.Button("Start Preprocessing", id="preprocess-button", color="primary", className="mr-2", style={"display": "block", "margin": "auto"})
        ]
    else:
        return None


def clean_tweets(df, options):
    if 'lowercase' in options:
        df['Text'] = df['Text'].str.lower()
    if 'punctuations' in options:
        df['Text'] = df['Text'].str.replace('[^\w\s]', '')
    if 'stopwords' in options:
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    if 'digits' in options:
        df['Text'] = df['Text'].str.replace('\d+', '')
    if 'urls' in options:
        df['Text'] = df['Text'].replace(to_replace=r'http\S+', value='', regex=True).replace(to_replace=r'www\S+', value='', regex=True)
    if 'numerical_values' in options:
        df['Text'] = df['Text'].str.replace('\d+', '')
    if 'non_alphabetic_characters' in options:
        df['Text'] = df['Text'].str.replace('[^a-zA-Z]', ' ')
    if 'missing_data' in options:
        df.dropna(subset=['Text'], inplace=True)
    return df


# Clean tweets and show 'See Dashboard' button
@Dashboard.callback(
    Output("dashboard-button-section", "children"),
    [Input("preprocess-button", "n_clicks")],
    [State("cleaning-options", "value")]
)
def clean_and_display_tweets(n_clicks, cleaning_options):
    if n_clicks and n_clicks > 0:
        # Loading the tweets data
        df = pd.read_csv('retrieved_tweets.csv')

        # Cleaning the tweets
        cleaned_df = clean_tweets(df, cleaning_options)

        # Saving the cleaned tweets to a CSV file
        cleaned_df.to_csv('cleaned_tweets.csv', index=False)

        # Displaying the "See Dashboard" button
        return [
            dbc.Spinner([
                dbc.Button("See Dashboard", id="see-dashboard-button", color="primary", className="mr-2", style={"display": "block", "margin": "auto"})
            ], color="primary", type="grow"),
            # Adding a hidden div to trigger the callback to open phase2.py
            html.Div(id="open-phase2-div", style={"display": "none"})
        ]
    else:
        return None

# Callback to open and run phase2.py when the "See Dashboard" button is clicked
@Dashboard.callback(
    Output("open-phase2-div", "children"),
    [Input("see-dashboard-button", "n_clicks")]
)
def open_phase2_script(n_clicks):
    if n_clicks and n_clicks > 0:
        python_executable = "python"
        phase2_script = "phase2.py"

        # Running phase2.py using subprocess
        subprocess.run([python_executable, phase2_script])

    return None

if __name__ == "__main__":
    Dashboard.run_server(debug=True)

