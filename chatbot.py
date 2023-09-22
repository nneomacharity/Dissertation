import dash                              
from dash import html
from dash import dcc
from dash.dependencies import Output, Input, State
from datetime import date
from dash_extensions import Lottie       
import dash_bootstrap_components as dbc  
import plotly.express as px                                  
import calendar

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Flowise Chatbot Integration with Dash", style={"textAlign": "center"}),
    html.Div(id="chatbot-container", children=[
        # Embed the chatbot script
        html.Script(type="module", src="https://cdn.jsdelivr.net/npm/flowise-embed/dist/web.js", children=[
            '''
            import Chatbot from "https://cdn.jsdelivr.net/npm/flowise-embed/dist/web.js";
            Chatbot.init({
                chatflowid: "384f6b06-799a-4d31-9167-42854083a7bf",
                apiHost: "http://localhost:3000",
            });
            '''
        ])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

