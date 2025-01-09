import dash 
from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor, XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

df = px.data.iris()
columns = df.columns
species_encoder = LabelEncoder()

# encoding the species column 
df['species_encoded'] = species_encoder.fit_transform(df['species'])

app = Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#f4f8fb', 'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Iris Dataset Dashboard", style={'textAlign': 'center', 'color': '#0047ab', 'marginBottom': '30px'}),
    
    # Dropdown to select target feature
    html.Div([
        html.Label('Select Target Feature', style={'color' : '#0047ab', 'fontsize' : '18px'}),
        dcc.Dropdown(
            id = 'target-feature-dropdown',
            options = [{'label' : col, 'value' : col} for col in df.columns if col != 'species_encoded'],   # we will later on map the species encoded to species in the drop down selction 
            value=columns[0],
            style={'width' : '50%', 'marginBottom' : '20px'} 
        )
    ]),
    
    
    # line splot for all the features
    html.Div([
        dcc.Graph(id='line-plot')
    ], style={'marginBottom' : '30px'}),
    
    # Dropdown for regular techniques
    html.Div([
        html.Label("selct regularization Techniques : ", style={'color' : '#0047ab', 'fontSize' : '18px'}),
        dcc.Dropdown(
            id='regularization-dropdown',
            options=[
                {'label' : 'None',             'value' : 'none'},
                {'label' : 'Normalization',    'value' : 'normalization'},
                {'label' : 'Standardization',  'value' : 'standardization'}
            ],
            value='none',
            style={'width' : '50%', 
                   'marginBottom' : '20px'}
        )
    ]),
    
    #Drop down for selecting the model
    html.Div([
        html.Label('Select Machine Learning Model : ', style={'color' : '#0047ab', 'fontSize' : '18px'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label' : 'Linear Regression', 'value' : 'linear'},
                {'label' : 'Logistic Regression', 'value' : 'logistic'},
                {'label' : 'Random Forest Regression', 'value' : 'random_forest_reg'},
                {'label' : 'Random Forest Classifier', 'value' : 'random_forest_cls'},
                {'label' : 'XGBoost Regressor', 'value' : 'xgboost_reg'},
                {'label' : 'XGBoost Classifier', 'value' : 'xgboost_cls'}
            ],
            value='linear',
            style={'width' : '50%', 'marginBottom' : '20px'}
        ) 
    ]),
    
    # Model Performance
    html.Div([
        html.H4('Model Performance', style={'color' : '#0047ab'}),
        html.Div(id='model-perfomance', style={'fontSize' : '18px', 'marginTop' : '10px'})
    ], style={'marginBottom' : '30px'}),
    
    # Input section for prediction 
    html.Div([
        html.H4("Input values for Prediction", style={'color' : '#0047ab'}),
        html.Div(id='input-field', style={'marginBottom' : '20px'}),
        html.Button('Predict', id='predict-button', n_clicks=0,
                    style={'backgroundColor' : '#0047ab', 'color' : 'white', 'border' : 'none', 'padding' : '10px 20px'}),
        html.Div(id='prediction-output', style={'#fontSize' : '18px', 'MarginTop' : '20px', 'color' : '#0047ab'})
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
