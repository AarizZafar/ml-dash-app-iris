import dash 
from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = px.data.iris()                         # loading dataset

species_encoder = LabelEncoder()            # label encoder
df['species_encoded'] = species_encoder.fit_transform(df['species'])

columns = df.columns                        # extracting all coulmns names 

# initialize the app 
app = Dash(__name__)

# app layout
app.layout = html.Div(style={'backgroundColor' : '#f4f8fb' , 'fontFamily' : 'Arial, sans-serif', 'padding' : '20px'}, 
                      children = [
    html.H1("Iris DashBoard", style={'textAlign' : 'center', 'color' : '#0047ab', 'marginBottom' : '30px'}),
                          
    # Dropdown to select target feature
    html.Div([
    html.Label('Select Target Feature', style={'color' : '#0047ab', 'fontSize' : '18px'}),
        dcc.Dropdown(
            id='target-feature-dropdown',
            options=[{'label' : col, 'value' : col} for col in df.columns if col != 'species_encoded'],   # the encoded column will not be in the drop down
            value=None,
            style={'width' : '50%', 'marginBottom' : '20px'}
        )
    ]),
                    
    # Dropdown for regularization techniques 
    html.Div([
        html.Label('Select Regularization Technique', style={'color' : '#0047ab', 'fontSize' : '18px'}),
        dcc.Dropdown(
            id='regularization-dropdown',
            options=[
                {'label' : 'None', 'value' : 'none'},
                {'label' : 'Normalization', 'value' : 'normalization'},
                {'label' : 'Standardization', 'value' : 'standardization'}
            ],
            value='none',
            style={'width' : '50%', 'marginBottom' : '20px'}
        )
    ], id='regularization-div', style={'display' : 'none'}),    # display - None makes the drop down not appear 
    
    # Dropdown foe Machine Learning model
    html.Div([
        html.Label('Select Machine Learning Model', style={'color' : '#0047ab', 'fontSize' : '18px'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[],
            value=None,
            style={'width' : '50%', 'marginBottom' : '20px'}
        )
    ], id='model-div', style={'display' : 'none'}),
    
    # Line plot for feature 
    html.Div([
        dcc.Graph(id='line-plot')
    ], style={'marginBottom' : '30px'}),
    
    # Model Performance output
    html.Div([
        html.H4('Model Performance', style={'color' : '$0047ab'}),
        html.Div(id='model-performance', style={'fontsize' : '18px', 'marginTop' : '10px'})
    ], style={'marginBottom' : '30px'}),
    
    # Input fields for prediction 
    html.Div([
        html.H4("Input Value for Prediction", style={'color' : '#0047ab'}),
        html.Div(id='input-fields', style={'marginBottom' : '20px'}),
        html.Button('predict', id='predict-button', n_clicks=0,
                    style={'backgroundColor' : '#0047ab', 'color' : 'white', 'border' : 'none', 'padding' : '10px 20px'}),
        html.Div(id='prediction-output', style={'fontsize' : '18px', 'marginTop' : '20px', 'color' : '#0047ab'})
    ])
])


@app.callback(
    [Output('regularization-div', 'style'),
     Output('model-div', 'style'),
     Output('model-dropdown', 'options')],
    Input('target-feature-dropdown', 'value')
)

def toggle_visibility(target_feature):
    if not target_feature:                                  # Till we do not select the target feature we will hide the ML and Regularization drop down
        return {'display' : 'none'}, {'display' : 'none'}, []       
    
    isCata = target_feature == 'species'                    
    model_options = [                                       # the drop down to be shown when we select a target feature 
                                                            # if categorical value classification models
        {'label' : 'Logistic Regression',           'value' : 'logistic'},
        {'label' : 'Random Forest Classifier',      'value' : 'random_forest_cls'},
        {'label' : 'XGBoost Classifier',            'value' : 'xgboost_cls'}
    ] if isCata else [                                      # if continous value - regression models 
        {'label' : 'Linear Regression' ,            'value' : 'linear'},
        {'label' : 'Random Forest Regressor',       'value' : 'random_forest_reg'},
        {'label' : 'XGBoost Regressor',             'value' : 'xgboost_reg'}
    ]
    
    return {'display' : 'block'}, {'display' : 'block'}, model_options

@app.callback(
    [Output('line-plot',               'figure'),
     Output('model-performance',       'children'),
     Output('input-fields',            'children'),
     Output('prediction-output',       'children')],
    [Input('target-feature-dropdown',   'value'),
     Input('regularization-dropdown',   'value'),
     Input('model-dropdown',            'value')]
)

def update_dashboard(target_feature, regularization, model_type):
    if not target_feature or not model_type:
        return {}, '', [], ''
    
    isCata = target_feature == 'species'
    
    # splitting the data
    x = df.drop(columns=[target_feature, 'species'] if 'species' in df.columns else target_feature)
    y = df['species_encoded'] if target_feature == 'species' else df[target_feature]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # applying regularization 
    if regularization == 'normalization':
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif regularization == 'standardization':
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
    line_fig = px.line(df.drop(columns=['species']), title='Line Plot of Features', template='plotly_white')
    line_fig.update_layout(plot_bgcolor='#f4f8fb', paper_bgcolor='#f4f8fb', title_font_color='#0047ab')
    
    # train selected model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'logistic':
        model = LogisticRegression(max_iter=200)
    elif model_type == 'random_forest_reg':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'random_forest_cls':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'xgboost_reg':
        model = XGBRegressor(random_state=42)
    elif model_type == 'xgboost_cls':
        model = XGBClassifier(random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    if not isCata:                      # Regression Metrix
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        performance = f'Model : {model_type.replace('_', ' ').title()} - Mean Squared Error : {mse:.4f}, R2 Score {r2:.4f}'
    else:                               # Classifcation metrix 
        y_pred_class = (y_pred > 0.5).astype(int) if model_type in ['logistic', 'xgboost_cls'] else y_pred
        accuracy = accuracy_score(y_test, y_pred_class)
        performance = f'Model : {model_type.replace('_', ' ').title()} - Accuracy : {accuracy:.4f}'
        
    input_fields = [
        html.Div([
            html.Label(f'{col}:', style={'color' : '#0047ab'}),
            dcc.Input(id=f'input - {col}', type='number', placeholder=f'Enter {col}', style={'marginBottom' : '10px'})
        ]) for col in x.columns if col != target_feature
    ]
    
    return line_fig, performance, input_fields, ''

if __name__ == '__main__':
    app.run(debug=True)