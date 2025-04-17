# src/web_app.py
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load the trained model
try:
    with open('tuned_model.pkl', 'rb') as file:  # Model is in src/
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: tuned_model.pkl not found in src/ directory.")

# Load the dataset for at-risk students and charts
try:
    data = pd.read_csv('../notebooks/Student_performance_data .csv')  # Path updated to notebooks/
except FileNotFoundError:
    print("Error: Student_performance_data .csv not found in notebooks/ directory.")
    data = pd.DataFrame()  # Fallback empty DataFrame to prevent crashes

# Compute dashboard metrics dynamically
avg_gpa = round(data['GPA'].mean(), 2) if not data.empty else 3.12
at_risk_count = len(data[data['GPA'] < 2.5]) if not data.empty else 245
avg_absences = round(data['Absences'].mean(), 1) if not data.empty else 14.5
tutoring_participation = round((data['Tutoring'].mean() * 100), 1) if not data.empty else 32

# Define the preprocessor (same as used in training)
categorical_features = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 
                       'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
numerical_features = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Create a pipeline with preprocessor and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Loading Screen (simulated with dcc.Interval)
loading_screen = html.Div(
    id="loading-screen",
    children=[
        html.Div("BrightPath Academy", className="loading-text"),
        html.Div(
            html.Div(id="progressBar", className="progress"),
            className="progress-bar"
        )
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100vw",
        "height": "100vh",
        "background": "#1e3a8a",
        "color": "white",
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "flexDirection": "column",
        "zIndex": 9999,
        "fontFamily": "'Inter', sans-serif",
        "transition": "opacity 0.5s ease"
    }
)

# Sidebar layout
sidebar = dbc.Col(
    [
        html.Br(),
        html.Br(),
        html.H3("BrightPath Academy", className="text-center text-white mb-4"),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Center([
            dbc.Nav(
                [
                    dbc.NavLink([html.I(className="bi bi-speedometer2 me-2"), "Dashboard"], href="#dashboard", className="text-white"),
                    dbc.NavLink([html.I(className="bi bi-calculator me-2"), "Predict Math Score"], href="#predict", className="text-white"),
                    dbc.NavLink([html.I(className="bi bi-exclamation-triangle me-2"), "At-Risk Students"], href="#risk", className="text-white"),
                    dbc.NavLink([html.I(className="bi bi-bar-chart-line me-2"), "Performance Trends"], href="#trends", className="text-white"),
                ],
                vertical=True,
                pills=True,
                className="mt-4"
            ),
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Center(
            html.Img(src="/assets/logo.png", style={"width": "40%", "borderRadius": "12px"})
        ),
    ],
    width=2,
    style={"backgroundColor": "#1e3a8a", "color": "white", "padding": "20px", "height": "100vh", "position": "fixed"}
)

# Main content layout
main_content = dbc.Col(
    [
        # Dashboard Section
        html.Div(
            [
                html.Div(
                    html.H1("Student Performance Dashboard", className="mb-4"),
                    className="fixed-box",
                    style={
                        "position": "fixed",
                        "top": "20px",
                        "width": "65%",
                        "height": "10%",
                        "left": "60%",
                        "transform": "translateX(-50%)",
                        "backgroundColor": "white",
                        "padding": "20px 40px",
                        "borderRadius": "15px",
                        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                        "zIndex": 10,
                        "textAlign": "center"
                    }
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(dbc.Card([html.H5("Average GPA"), html.P(f"{avg_gpa}", className="fs-3")], body=True), width=3),
                        dbc.Col(dbc.Card([html.H5("At-Risk Students"), html.P(f"{at_risk_count}", className="fs-3")], body=True), width=3),
                        dbc.Col(dbc.Card([html.H5("Average Absences"), html.P(f"{avg_absences}", className="fs-3")], body=True), width=3),
                        dbc.Col(dbc.Card([html.H5("Tutoring Participation"), html.P(f"{tutoring_participation}%", className="fs-3")], body=True), width=3),
                    ],
                    className="mb-4"
                ),
            ],
            id="dashboard"
        ),
        html.Center(
            html.Img(src="/assets/logo.png", style={"width": "40%", "borderRadius": "12px"})
        ),

        # Prediction Form
        html.Div(
            [
                html.H2("Predict Math Score", className="mb-4"),
                dbc.Form(
                    [
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Age"),
                                dcc.Input(type="number", min=15, max=18, placeholder="Enter Age", id="age", className="form-control", required=True),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Gender"),
                                dcc.Dropdown(
                                    options=[{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}],
                                    placeholder="Select Gender", id="gender", className="form-select", required=True
                                ),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Ethnicity"),
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Caucasian', 'value': 0},
                                        {'label': 'African American', 'value': 1},
                                        {'label': 'Asian', 'value': 2},
                                        {'label': 'Other', 'value': 3}
                                    ],
                                    placeholder="Select Ethnicity", id="ethnicity", className="form-select", required=True
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Parental Education"),
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'None', 'value': 0},
                                        {'label': 'High School', 'value': 1},
                                        {'label': 'Some College', 'value': 2},
                                        {'label': 'Bachelor\'s', 'value': 3},
                                        {'label': 'Higher', 'value': 4}
                                    ],
                                    placeholder="Select Education", id="parental_education", className="form-select", required=True
                                ),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Study Time (Weekly Hours)"),
                                dcc.Input(type="number", min=0, max=20, placeholder="Enter hours (0-20)", id="study_time", className="form-control", required=True),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Absences"),
                                dcc.Input(type="number", min=0, max=30, placeholder="Enter absences", id="absences", className="form-control", required=True),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Tutoring Status"),
                                dcc.Dropdown(
                                    options=[{'label': 'Yes', 'value': 1}, {'label': 'None', 'value': 0}],
                                    placeholder="Select Tutoring Status", id="tutoring", className="form-select", required=True
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Parental Involvement"),
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'None', 'value': 0},
                                        {'label': 'Low', 'value': 1},
                                        {'label': 'Moderate', 'value': 2},
                                        {'label': 'High', 'value': 3},
                                        {'label': 'Very High', 'value': 4}
                                    ],
                                    placeholder="Select Parental Involvement", id="parental_support", className="form-select", required=True
                                ),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("GPA"),
                                dcc.Input(type="number", min=2.0, max=4.0, step=0.1, placeholder="Enter GPA", id="gpa", className="form-control", required=True),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Extracurricular Activities"),
                                dbc.Row([
                                    # Left Column
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Extracurricular Activities", className="mb-3"),
                                            html.Label("Extracurricular:", className="form-label d-block mb-1"),
                                            dcc.RadioItems(
                                                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                                                id="extracurricular",
                                                className="d-flex gap-3",
                                                labelClassName="form-check-label",
                                                inputClassName="form-check-input",
                                                required=True
                                            ),
                                            html.Label("Sports:", className="form-label d-block mb-1 mt-2"),
                                            dcc.RadioItems(
                                                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                                                id="sports",
                                                className="d-flex gap-3",
                                                labelClassName="form-check-label",
                                                inputClassName="form-check-input",
                                                required=True
                                            ),
                                        ], className="p-3 border rounded bg-light shadow-sm w-100")
                                    ], width=6),
                                    # Right Column
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Extracurricular Activities", className="mb-3"),
                                            html.Label("Music:", className="form-label d-block mb-1"),
                                            dcc.RadioItems(
                                                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                                                id="music",
                                                className="d-flex gap-3",
                                                labelClassName="form-check-label",
                                                inputClassName="form-check-input",
                                                required=True
                                            ),
                                            html.Label("Volunteering:", className="form-label d-block mb-1 mt-2"),
                                            dcc.RadioItems(
                                                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                                                id="volunteering",
                                                className="d-flex gap-3",
                                                labelClassName="form-check-label",
                                                inputClassName="form-check-input",
                                                required=True
                                            ),
                                        ], className="p-3 border rounded bg-light shadow-sm w-100")
                                    ], width=6),
                                ]),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Button("Predict Math Score", id="predict-button", color="primary", className="w-100"),
                    ]
                ),
                html.Div(id="prediction-output", className="mt-3"),
            ],
            id="predict",
            className="card p-4 mb-5"
        ),

        # At-Risk Students
        html.Div(
            [
                html.H2("At-Risk Students", className="mb-4"),
                dbc.Label("Filter by GPA"),
                dcc.Dropdown(
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Below 2.0 (F)', 'value': 'below2'},
                        {'label': 'Below 2.5 (D or F)', 'value': 'below2.5'}
                    ],
                    value='all',
                    id='gpa-filter',
                    className="form-select mb-3"
                ),
                dash_table.DataTable(
                    id='risk-table',
                    columns=[
                        {'name': 'Student ID', 'id': 'StudentID'},
                        {'name': 'GPA', 'id': 'GPA'},
                        {'name': 'Grade Class', 'id': 'GradeClass'},
                        {'name': 'Absences', 'id': 'Absences'},
                        {'name': 'Actions', 'id': 'Actions'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    page_size=10,
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Actions'},
                            'textAlign': 'center'
                        }
                    ]
                ),
            ],
            id="risk",
            className="card p-4 mb-5"
        ),

        # Performance Trends
        html.Div(
            [
                html.H2("Performance Trends", className="mb-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="gpa-vs-study-time", style={"maxHeight": "300px"}), width=6),
                    dbc.Col(dcc.Graph(id="grade-distribution", style={"maxHeight": "300px"}), width=6),
                ]),
            ],
            id="trends",
            className="card p-4"
        ),
    ],
    width=10,
    style={"marginLeft": "270px", "padding": "20px"}
)

# App layout
app.layout = dbc.Container(
    [
        loading_screen,
        dcc.Interval(id="loading-interval", interval=20, n_intervals=0, max_intervals=100),
        dbc.Row([sidebar, main_content]),
        # Custom CSS to match dashboard.html
        html.Style("""
            body {
                background-color: #f4f7fc;
                font-family: 'Inter', sans-serif;
            }
            .card {
                border: none;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .form-control, .form-select {
                border-radius: 8px;
                padding: 12px;
            }
            .btn-primary {
                background-color: #3b82f6;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: 600;
            }
            .btn-primary:hover {
                background-color: #1e40af;
            }
            .loading-text {
                font-size: 2rem;
                margin-bottom: 20px;
                font-weight: bold;
            }
            .progress-bar {
                width: 300px;
                height: 20px;
                background-color: #ffffff33;
                border-radius: 10px;
                overflow: hidden;
            }
            .progress {
                height: 100%;
                width: 0;
                background-color: #facc15;
                transition: width 0.1s ease-in-out;
            }
            .recommendation {
                background-color: #fef3c7;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .fixed-box h1 {
                margin: 0;
                font-size: 2em;
            }
        """)
    ],
    fluid=True
)

# Callback for loading screen
@app.callback(
    [Output("progressBar", "style"), Output("loading-screen", "style")],
    Input("loading-interval", "n_intervals")
)
def update_loading_screen(n_intervals):
    progress = min(n_intervals, 100)
    progress_style = {"width": f"{progress}%", "height": "100%", "backgroundColor": "#facc15"}
    loading_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100vw",
        "height": "100vh",
        "background": "#1e3a8a",
        "color": "white",
        "display": "flex" if progress < 100 else "none",
        "justifyContent": "center",
        "alignItems": "center",
        "flexDirection": "column",
        "zIndex": 9999,
        "fontFamily": "'Inter', sans-serif",
        "transition": "opacity 0.5s ease",
        "opacity": 1 if progress < 100 else 0
    }
    return progress_style, loading_style

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        State('age', 'value'),
        State('gender', 'value'),
        State('ethnicity', 'value'),
        State('parental_education', 'value'),
        State('study_time', 'value'),
        State('absences', 'value'),
        State('tutoring', 'value'),
        State('parental_support', 'value'),
        State('gpa', 'value'),
        State('extracurricular', 'value'),
        State('sports', 'value'),
        State('music', 'value'),
        State('volunteering', 'value')
    ]
)
def predict_score(n_clicks, age, gender, ethnicity, parental_education, study_time,
                  absences, tutoring, parental_support, gpa, extracurricular, sports, music, volunteering):
    if n_clicks is None:
        return ""
    
    # Validate inputs
    inputs = [age, gender, ethnicity, parental_education, study_time, absences, tutoring, 
              parental_support, gpa, extracurricular, sports, music, volunteering]
    if None in inputs:
        return dbc.Alert("Error: Please fill out all fields.", color="danger")

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Ethnicity': [ethnicity],
        'ParentalEducation': [parental_education],
        'StudyTimeWeekly': [study_time],
        'Absences': [absences],
        'Tutoring': [tutoring],
        'ParentalSupport': [parental_support],
        'GPA': [gpa],
        'Extracurricular': [extracurricular],
        'Sports': [sports],
        'Music': [music],
        'Volunteering': [volunteering]
    })

    try:
        # Predict using the pipeline
        prediction = pipeline.predict(input_data)[0]
        # Assuming the model predicts GradeClass (0=A, 1=B, 2=C, 3=D, 4=F)
        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        predicted_grade = grade_map.get(prediction, 'Unknown')
        
        # Convert GradeClass to approximate math score (simplified mapping)
        score_map = {0: 90, 1: 80, 2: 70, 3: 60, 4: 50}
        predicted_score = score_map.get(prediction, 50)

        output = html.Div(
            [
                html.H4("Prediction Result"),
                html.P(f"Predicted Math Score: {predicted_score} (Grade: {predicted_grade})"),
                html.P([
                    html.Strong("Recommendation: "),
                    "Consider enrolling in a tutoring program and increasing weekly study time."
                ]) if predicted_score < 60 else ""
            ],
            className="recommendation"
        )
        return output
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

# Callback for at-risk students table
@app.callback(
    Output('risk-table', 'data'),
    Input('gpa-filter', 'value')
)
def update_risk_table(filter_value):
    if data.empty:
        return []
    
    filtered_data = data.copy()
    if filter_value == 'below2':
        filtered_data = filtered_data[filtered_data['GPA'] < 2.0]
    elif filter_value == 'below2.5':
        filtered_data = filtered_data[filtered_data['GPA'] < 2.5]
    
    # Map GradeClass to letter grades
    filtered_data['GradeClass'] = filtered_data['GradeClass'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'})
    filtered_data['GPA'] = filtered_data['GPA'].round(2)
    filtered_data['Actions'] = 'Suggest Tutoring'
    
    return filtered_data[['StudentID', 'GPA', 'GradeClass', 'Absences', 'Actions']].to_dict('records')

# Callback for GPA vs Study Time chart
@app.callback(
    Output('gpa-vs-study-time', 'figure'),
    Input('gpa-vs-study-time', 'id')
)
def update_gpa_study_time_chart(_):
    if data.empty:
        return go.Figure()
    
    fig = px.scatter(
        data,
        x='StudyTimeWeekly',
        y='GPA',
        title='GPA vs Study Time',
        labels={'StudyTimeWeekly': 'Study Time (Hours/Week)', 'GPA': 'GPA'},
        color_discrete_sequence=['#3b82f6']
    )
    fig.update_layout(
        height=300
    )
    return fig

# Callback for Grade Distribution chart
@app.callback(
    Output('grade-distribution', 'figure'),
    Input('grade-distribution', 'id')
)
def update_grade_distribution_chart(_):
    if data.empty:
        return go.Figure()
    
    grade_counts = data['GradeClass'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}).value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=grade_counts.index,
            y=grade_counts.values,
            marker_color='#3b82f6'
        )
    ])
    fig.update_layout(
        title='Grade Distribution',
        xaxis_title='Grade',
        yaxis_title='Number of Students',
        yaxis=dict(tick0=0),
        height=300
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)