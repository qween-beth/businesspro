from flask import Flask, send_file, make_response, render_template, request, flash, jsonify, session, redirect, url_for, session, has_request_context
from data_analyzer import AIDataAnalyzer
from io import StringIO
from business_intelligence import BusinessAnalyzer
import pandas as pd
import numpy as np
import tempfile
import random
from datetime import datetime, timedelta
import json
import plotly
import plotly.express as px
import plotly.utils
from plotly.graph_objs import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import os
import io
import functools
from typing import Tuple, Dict, Any, Union, Optional, List
from json import JSONEncoder
import uuid

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.json_encoder = CustomJSONEncoder
app.secret_key = os.urandom(24)
data_analyzer = AIDataAnalyzer()
business_intelligence = BusinessAnalyzer()

# App configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.secret_key = os.urandom(24)

# Add this code after your imports but before the Flask app initialization

class FileHandler:
    """Handle file operations with proper cleanup and session management."""
    
    def __init__(self, app):
        self.app = app
        self.temp_files = {}
        
    def save_temp_file(self, df: pd.DataFrame, session_key: str) -> str:
        """Save DataFrame to temporary file and track it."""
        # Clean up old file if it exists
        self.cleanup_temp_file(session_key)
        
        # Create new temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        # Track the file
        self.temp_files[session_key] = temp_file.name
        return temp_file.name
    
    def cleanup_temp_file(self, session_key: str) -> None:
        """Clean up temporary file for session."""
        if session_key in self.temp_files:
            try:
                os.unlink(self.temp_files[session_key])
                del self.temp_files[session_key]
            except Exception as e:
                print(f"Error cleaning up file for session {session_key}: {e}")

    
    def cleanup_all(self) -> None:
        """Clean up all temporary files."""
        for session_id in list(self.temp_files.keys()):
            self.cleanup_temp_file(session_id)

def get_session_id():
    """Generate or retrieve a unique session ID."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

class AnalysisError(Exception):
    """Custom exception for analysis errors."""
    pass

def handle_analysis_error(func):
    """Decorator for handling analysis errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'The uploaded file is empty'}), 400
        except pd.errors.ParserError:
            return jsonify({'error': 'Unable to parse the CSV file. Please check the format'}), 400
        except MemoryError:
            return jsonify({'error': 'The file is too large to process'}), 413
        except AnalysisError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred'}), 500
    return wrapper

# Initialize handlers
file_handler = FileHandler(app)


class DateHandler:
    """Handle date parsing and validation for DataFrames."""
    
    DEFAULT_FORMATS = [
        '%Y-%m-%d',          # 2023-12-31
        '%m/%d/%Y',          # 12/31/2023
        '%d/%m/%Y',          # 31/12/2023
        '%Y/%m/%d',          # 2023/12/31
        '%d-%m-%Y',          # 31-12-2023
        '%m-%d-%Y',          # 12-31-2023
        '%Y%m%d',            # 20231231
        '%d.%m.%Y',          # 31.12.2023
        '%Y.%m.%d',          # 2023.12.31
        '%d %b %Y',          # 31 Dec 2023
        '%d %B %Y',          # 31 December 2023
        '%Y-%m-%d %H:%M:%S', # 2023-12-31 23:59:59
        '%Y-%m-%d %H:%M',    # 2023-12-31 23:59
        '%d/%m/%Y %H:%M',    # 31/12/2023 23:59
        '%m/%d/%Y %H:%M'     # 12/31/2023 23:59
    ]


    @staticmethod
    def is_date_column(series: pd.Series) -> bool:
        """
        Check if a series likely contains dates.
        
        Args:
            series (pd.Series): The series to check
            
        Returns:
            bool: True if the series likely contains dates
        """
        if series.dtype == 'datetime64[ns]':
            return True
            
        if series.dtype != 'object':
            return False
            
        # Sample the series for checking
        sample_size = min(1000, len(series))
        sample = series.dropna().sample(n=sample_size) if len(series) > sample_size else series.dropna()
        
        if len(sample) == 0:
            return False
            
        # Try parsing a sample with our default formats
        for date_format in DateHandler.DEFAULT_FORMATS:
            try:
                pd.to_datetime(sample.iloc[0], format=date_format)
                return True
            except (ValueError, TypeError):
                continue
                
        return False

    @staticmethod
    def detect_date_format(series: pd.Series) -> Optional[str]:
        """
        Detect the date format in a series.
        
        Args:
            series (pd.Series): The series containing dates
            
        Returns:
            Optional[str]: The detected date format, or None if no format is detected
        """
        sample = series.dropna().iloc[0] if not series.empty else None
        if not sample:
            return None
            
        for date_format in DateHandler.DEFAULT_FORMATS:
            try:
                datetime.strptime(str(sample), date_format)
                # Verify format works for the whole series
                pd.to_datetime(series.dropna().head(), format=date_format)
                return date_format
            except (ValueError, TypeError):
                continue
                
        return None
    
    @staticmethod
    def verify_date_parsing(df: pd.DataFrame, parsed_df: pd.DataFrame) -> bool:
        """Verify date parsing succeeded"""
        date_cols = parsed_df.select_dtypes(include=['datetime64']).columns
        return len(date_cols) > 0

    @staticmethod
    def parse_dates(df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Parse date columns in the DataFrame using detected or specified formats.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_columns (Optional[List[str]]): List of column names to parse as dates
            
        Returns:
            pd.DataFrame: DataFrame with parsed date columns
        """
        df = df.copy()
        
        # If no specific columns are provided, check all object columns
        columns_to_check = date_columns if date_columns else df.select_dtypes(include=['object']).columns
        
        for col in columns_to_check:
            if col not in df.columns:
                continue
                
            if DateHandler.is_date_column(df[col]):
                detected_format = DateHandler.detect_date_format(df[col])
                if detected_format:
                    try:
                        df[col] = pd.to_datetime(df[col], format=detected_format)
                    except (ValueError, TypeError):
                        # If format detection failed, try with a flexible parser
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except Exception:
                            pass
        
        return df



class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle NaN values
        if pd.isna(obj):
            return None

        # Handle NumPy types
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle Plotly figures
        if isinstance(obj, Figure):
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(obj))

        # Fallback to the parent class's default method
        return super().default(obj)


    
def handle_file_upload(file) -> pd.DataFrame:
    """
    Handle file upload and initial data processing.
    
    Args:
        file: The uploaded file object
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Process dates using the DateHandler
    df = DateHandler.parse_dates(df)
    
    return df

def cleanup_temp_file(temp_file_path):
    """Helper function to clean up a temporary file."""
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")

@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up temporary files when the session ends."""
    if has_request_context() and 'session_id' in session:
        file_handler.cleanup_temp_file(session['session_id'])

# @atexit.register
# def cleanup_all():
#     """Clean up all temporary files when the application shuts down."""
#     file_handler.cleanup_all()

def generate_appropriate_plot(df: pd.DataFrame, question: str) -> Union[str, None]:
    """
    Generate an appropriate plot based on the question and data characteristics.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the data to plot
        question (str): User question/prompt to determine plot type
    
    Returns:
        Union[str, None]: JSON string of the plotly figure or None if no suitable plot can be generated
    """
    try:
        # Common layout settings
        layout_settings = dict(
            height=500,  # Taller default height
            margin=dict(t=50, r=20, b=40, l=60),
            font=dict(size=12),
            title_font=dict(size=16),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template="plotly_white"  # Clean, professional template
        )

        # Time series plot
        if any(word in question.lower() for word in ['trend', 'time', 'over time']):
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.line(df, x=date_cols[0], y=numeric_cols[0],
                                title=f'Trend of {numeric_cols[0]} Over Time')
                    
                    # Enhance line appearance
                    fig.update_traces(
                        line=dict(width=2),
                        mode='lines+markers',
                        marker=dict(
                            size=6,
                            line=dict(width=1, color='DarkSlateGrey')
                        )
                    )
                    
                    fig.update_layout(
                        **layout_settings,
                        xaxis=dict(
                            title_standoff=15,
                            tickfont=dict(size=12),
                            title_font=dict(size=14),
                            gridcolor='lightgray'
                        ),
                        yaxis=dict(
                            title_standoff=15,
                            tickfont=dict(size=12),
                            title_font=dict(size=14),
                            gridcolor='lightgray'
                        )
                    )
                    
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Distribution plot
        if any(word in question.lower() for word in ['distribution', 'spread', 'histogram']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0],
                                 title=f'Distribution of {numeric_cols[0]}',
                                 marginal='box',  # Add box plot on top
                                 histnorm='probability density',  # Normalize histogram
                                 color_discrete_sequence=['rgb(67, 147, 195)'])
                
                fig.update_layout(
                    **layout_settings,
                    bargap=0.1,  # Add gap between bars
                    xaxis=dict(
                        title_standoff=15,
                        tickfont=dict(size=12),
                        title_font=dict(size=14)
                    ),
                    yaxis=dict(
                        title="Density",
                        title_standoff=15,
                        tickfont=dict(size=12),
                        title_font=dict(size=14)
                    )
                )
                
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Correlation/Scatter plot
        if any(word in question.lower() for word in ['correlation', 'relationship', 'scatter']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               title=f'Relationship between {numeric_cols[0]} and {numeric_cols[1]}',
                               trendline='ols',  # Add trend line
                               trendline_color_override='red')
                
                fig.update_traces(
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        line=dict(width=1, color='DarkSlateGrey')
                    )
                )
                
                fig.update_layout(
                    **layout_settings,
                    xaxis=dict(
                        title_standoff=15,
                        tickfont=dict(size=12),
                        title_font=dict(size=14),
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        title_standoff=15,
                        tickfont=dict(size=12),
                        title_font=dict(size=14),
                        gridcolor='lightgray'
                    )
                )
                
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Bar plot for categorical comparisons
        if any(word in question.lower() for word in ['compare', 'comparison', 'differences']):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                           title=f'{numeric_cols[0]} by {categorical_cols[0]}',
                           color_discrete_sequence=['rgb(67, 147, 195)'])
                
                fig.update_layout(
                    **layout_settings,
                    xaxis=dict(
                        title_standoff=15,
                        tickfont=dict(size=12),
                        title_font=dict(size=14),
                        categoryorder='total descending'  # Sort bars by value
                    ),
                    yaxis=dict(
                        title_standoff=15,
                        tickfont=dict(size=12),
                        title_font=dict(size=14),
                        gridcolor='lightgray'
                    )
                )
                
                fig.update_traces(
                    marker_line_color='DarkSlateGrey',
                    marker_line_width=1
                )
                
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return None

    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        return None
    
def handle_ajax_request() -> Dict[str, Any]:
    """Handle AJAX requests for real-time analysis."""
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        if 'temp_file_path' not in session:
            return jsonify({'error': 'Please upload a file first'}), 400
            
        # Load the data
        df = pd.read_csv(session['temp_file_path'])
        df = DateHandler.parse_dates(df)  # Use the date parser we created earlier
        
        # Get AI analysis
        context = data_analyzer.prepare_data_context(df)
        answer = data_analyzer.ask_groq(question, context)
        
        # Check if visualization is needed
        plot_data = None
        if any(keyword in question.lower() for keyword in 
               ['trend', 'plot', 'graph', 'visualize', 'show', 'distribution']):
            try:
                plot_data = generate_appropriate_plot(df, question)
            except Exception as e:
                print(f"Plot generation error: {str(e)}")
        
        return jsonify({
            'success': True,
            'answer': answer,
            'plot': plot_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_summary_stats(df):
    """Generate summary statistics for numerical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    return summary.to_dict()

def detect_outliers(df):
    """Detect outliers using IQR method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = {
            'count': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers

def generate_correlation_matrix(df):
    """Generate correlation matrix for numerical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        return corr_matrix.to_dict()
    return {}

def generate_plots(df):
    """Generate various plots based on data types."""
    plots = []
    
    # Distribution plots for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        plots.append({
            'type': 'plot',
            'data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        })
    
    # Time series plots if date columns exist
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        numeric_col = numeric_cols[0]
        fig = px.line(df, x=date_col, y=numeric_col, 
                     title=f'Time Series: {numeric_col} over {date_col}')
        plots.append({
            'type': 'plot',
            'data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        })
    
    return plots

def analyze_question(df, question):
    """Analyze data based on user question."""
    question = question.lower()
    response = {'answer': '', 'visualizations': []}
    
    # Extract column names from the question
    def extract_columns(question, df):
        cols = [col.lower() for col in df.columns]
        return [col for col in cols if col in question]
    
    extracted_cols = extract_columns(question, df)
    
    # Time series analysis
    if any(word in question for word in ['trend', 'over time', 'time series']):
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            numeric_col = extracted_cols[0] if extracted_cols else numeric_cols[0]
            fig = px.line(df, x=date_col, y=numeric_col,
                         title=f'Trend Analysis: {numeric_col} over time')
            response['visualizations'].append({
                'type': 'plot',
                'data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            })
            response['answer'] = f"Showing trend analysis for {numeric_col} over time."
        else:
            response['answer'] = "Unable to perform trend analysis. Ensure your dataset contains date and numeric columns."
    
    # Distribution analysis
    elif any(word in question for word in ['distribution', 'spread', 'histogram']):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = extracted_cols[0] if extracted_cols else numeric_cols[0]
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            response['visualizations'].append({
                'type': 'plot',
                'data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            })
            response['answer'] = f"Showing distribution analysis for {col}."
        else:
            response['answer'] = "Unable to perform distribution analysis. Ensure your dataset contains numeric columns."
    
    # Correlation analysis
    elif any(word in question for word in ['correlation', 'relationship', 'compare']):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1, col2 = extracted_cols[:2] if len(extracted_cols) >= 2 else (numeric_cols[0], numeric_cols[1])
            fig = px.scatter(df, x=col1, y=col2,
                           title=f'Correlation: {col1} vs {col2}')
            response['visualizations'].append({
                'type': 'plot',
                'data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            })
            response['answer'] = f"Showing correlation between {col1} and {col2}."
        else:
            response['answer'] = "Unable to perform correlation analysis. Ensure your dataset contains at least two numeric columns."
    
    # Summary statistics
    elif any(word in question for word in ['summary', 'statistics', 'stats']):
        try:
            summary = generate_summary_stats(df)
            response['answer'] = f"Here are the summary statistics:\n{json.dumps(summary, indent=2)}"
        except Exception as e:
            response['answer'] = f"Error generating summary statistics: {str(e)}"
    
    else:
        response['answer'] = "I'm not sure how to analyze that. Try asking about trends, distributions, correlations, or summary statistics."
    
    return response


# Add this to other routes that return DataFrame-based JSON
def prepare_dataframe_for_json(df):
    """Helper function to prepare DataFrame for JSON serialization"""
    return df.replace({np.nan: None}).to_dict('records')


# Register the custom JSON encoder with Flask
app.json_encoder = CustomJSONEncoder

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
@handle_analysis_error
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400

    # Get or create session ID
    session_id = get_session_id()

    # Process the file using chunked reading for large files
    chunks = []
    try:
        for chunk in pd.read_csv(file, chunksize=10000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    except pd.errors.EmptyDataError:
        raise AnalysisError('The uploaded file is empty')
    except Exception as e:
        raise AnalysisError(f'Error reading file: {str(e)}')

    # Process dates
    df = DateHandler.parse_dates(df)
    
    # Save to temp file using FileHandler with session ID
    temp_file_path = file_handler.save_temp_file(df, session_id)
    session['temp_file_path'] = temp_file_path

    # Generate initial analysis
    summary_stats = generate_summary_stats(df)
    outliers = detect_outliers(df)
    plots = generate_plots(df)
    
    # Handle NaN values in the DataFrame before converting to dict
    df_dict = df.head(100).replace({np.nan: None}).to_dict('records')
    
    response_data = {
        'success': True,
        'data': df_dict,
        'columns': df.columns.tolist(),
        'preview': df.head().to_html(classes='table table-striped'),
        'stats': {
            'rows': len(df),
            'columns': len(df.columns),
            'size': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        },
        'summary_stats': {
            k: {col: (None if pd.isna(v) else v) for col, v in stats.items()} 
            for k, stats in summary_stats.items()
        },
        'outliers': {
            k: {
                'count': v['count'],
                'lower_bound': None if pd.isna(v['lower_bound']) else v['lower_bound'],
                'upper_bound': None if pd.isna(v['upper_bound']) else v['upper_bound']
            }
            for k, v in outliers.items()
        },
        'visualizations': plots
    }
    
    return jsonify(response_data)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'temp_file_path' not in session:
            return jsonify({'error': 'No data uploaded'}), 400
            
        df = pd.read_csv(session['temp_file_path'])
        df = DateHandler.parse_dates(df)
        
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        response = analyze_question(df, question)
        
        # Ensure any DataFrame data in response is properly converted
        if 'data' in response:
            response['data'] = prepare_dataframe_for_json(response['data'])
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Analysis Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/initial-analysis', methods=['POST'])
def initial_analysis():
    try:
        if 'temp_file_path' not in session:
            return jsonify({'error': 'No data uploaded'}), 400
            
        df = pd.read_csv(session['temp_file_path'])
        df = DateHandler.parse_dates(df)
        
        # Generate initial visualizations
        plots = generate_plots(df)
        
        return jsonify({
            'visualizations': plots
        })
        
    except Exception as e:
        print(f"Initial Analysis Error: {e}")
        return jsonify({'error': str(e)}), 500




@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Enhanced API endpoint for asking questions about the data."""
    try:
        if 'temp_file_path' not in session:
            return jsonify({'error': 'No data uploaded yet'}), 400

        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Load the data
        df = pd.read_csv(session['temp_file_path'])
        df = DateHandler.parse_dates(df)  # Ensure dates are parsed

        # Prepare context and get answer
        context = data_analyzer.prepare_data_context(df)
        answer = data_analyzer.ask_groq(question, context)

        # Generate a visualization if appropriate
        plot_data = None
        if any(keyword in question.lower() for keyword in ['trend', 'plot', 'graph', 'visualize', 'show']):
            try:
                # Simple time series plot for numeric columns
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fig = px.line(df, x=date_cols[0], y=numeric_cols[0], 
                                    title=f'{numeric_cols[0]} Over Time')
                        plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            except Exception as e:
                print(f"Error generating plot: {str(e)}")

        return jsonify({
            'answer': answer,
            'plot': plot_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



def generate_sample_data(num_rows=100):
    """Generate sample business data for testing."""
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(num_rows)]
    
    # Generate sample data
    data = {
        'Date': dates,
        'Sales': [random.uniform(5000, 15000) for _ in range(num_rows)],
        'Revenue': [random.uniform(50000, 150000) for _ in range(num_rows)],
        'Customers': [random.randint(100, 500) for _ in range(num_rows)],
        'Region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(num_rows)],
        'Product_Category': [random.choice(['Electronics', 'Clothing', 'Food', 'Books']) for _ in range(num_rows)],
        'Customer_Satisfaction': [random.uniform(3.5, 5.0) for _ in range(num_rows)],
        'Marketing_Spend': [random.uniform(1000, 5000) for _ in range(num_rows)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some trends and patterns
    df['Sales'] = df['Sales'] + np.sin(np.arange(len(df))) * 1000  # Add seasonality
    df['Revenue'] = df['Sales'] * random.uniform(9.5, 10.5)  # Revenue correlated with sales
    
    # Add some missing values
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, 'Customer_Satisfaction'] = np.nan
    
    return df


@app.route('/download_sample')
def download_sample():
    """Generate and download enhanced sample CSV file."""
    try:
        df = generate_sample_data()
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        output = make_response(buffer.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=sample_business_data.csv"
        output.headers["Content-type"] = "text/csv"
        
        return output
    
    except Exception as e:
        flash(f'Error generating sample data: {str(e)}')
        return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(debug=True)