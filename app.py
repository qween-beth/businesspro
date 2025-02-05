from flask import Flask, send_file, make_response, render_template, request, flash, jsonify, session, redirect, url_for, session, has_request_context
from data_analyzer import AIDataAnalyzer
from io import StringIO
from business_intelligence import BusinessAnalyzer
from perform_initial_analysis import perform_initial_analysis 
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
import uuid

app = Flask(__name__, static_folder='static', static_url_path='/static')

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)



app.secret_key = os.urandom(24)
data_analyzer = AIDataAnalyzer()
business_intelligence = BusinessAnalyzer()

app.json_encoder = CustomJSONEncoder

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

def extract_columns(question, df):
    """Extract relevant column names from the question."""
    cols = [col.lower() for col in df.columns]
    return [col for col in cols if col in question.lower()]

def generate_trend_plot(df):
    """Generate time series plot if applicable."""
    date_cols = df.select_dtypes(include=['datetime64']).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if not date_cols or not numeric_cols:
        return []  # Return an empty list if no valid columns are found

    fig = px.line(df, x=date_cols[0], y=numeric_cols[0], title=f'Trend of {numeric_cols[0]} Over Time')
    return [{'type': 'plot', 'data': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))}]

def generate_distribution_plot(df):
    """Generate distribution plot if applicable."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        fig = px.histogram(df, x=numeric_cols[0],
                          title=f'Distribution of {numeric_cols[0]}',
                          marginal='box')
        return [{'type': 'plot', 'data': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))}]
    return []

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
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    
    return outliers

def analyze_question(df, question):
    """Enhanced analysis function with better question handling and visualization."""
    try:
        question = question.lower()
        response = {
            'answer': '',
            'visualizations': [],
            'data': {}
        }

        # Handle correlation analysis
        if any(word in question for word in ['correlation', 'correlate', 'relationship']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                
                # Create correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation Coefficient"),
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                
                # Properly serialize the Plotly figure
                response['visualizations'].append({
                    'type': 'plot',
                    'data': json.loads(fig.to_json())
                })
                
                # Add correlation insights
                strong_correlations = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr = corr_matrix.iloc[i,j]
                        if abs(corr) > 0.5:
                            strong_correlations.append(
                                f"{numeric_cols[i]} and {numeric_cols[j]}: {corr:.2f}"
                            )
                
                response['answer'] = "Here's the correlation analysis:\n"
                if strong_correlations:
                    response['answer'] += "\nStrong correlations found:\n- " + "\n- ".join(strong_correlations)
                else:
                    response['answer'] += "\nNo strong correlations found between numeric variables."

        # Handle category comparisons
        elif any(word in question for word in ['compare', 'comparison', 'categories', 'categorical']):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # Calculate summary statistics by category
                summary = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).round(2)
                
                # Create comparison bar plot
                fig = px.bar(
                    df,
                    x=cat_col,
                    y=num_col,
                    title=f'{num_col} by {cat_col}',
                    labels={cat_col: cat_col.replace('_', ' ').title(), 
                           num_col: num_col.replace('_', ' ').title()},
                    color=cat_col
                )
                
                # Properly serialize the Plotly figure
                response['visualizations'].append({
                    'type': 'plot',
                    'data': json.loads(fig.to_json())
                })
                
                response['answer'] = f"Comparison of {num_col} across {cat_col} categories:\n\n"
                response['answer'] += "Summary statistics by category:\n"
                response['answer'] += str(summary)
                
                highest_cat = summary.nlargest(1, 'mean').index[0]
                lowest_cat = summary.nsmallest(1, 'mean').index[0]
                response['answer'] += f"\n\nKey findings:\n"
                response['answer'] += f"- Highest average {num_col}: {highest_cat}\n"
                response['answer'] += f"- Lowest average {num_col}: {lowest_cat}\n"

        # Handle trend analysis
        elif any(word in question for word in ['trend', 'trends', 'over time']):
            date_cols = df.select_dtypes(include=['datetime64']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                
                # Create trend line plot
                fig = px.line(
                    df,
                    x=date_col,
                    y=num_col,
                    title=f'Trend of {num_col} Over Time'
                )
                
                # Properly serialize the Plotly figure
                response['visualizations'].append({
                    'type': 'plot',
                    'data': json.loads(fig.to_json())
                })
                
                first_value = df[num_col].iloc[0]
                last_value = df[num_col].iloc[-1]
                percent_change = ((last_value - first_value) / first_value) * 100
                
                response['answer'] = f"Trend analysis for {num_col}:\n"
                response['answer'] += f"- Overall change: {percent_change:.1f}%\n"
                response['answer'] += f"- Starting value: {first_value:.2f}\n"
                response['answer'] += f"- Ending value: {last_value:.2f}\n"

        else:
            response['answer'] = "I can help you analyze:\n- Correlations between variables\n- Compare categories\n- Identify trends over time\nPlease specify what you'd like to analyze."

        return response

    except Exception as e:
        print(f"Error in analyze_question: {str(e)}")
        return {
            'answer': f'An error occurred while analyzing the data: {str(e)}',
            'visualizations': []
        }


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
        df = DateHandler.parse_dates(df)

        # Get AI analysis (assuming you have defined ask_groq and prepare_data_context elsewhere)
        context = data_analyzer.prepare_data_context(df)
        answer = data_analyzer.ask_groq(question, context)

        # Check if visualization is needed
        plot_data = generate_appropriate_plot(df, question)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




def prepare_dataframe_for_json(df):
    """Helper function to prepare DataFrame or dict for JSON serialization"""
    if isinstance(df, pd.DataFrame):
        return df.replace({np.nan: None}).to_dict('records')
    elif isinstance(df, dict):
        # You can choose how you want to handle dict objects here
        # For example, if you need to convert a dict to a DataFrame first:
        return pd.DataFrame([df]).replace({np.nan: None}).to_dict('records')
    else:
        raise TypeError(f"Expected a pandas DataFrame or dict, got {type(df)}")

# Initialize handlers
file_handler = FileHandler(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        print("DEBUG: Upload request received.")

        if 'file' not in request.files:
            print("DEBUG: No file part in request.")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            print("DEBUG: No file selected.")
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            print("DEBUG: Invalid file format.")
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

        print("DEBUG: File validated. Reading CSV file...")

        chunks = []
        try:
            for chunk in pd.read_csv(file, chunksize=10000, encoding='utf-8', on_bad_lines='skip'):
                chunks.append(chunk)

            if not chunks:
                print("DEBUG: CSV file is empty after reading.")
                return jsonify({'error': 'The uploaded file is empty'}), 400

            df = pd.concat(chunks, ignore_index=True)

        except pd.errors.EmptyDataError:
            print("DEBUG: Pandas detected empty CSV file.")
            return jsonify({'error': 'The uploaded file is empty'}), 400
        except pd.errors.ParserError:
            print("DEBUG: CSV file parsing error.")
            return jsonify({'error': 'Error parsing the CSV file. Ensure it is properly formatted.'}), 400
        except Exception as e:
            print(f"DEBUG: Unexpected error reading CSV file - {str(e)}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500

        print("DEBUG: CSV file successfully read. Checking for valid data.")

        # Parse dates
        df = DateHandler.parse_dates(df)
        print("DEBUG: Date parsing completed.")

        session_id = get_session_id()
        temp_file_path = file_handler.save_temp_file(df, session_id)
        session['temp_file_path'] = temp_file_path

        print(f"DEBUG: File saved to temporary location: {temp_file_path}")

        # Perform initial analysis
        initial_analysis_response = perform_initial_analysis(df)

        # Include success message
        initial_analysis_response['message'] = 'File uploaded successfully!'

        # 🔥 Fix: Convert response using `json.dumps()` and CustomJSONEncoder
        return app.response_class(
            response=json.dumps(initial_analysis_response, cls=CustomJSONEncoder),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        print(f"DEBUG: Upload Error - {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'temp_file_path' not in session:
            return jsonify({'error': 'No data uploaded'}), 400
            
        # Load the CSV and parse dates
        df = pd.read_csv(session['temp_file_path'])
        df = DateHandler.parse_dates(df)
        
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get the analysis response
        response = analyze_question(df, question)
        
        # Return the response using the CustomJSONEncoder
        return app.response_class(
            response=json.dumps(response, cls=CustomJSONEncoder),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        print(f"Analysis Error: {e}")
        return jsonify({'error': str(e)}), 500




@app.route('/api/correlation', methods=['GET'])
def get_correlation():
    """Dedicated endpoint for correlation matrix."""
    try:
        if 'temp_file_path' not in session:
            return jsonify({'error': 'No data uploaded'}), 400
            
        df = pd.read_csv(session['temp_file_path'])
        
        # Generate correlation matrix
        correlation_matrix = generate_correlation_matrix(df)
        
        if not correlation_matrix:
            return jsonify({'error': 'No correlation matrix available'}), 400
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return jsonify({
            'correlation_matrix': correlation_matrix,
            'numeric_columns': numeric_columns
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Enhanced API endpoint for asking questions about the data."""
    try:
        if 'temp_file_path' not in session:
            return jsonify({'error': 'No data uploaded yet'}), 400

        temp_file_path = session.get('temp_file_path')
        
        # Ensure the file path is valid
        if not temp_file_path or not os.path.exists(temp_file_path):
            return jsonify({'error': 'Data file not found'}), 400

        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Load the data safely
        try:
            df = pd.read_csv(temp_file_path)
            df = DateHandler.parse_dates(df)  # Ensure dates are parsed
        except Exception as e:
            return jsonify({'error': f'Error loading data: {str(e)}'}), 500

        # Analyze the question
        response = analyze_question(df, question)

        # Ensure there is a valid answer
        answer = response.get('answer', 'No answer generated.')

        # Handle visualization safely
        plot_data = None
        if 'visualizations' in response and response['visualizations']:
            try:
                plot_data = json.dumps(response['visualizations'][0]['data'], cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e:
                plot_data = None  # Ensure no crash on JSON encoding errors
                print(f"DEBUG: Plot serialization error - {e}")

        return jsonify({'answer': answer, 'plot': plot_data})

    except Exception as e:
        print(f"DEBUG: API Error - {e}")
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