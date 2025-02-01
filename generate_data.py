
from flask import Flask, send_file, make_response
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

app = Flask(__name__)

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
    """Generate and download sample CSV file."""
    try:
        # Generate the sample data
        df = generate_sample_data()
        
        # Create a string buffer
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Create the response
        output = make_response(buffer.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=sample_business_data.csv"
        output.headers["Content-type"] = "text/csv"
        
        return output
    
    except Exception as e:
        return str(e), 500

