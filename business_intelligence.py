import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import plotly
import plotly.express as px
from typing import Dict, List, Any
import numpy as np

load_dotenv()

    
class BusinessAnalyzer:
    @staticmethod
    def generate_plots(df: pd.DataFrame) -> Dict[str, str]:
        """Generate various business-relevant visualizations."""
        plots = {}
        
        # Time series plot for numeric columns if date column exists
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                fig = px.line(df, x=date_col, y=col, title=f'{col} Over Time')
                plots[f'{col}_trend'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            plots[f'{col}_dist'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return plots
    
    @staticmethod
    def extract_columns(question: str, df: pd.DataFrame) -> List[str]:
        """Extract relevant column names from the question."""
        cols = [col.lower() for col in df.columns]
        return [col for col in cols if col in question.lower()]

    @staticmethod
    def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate key business metrics and summary statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_data": df.isnull().sum().to_dict(),
            "numeric_columns": {
                col: {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "growth": ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0] * 100) 
                        if len(df) > 1 else 0
                }
                for col in numeric_cols
            }
        }
        return summary
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, threshold: float = 1.5) -> Dict[str, List]:
        """Detect outliers in numeric columns using IQR method."""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))
            outliers[col] = df[outlier_mask][col].tolist()
        
        return outliers
    
    @staticmethod
    def analyze_question(df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Analyze data based on user question."""
        question = question.lower()
        response = {'answer': '', 'visualizations': []}

        extracted_cols = BusinessAnalyzer.extract_columns(question, df)

        # Time series analysis
        if any(word in question for word in ['trend', 'over time', 'time series']):
            date_cols = df.select_dtypes(include=['datetime64']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                date_col = date_cols[0]
                numeric_col = extracted_cols[0] if extracted_cols else numeric_cols[0]
                fig = px.line(df, x=date_col, y=numeric_col, title=f'Trend Analysis: {numeric_col} over time')
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
                fig = px.scatter(df, x=col1, y=col2, title=f'Correlation: {col1} vs {col2}')
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
                summary = BusinessAnalyzer.generate_summary_stats(df)
                response['answer'] = f"Here are the summary statistics:\n{json.dumps(summary, indent=2)}"
            except Exception as e:
                response['answer'] = f"Error generating summary statistics: {str(e)}"

        else:
            response['answer'] = "I'm not sure how to analyze that. Try asking about trends, distributions, correlations, or summary statistics."
        
        return response
