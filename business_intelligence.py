import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import json
import plotly
import plotly.express as px
from typing import Dict, List, Any

load_dotenv()

class BusinessAnalyzer:
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