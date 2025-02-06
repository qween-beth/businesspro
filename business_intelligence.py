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
    def generate_comparison_plot(df, layout_settings):
        """Generate bar plot for comparisons between numerical and categorical columns."""
        # Extract categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check if there are both categorical and numerical columns available
        if not categorical_cols or not numeric_cols:
            return None  # No comparison possible if one of the sets is empty
        
        # Generate bar plot for the first categorical and numerical column
        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], 
                    title=f'{numeric_cols[0]} by {categorical_cols[0]}')
        
        # Apply layout settings
        fig.update_layout(**layout_settings)

        # Return the plot as a JSON object for rendering
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
        """Generate comprehensive summary statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        summary = {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # in MB
            },
            "missing_data": df.isnull().sum().to_dict(),
            "numeric_columns": {},
            "categorical_columns": {}
        }
        
        # Numeric column analysis
        for col in numeric_cols:
            summary["numeric_columns"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q1": float(df[col].quantile(0.25)),
                "q3": float(df[col].quantile(0.75)),
                "missing": int(df[col].isnull().sum()),
                "missing_percentage": float(df[col].isnull().mean() * 100)
            }
        
        # Categorical column analysis
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary["categorical_columns"][col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": value_counts.index[0] if not value_counts.empty else None,
                "most_common_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "missing": int(df[col].isnull().sum()),
                "missing_percentage": float(df[col].isnull().mean() * 100)
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
    def analyze_distribution(df: pd.DataFrame, column: str = None) -> Dict[str, Any]:
        """Analyze distribution of numeric columns."""
        if column and column in df.columns:
            numeric_cols = [column]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        results = {
            'answer': 'Distribution Analysis Results:\n',
            'visualizations': []
        }
        
        for col in numeric_cols:
            stats = df[col].describe()
            
            # Create histogram with box plot
            fig = px.histogram(
                df, 
                x=col,
                title=f'Distribution of {col}',
                marginal='box',
                nbins=30,
                color_discrete_sequence=['#63b9ff']
            )
            
            results['visualizations'].append({
                'type': 'plot',
                'data': json.loads(fig.to_json())
            })
            
            # Add statistics to answer
            results['answer'] += f"\n{col}:\n"
            results['answer'] += f"Mean: {stats['mean']:.2f}\n"
            results['answer'] += f"Median: {stats['50%']:.2f}\n"
            results['answer'] += f"Std Dev: {stats['std']:.2f}\n"
            results['answer'] += f"Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
        
        return results
    
    
    @staticmethod
    def analyze_trends(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends over time for numeric columns."""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            return {
                'answer': 'No date column or numeric columns found for trend analysis.',
                'visualizations': []
            }
        
        date_col = date_cols[0]
        results = {
            'answer': 'Trend Analysis Results:\n',
            'visualizations': []
        }
        
        for col in numeric_cols:
            fig = px.line(
                df,
                x=date_col,
                y=col,
                title=f'Trend Analysis: {col} over time'
            )
            
            results['visualizations'].append({
                'type': 'plot',
                'data': json.loads(fig.to_json())
            })
            
            # Calculate key metrics
            total_change = df[col].iloc[-1] - df[col].iloc[0]
            pct_change = (total_change / df[col].iloc[0]) * 100
            
            results['answer'] += f"\n{col}:\n"
            results['answer'] += f"Total change: {total_change:.2f}\n"
            results['answer'] += f"Percentage change: {pct_change:.2f}%\n"
        
        return results


    @staticmethod
    def analyze_question(df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Analyze data based on user question."""
        question = question.lower()
        response = {'answer': '', 'visualizations': []}

        extracted_cols = BusinessAnalyzer.extract_columns(question, df)

        # Distribution analysis
        if any(word in question for word in ['distribution', 'analyze distribution', 'spread', 'histogram']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = BusinessAnalyzer.generate_summary_stats(df)
                response['answer'] = "Distribution Analysis Results:\n\n"
                
                # Add distribution plots for each numeric column
                for col in numeric_cols:
                    # Create histogram with box plot
                    fig = px.histogram(
                        df, 
                        x=col,
                        title=f'Distribution of {col}',
                        marginal='box',
                        nbins=30,
                        color_discrete_sequence=['#63b9ff']
                    )
                    
                    fig.update_layout(
                        showlegend=False,
                        xaxis_title=col,
                        yaxis_title='Count',
                        height=400,
                        margin=dict(t=50, r=20, b=40, l=60)
                    )
                    
                    # Add to visualizations
                    response['visualizations'].append({
                        'type': 'plot',
                        'data': json.loads(fig.to_json())
                    })
                    
                    # Add statistics for this column
                    col_stats = summary_stats['numeric_columns'][col]
                    response['answer'] += f"\n{col}:\n"
                    response['answer'] += f"- Mean: {col_stats['mean']:.2f}\n"
                    response['answer'] += f"- Median: {col_stats['median']:.2f}\n"
                    response['answer'] += f"- Standard Deviation: {col_stats['std']:.2f}\n"
                    response['answer'] += f"- Range: {col_stats['min']:.2f} to {col_stats['max']:.2f}\n"
                    response['answer'] += f"- Growth: {col_stats['growth']:.2f}%\n"
            else:
                response['answer'] = "No numeric columns found in the dataset for distribution analysis."


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


       #Handle category comparisons
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
        elif any(word in question for word in ['correlation', 'correlate', 'relationship']):
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
