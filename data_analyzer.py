import os
from groq import Groq
import pandas as pd
import plotly.express as px
import json
from typing import Dict, Any, Tuple, List
import numpy as np
from datetime import datetime
import re

class AIDataAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.date_patterns = {
            'months': {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            },
            'years': r'\b(20\d{2})\b'
        }
        self.metric_mappings = {
            'satisfaction': ['satisfaction', 'rating', 'score', 'happy', 'satisfied'],
            'sales': ['sales', 'revenue', 'amount', 'selling', 'sold'],
            'products': ['product', 'item', 'sku', 'merchandise'],
            'customers': ['customer', 'client', 'buyer', 'user']
        }

    def analyze(self, df: pd.DataFrame, question: str) -> Tuple[str, List[Dict]]:
        # Extract metadata and context
        metadata = self._extract_metadata(df, question)
        context = self._prepare_context(df)
        
        # Filter data based on time period if specified
        filtered_df = self._filter_time_period(df, metadata['time_period'])
        
        # Generate answer and visualizations
        answer = self._get_analysis(filtered_df, question, context, metadata)
        visualizations = self._create_visualizations(filtered_df, metadata)
        
        return answer, visualizations


    def prepare_data_context(self, df: pd.DataFrame) -> str:
            # Get column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Calculate basic statistics
            stats = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()
            
            context = f"""
                Dataset Analysis:
                - Total records: {len(df)}
                - Numeric columns: {', '.join(numeric_cols)}
                - Date columns: {', '.join(date_cols)}
                - Categorical columns: {', '.join(categorical_cols)}
                
                Summary Statistics:
                {stats.to_string() if not stats.empty else 'No numeric columns'}
                
                Recent trends:
                {self._get_recent_trends(df, numeric_cols, date_cols) if date_cols else 'No time-based data available'}
            """
            return context
    
    

    def _prepare_context(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        stats = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()
        
        return f"""
            Dataset Overview:
            - Total records: {len(df)}
            - Time range: {df[date_cols[0]].min() if date_cols else 'N/A'} to {df[date_cols[0]].max() if date_cols else 'N/A'}
            - Numeric metrics: {', '.join(numeric_cols)}
            - Categories: {', '.join(categorical_cols)}
            
            Key Statistics:
            {stats.to_string() if not stats.empty else 'No numeric data available'}
            
            Recent Trends:
            {self._get_recent_trends(df, numeric_cols, date_cols) if date_cols else 'No time-based data available'}
        """

    def _get_recent_trends(self, df: pd.DataFrame, numeric_cols: list, date_cols: list) -> str:
        if not date_cols or not numeric_cols:
            return ""
        
        date_col = date_cols[0]
        df = df.sort_values(date_col)
        recent_data = df.tail(5)
        
        trends = []
        for col in numeric_cols:
            last_value = recent_data[col].iloc[-1]
            pct_change = ((last_value - recent_data[col].iloc[0]) / recent_data[col].iloc[0]) * 100
            trends.append(f"{col}: {pct_change:.1f}% change over last 5 periods")
        
        return "\n".join(trends)

    def _extract_metadata(self, df: pd.DataFrame, question: str) -> Dict:
        return {
            'time_period': self._extract_time_period(question),
            'metrics': self._identify_relevant_metrics(df, question),
            'categories': self._identify_categories(df, question),
            'analysis_type': self._determine_analysis_type(question)
        }

    def _extract_time_period(self, question: str) -> Dict:
        question = question.lower()
        period = {}
        
        # Extract month
        for month, num in self.date_patterns['months'].items():
            if month in question:
                period['month'] = num
                break
        
        # Extract year
        year_match = re.search(self.date_patterns['years'], question)
        if year_match:
            period['year'] = int(year_match.group())
        
        return period

    def _identify_relevant_metrics(self, df: pd.DataFrame, question: str) -> List[str]:
        question_tokens = set(question.lower().split())
        metrics = []
        
        # Check direct column matches
        for col in df.columns:
            col_lower = col.lower()
            if any(token in col_lower for token in question_tokens):
                metrics.append(col)
        
        # Check semantic matches
        for metric_type, terms in self.metric_mappings.items():
            if any(term in question_tokens for term in terms):
                for col in df.columns:
                    if any(term in col.lower() for term in terms):
                        metrics.append(col)
        
        return list(set(metrics))

    def _identify_categories(self, df: pd.DataFrame, question: str) -> List[str]:
        categorical_cols = df.select_dtypes(include=['object']).columns
        question_tokens = set(question.lower().split())
        
        return [col for col in categorical_cols if any(token in col.lower() for token in question_tokens)]

    def _determine_analysis_type(self, question: str) -> str:
        question = question.lower()
        if any(term in question for term in ['trend', 'over time', 'historical']):
            return 'trend'
        elif any(term in question for term in ['compare', 'versus', 'vs', 'difference']):
            return 'comparison'
        elif any(term in question for term in ['top', 'best', 'worst', 'highest', 'lowest']):
            return 'ranking'
        return 'general'

    def _filter_time_period(self, df: pd.DataFrame, period: Dict) -> pd.DataFrame:
        if not period:
            return df
            
        date_col = next((col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])), None)
        if not date_col:
            return df

        mask = pd.Series(True, index=df.index)
        if period.get('month'):
            mask &= df[date_col].dt.month == period['month']
        if period.get('year'):
            mask &= df[date_col].dt.year == period['year']
            
        return df[mask]

    def ask_groq(self, question: str, context: str) -> str:
        try:
            prompt = f"""
                Analyze this dataset:
                {context}
                
                Question: {question}
                
                Provide a clear, data-driven answer with:
                1. Key insights from the data
                2. Notable trends or patterns
                3. Specific numbers to support findings
                4. Any important correlations or relationships
                
                Keep the response concise but informative.
            """
            
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a data analyst providing clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
        
        
    def _get_analysis(self, df: pd.DataFrame, question: str, context: str, metadata: Dict) -> str:
        prompt = f"""
        Analyze this data focusing on {', '.join(metadata['metrics'])}
        Context: {context}
        Question: {question}
        Time period: {metadata['time_period']}
        Analysis type: {metadata['analysis_type']}
        
        Provide natural, conversational insights with:
        1. Direct answer to the question
        2. Specific numbers and trends
        3. Important patterns or anomalies
        4. Brief explanation of key findings
        """
        
        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a friendly data analyst providing clear, conversational insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350
        )
        return response.choices[0].message.content

    def _create_visualizations(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        visualizations = []
        
        if metadata['analysis_type'] == 'trend':
            visualizations.extend(self._create_trend_visualizations(df, metadata))
        elif metadata['analysis_type'] == 'comparison':
            visualizations.extend(self._create_comparison_visualizations(df, metadata))
        elif metadata['analysis_type'] == 'ranking':
            visualizations.extend(self._create_ranking_visualizations(df, metadata))
        else:
            visualizations.extend(self._create_general_visualizations(df, metadata))
        
        return visualizations

    def _create_trend_visualizations(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        visualizations = []
        date_col = next((col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])), None)
        
        if date_col and metadata['metrics']:
            for metric in metadata['metrics']:
                if metric in df.columns:
                    fig = px.line(df, x=date_col, y=metric, 
                                title=f'Trend of {metric}')
                    visualizations.append({
                        'type': 'plot',
                        'data': json.loads(fig.to_json())
                    })
        
        return visualizations

    def _create_comparison_visualizations(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        visualizations = []
        
        if metadata['categories'] and metadata['metrics']:
            for category in metadata['categories']:
                for metric in metadata['metrics']:
                    if metric in df.columns:
                        fig = px.box(df, x=category, y=metric,
                                   title=f'{metric} by {category}')
                        visualizations.append({
                            'type': 'plot',
                            'data': json.loads(fig.to_json())
                        })
        
        return visualizations

    def _create_ranking_visualizations(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        visualizations = []
        
        if metadata['categories'] and metadata['metrics']:
            for category in metadata['categories']:
                for metric in metadata['metrics']:
                    if metric in df.columns:
                        agg_df = df.groupby(category)[metric].sum().sort_values(ascending=False)
                        fig = px.bar(agg_df, title=f'Top {category} by {metric}')
                        visualizations.append({
                            'type': 'plot',
                            'data': json.loads(fig.to_json())
                        })
        
        return visualizations

    def _create_general_visualizations(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        visualizations = []
        
        # Create appropriate visualization based on available data
        if metadata['metrics']:
            for metric in metadata['metrics']:
                if metric in df.columns:
                    if metadata['categories']:
                        fig = px.bar(df, x=metadata['categories'][0], y=metric,
                                   title=f'{metric} by {metadata['categories'][0]}')
                    else:
                        fig = px.histogram(df, x=metric,
                                         title=f'Distribution of {metric}')
                    visualizations.append({
                        'type': 'plot',
                        'data': json.loads(fig.to_json())
                    })
        
        return visualizations