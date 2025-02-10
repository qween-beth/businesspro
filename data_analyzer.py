import os
from groq import Groq
import pandas as pd
import plotly.express as px
import json
from typing import Dict, Any, Tuple, List
import numpy as np
from datetime import datetime
import re



class EnhancedAnalyzer:
    def __init__(self):
        self.ai_analyzer = AIDataAnalyzer()
        
    def analyze(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Combine AI insights with automated visualizations."""
        try:
            # Get AI analysis and visualizations
            ai_answer, ai_visualizations = self.ai_analyzer.analyze(df, question)
            
            # Get traditional analysis
            bi_response = self.ai_analyzer.analyze_question(df, question)
            
            # Combine insights
            combined_answer = ai_answer
            if bi_response.get('answer'):
                combined_answer += f"\n\nAdditional Analysis:\n{bi_response['answer']}"
            
            # Process and combine visualizations
            plot_data = []
            
            # Helper function to process visualization data
            def process_viz(viz):
                if isinstance(viz, dict):
                    if 'type' in viz and viz['type'] == 'plot':
                        return {
                            'data': viz['data'].get('data', []),
                            'layout': viz['data'].get('layout', {})
                        }
                    elif 'data' in viz:
                        return {
                            'data': viz.get('data', []),
                            'layout': viz.get('layout', {})
                        }
                return None
            
            # Process AI visualizations
            if ai_visualizations:
                for viz in ai_visualizations:
                    processed_viz = process_viz(viz)
                    if processed_viz:
                        plot_data.append(processed_viz)
            
            # Process BI visualizations
            if bi_response.get('visualizations'):
                for viz in bi_response['visualizations']:
                    processed_viz = process_viz(viz)
                    if processed_viz:
                        plot_data.append(processed_viz)
            
            # Ensure proper layout settings for all plots
            for plot in plot_data:
                if 'layout' in plot:
                    plot['layout'].update({
                        'height': 400,
                        'margin': {'t': 30, 'r': 30, 'b': 50, 'l': 50},
                        'paper_bgcolor': '#ffffff',
                        'plot_bgcolor': '#f8f9fa',
                        'font': {
                            'family': 'Arial, sans-serif',
                            'size': 12,
                            'color': '#333333'
                        }
                    })
            
            return {
                'answer': combined_answer,
                'plots': plot_data
            }
            
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

    def generate_automatic_visualizations(self, df: pd.DataFrame, question: str) -> List[Dict]:
        """Generate relevant visualizations based on the question context."""
        plots = []
        question = question.lower()
        
        # Identify date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Time series analysis
        if any(term in question for term in ['trend', 'over time', 'historical']):
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                for num_col in numeric_cols[:2]:  # Limit to 2 metrics for clarity
                    fig = px.line(
                        df,
                        x=date_cols[0],
                        y=num_col,
                        title=f'{num_col} Over Time'
                    )
                    plots.append({
                        'data': json.loads(fig.to_json())['data'],
                        'layout': json.loads(fig.to_json())['layout']
                    })
        
        # Distribution analysis
        elif any(term in question for term in ['distribution', 'spread', 'range']):
            for col in numeric_cols[:2]:
                fig = px.histogram(
                    df,
                    x=col,
                    title=f'Distribution of {col}',
                    marginal='box'
                )
                plots.append({
                    'data': json.loads(fig.to_json())['data'],
                    'layout': json.loads(fig.to_json())['layout']
                })
        
        # Categorical comparisons
        elif any(term in question for term in ['compare', 'comparison', 'versus', 'vs']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                fig = px.box(
                    df,
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title=f'{numeric_cols[0]} by {categorical_cols[0]}'
                )
                plots.append({
                    'data': json.loads(fig.to_json())['data'],
                    'layout': json.loads(fig.to_json())['layout']
                })
        
        return plots

        
class AIDataAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.metric_mappings = {
            'satisfaction': ['satisfaction', 'rating', 'score', 'happy', 'satisfied'],
            'sales': ['sales', 'revenue', 'amount', 'selling', 'sold'],
            'products': ['product', 'item', 'sku', 'merchandise'],
            'customers': ['customer', 'client', 'buyer', 'user']
        }
        self.date_patterns = {
            'months': {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            },
            'years': r'\b(20\d{2})\b'
        }



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

    def analyze_question(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Analyze data based on user question and return answer with visualizations."""
        question = question.lower()
        metadata = self._extract_metadata(df, question)
        context = self._prepare_context(df)
        
        # Filter data if time period is specified
        filtered_df = self._filter_by_time(df, metadata['time_period'])
        
        # Determine analysis type and generate appropriate response
        if any(word in question for word in ['satisfaction', 'happy', 'satisfied']):
            return self._analyze_satisfaction(filtered_df)
        elif metadata['analysis_type'] == 'trend':
            return self._analyze_trends(filtered_df, metadata)
        elif metadata['analysis_type'] == 'comparison':
            return self._analyze_comparison(filtered_df, metadata)
        elif metadata['analysis_type'] == 'distribution':
            return self._analyze_distribution(filtered_df, metadata)
        else:
            return self._analyze_general(filtered_df, metadata)

    def _prepare_context(self, df: pd.DataFrame) -> Dict:
        """Prepare dataset context for analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        stats = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()
        
        return {
            'total_records': len(df),
            'time_range': {
                'start': df[date_cols[0]].min() if date_cols else None,
                'end': df[date_cols[0]].max() if date_cols else None
            },
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'statistics': stats.to_dict() if not stats.empty else {}
        }

    def _extract_metadata(self, df: pd.DataFrame, question: str) -> Dict:
        """Extract relevant metadata from the question and dataset."""
        return {
            'time_period': self._extract_time_period(question),
            'metrics': self._identify_metrics(df, question),
            'categories': self._identify_categories(df, question),
            'analysis_type': self._determine_analysis_type(question)
        }

    def _analyze_satisfaction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze customer satisfaction data with detailed insights."""
        satisfaction_cols = [col for col in df.columns 
                           if any(term in col.lower() 
                                 for term in self.metric_mappings['satisfaction'])]
        
        if not satisfaction_cols:
            return {
                'answer': 'No satisfaction-related columns found in the dataset.',
                'plots': []
            }

        sat_col = satisfaction_cols[0]
        insights = []
        plots = []

        # Basic statistics
        stats = df[sat_col].describe()
        insights.extend([
            f"Average satisfaction: {stats['mean']:.2f}",
            f"Standard deviation: {stats['std']:.2f}",
            f"Range: {stats['min']:.2f} to {stats['max']:.2f}"
        ])

        # Time-based analysis
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            df_sorted = df.sort_values(date_col)
            
            # Add trend plot
            fig_trend = px.line(
                df_sorted,
                x=date_col,
                y=sat_col,
                title='Satisfaction Trend Over Time'
            )
            plots.append({
                'data': fig_trend.to_dict()['data'],
                'layout': fig_trend.to_dict()['layout']
            })

            # Recent trend analysis
            if len(df_sorted) >= 6:
                recent_change = ((df_sorted[sat_col].iloc[-1] - df_sorted[sat_col].iloc[-6]) / 
                               df_sorted[sat_col].iloc[-6] * 100)
                insights.append(f"Recent trend: {recent_change:+.1f}% change over last 5 periods")

        # Distribution analysis
        fig_dist = px.histogram(
            df,
            x=sat_col,
            title='Distribution of Satisfaction Scores',
            marginal='box'
        )
        plots.append({
            'data': fig_dist.to_dict()['data'],
            'layout': fig_dist.to_dict()['layout']
        })

        # Additional distribution insights
        quartiles = df[sat_col].quantile([0.25, 0.75])
        insights.append(f"Middle 50% of scores fall between {quartiles[0.25]:.2f} and {quartiles[0.75]:.2f}")

        skewness = df[sat_col].skew()
        if abs(skewness) > 0.5:
            skew_direction = "higher" if skewness > 0 else "lower"
            insights.append(f"Scores tend to be {skew_direction} than the average")

        return {
            'answer': "Customer Satisfaction Analysis:\n\n" + "\n".join(f"• {insight}" for insight in insights),
            'plots': plots
        }

    def _analyze_trends(self, df: pd.DataFrame, metadata: Dict) -> Dict[str, Any]:
        """Analyze trends in the data."""
        insights = []
        plots = []
        
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if not date_cols or not metadata['metrics']:
            return {
                'answer': 'Unable to analyze trends. Need both date and metric columns.',
                'plots': []
            }
            
        date_col = date_cols[0]
        df_sorted = df.sort_values(date_col)
        
        for metric in metadata['metrics']:
            if metric not in df.columns:
                continue
                
            # Calculate overall trend
            total_change = ((df_sorted[metric].iloc[-1] - df_sorted[metric].iloc[0]) / 
                          df_sorted[metric].iloc[0] * 100)
            
            # Calculate recent trend
            recent_change = ((df_sorted[metric].iloc[-1] - df_sorted[metric].iloc[-6]) / 
                           df_sorted[metric].iloc[-6] * 100) if len(df_sorted) >= 6 else None
            
            insights.extend([
                f"Overall {metric} change: {total_change:+.1f}%",
                f"Recent {metric} trend: {recent_change:+.1f}% (last 5 periods)" if recent_change is not None else ""
            ])
            
            # Create trend plot
            fig = px.line(
                df_sorted,
                x=date_col,
                y=metric,
                title=f'Trend of {metric} Over Time'
            )
            plots.append({
                'data': fig.to_dict()['data'],
                'layout': fig.to_dict()['layout']
            })
            
            # Add moving average
            df_sorted[f'{metric}_MA'] = df_sorted[metric].rolling(window=3).mean()
            fig_ma = px.line(
                df_sorted,
                x=date_col,
                y=[metric, f'{metric}_MA'],
                title=f'{metric} with 3-period Moving Average'
            )
            plots.append({
                'data': fig_ma.to_dict()['data'],
                'layout': fig_ma.to_dict()['layout']
            })

        return {
            'answer': "Trend Analysis:\n\n" + "\n".join(f"• {insight}" for insight in insights if insight),
            'plots': plots
        }

    def _analyze_comparison(self, df: pd.DataFrame, metadata: Dict) -> Dict[str, Any]:
        """Analyze comparisons between categories."""
        insights = []
        plots = []
        
        if not metadata['categories'] or not metadata['metrics']:
            return {
                'answer': 'Unable to perform comparison. Need both categories and metrics.',
                'plots': []
            }
            
        for category in metadata['categories']:
            for metric in metadata['metrics']:
                if metric not in df.columns or category not in df.columns:
                    continue
                    
                # Calculate summary statistics by category
                summary = df.groupby(category)[metric].agg(['mean', 'std', 'count'])
                
                # Add insights
                top_category = summary.nlargest(1, 'mean').index[0]
                bottom_category = summary.nsmallest(1, 'mean').index[0]
                
                insights.extend([
                    f"Highest {metric}: {top_category} (avg: {summary.loc[top_category, 'mean']:.2f})",
                    f"Lowest {metric}: {bottom_category} (avg: {summary.loc[bottom_category, 'mean']:.2f})"
                ])
                
                # Create comparison plots
                fig_box = px.box(
                    df,
                    x=category,
                    y=metric,
                    title=f'Distribution of {metric} by {category}'
                )
                plots.append({
                    'data': fig_box.to_dict()['data'],
                    'layout': fig_box.to_dict()['layout']
                })
                
                fig_bar = px.bar(
                    summary.reset_index(),
                    x=category,
                    y='mean',
                    error_y='std',
                    title=f'Average {metric} by {category}'
                )
                plots.append({
                    'data': fig_bar.to_dict()['data'],
                    'layout': fig_bar.to_dict()['layout']
                })

        return {
            'answer': "Comparison Analysis:\n\n" + "\n".join(f"• {insight}" for insight in insights),
            'plots': plots
        }

    def _analyze_distribution(self, df: pd.DataFrame, metadata: Dict) -> Dict[str, Any]:
        """Analyze distribution of metrics."""
        insights = []
        plots = []
        
        for metric in metadata['metrics']:
            if metric not in df.columns:
                continue
                
            # Calculate distribution statistics
            stats = df[metric].describe()
            skewness = df[metric].skew()
            kurtosis = df[metric].kurtosis()
            
            insights.extend([
                f"{metric} summary:",
                f"• Average: {stats['mean']:.2f}",
                f"• Standard deviation: {stats['std']:.2f}",
                f"• Range: {stats['min']:.2f} to {stats['max']:.2f}",
                f"• Distribution shape: {'positively' if skewness > 0 else 'negatively'} skewed",
                f"• Peakedness: {'more' if kurtosis > 0 else 'less'} peaked than normal"
            ])
            
            # Create distribution plots
            fig_hist = px.histogram(
                df,
                x=metric,
                title=f'Distribution of {metric}',
                marginal='box'
            )
            plots.append({
                'data': fig_hist.to_dict()['data'],
                'layout': fig_hist.to_dict()['layout']
            })
            
            # Add QQ plot if sample size is sufficient
            if len(df) >= 30:
                from scipy import stats
                qq = stats.probplot(df[metric], dist="norm")
                fig_qq = px.scatter(
                    x=qq[0][0],
                    y=qq[0][1],
                    title=f'Q-Q Plot for {metric}'
                )
                fig_qq.add_scatter(
                    x=qq[0][0],
                    y=qq[0][0] * qq[1][0] + qq[1][1],
                    mode='lines'
                )
                plots.append({
                    'data': fig_qq.to_dict()['data'],
                    'layout': fig_qq.to_dict()['layout']
                })

        return {
            'answer': "Distribution Analysis:\n\n" + "\n".join(insights),
            'plots': plots
        }

    def _analyze_general(self, df: pd.DataFrame, metadata: Dict) -> Dict[str, Any]:
        """Perform general analysis when no specific type is identified."""
        insights = []
        plots = []
        
        # Basic dataset information
        insights.append(f"Dataset contains {len(df)} records")
        
        # Analyze each metric
        for metric in metadata['metrics']:
            if metric not in df.columns:
                continue
                
            stats = df[metric].describe()
            insights.extend([
                f"\n{metric} overview:",
                f"• Average: {stats['mean']:.2f}",
                f"• Range: {stats['min']:.2f} to {stats['max']:.2f}"
            ])
            
            # Create summary visualization
            fig = px.histogram(
                df,
                x=metric,
                title=f'Overview of {metric}',
                marginal='box'
            )
            plots.append({
                'data': fig.to_dict()['data'],
                'layout': fig.to_dict()['layout']
            })
            
            # Time-based analysis if available
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if date_cols:
                date_col = date_cols[0]
                fig_time = px.line(
                    df.sort_values(date_col),
                    x=date_col,
                    y=metric,
                    title=f'{metric} Over Time'
                )
                plots.append({
                    'data': fig_time.to_dict()['data'],
                    'layout': fig_time.to_dict()['layout']
                })

        return {
            'answer': "General Analysis:\n\n" + "\n".join(insights),
            'plots': plots
        }

    def _determine_analysis_type(self, question: str) -> str:
        """Determine the type of analysis needed based on the question."""
        question = question.lower()
        if any(term in question for term in ['trend', 'over time', 'historical']):
            return 'trend'
        elif any(term in question for term in ['compare', 'versus', 'vs', 'difference', 'between']):
            return 'comparison'
        elif any(term in question for term in ['distribution', 'spread', 'range', 'histogram']):
            return 'distribution'
        elif any(term in question for term in ['correlation', 'relationship', 'related']):
            return 'correlation'
        return 'general'

    def _identify_metrics(self, df: pd.DataFrame, question: str) -> List[str]:
        """Identify relevant metrics from the question and dataset."""
        question_tokens = set(question.lower().split())
        metrics = []
        
        # Direct column matches
        for col in df.columns:
            if any(token in col.lower() for token in question_tokens):
                metrics.append(col)
        
        # Semantic matches using metric mappings
        for metric_type, terms in self.metric_mappings.items():
            if any(term in question_tokens for term in terms):
                for col in df.columns:
                    if any(term in col.lower() for term in terms):
                        metrics.append(col)
        
        # If no metrics found, use all numeric columns
        if not metrics:
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return list(set(metrics))

    def _identify_categories(self, df: pd.DataFrame, question: str) -> List[str]:
        """Identify relevant categorical columns based on the question."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        question_tokens = set(question.lower().split())
        
        # Direct matches from question
        categories = [col for col in categorical_cols 
                     if any(token in col.lower() for token in question_tokens)]
        
        # If no categories found but categorical columns exist, use the first one
        if not categories and len(categorical_cols) > 0:
            categories = [categorical_cols[0]]
            
        return categories

    def _extract_time_period(self, question: str) -> Dict:
        """Extract time period information from the question."""
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
        
        # Extract relative time periods
        if 'last year' in question:
            period['relative'] = 'last_year'
        elif 'last month' in question:
            period['relative'] = 'last_month'
        elif 'last week' in question:
            period['relative'] = 'last_week'
        elif 'ytd' in question or 'year to date' in question:
            period['relative'] = 'ytd'
        
        return period

    def _filter_by_time(self, df: pd.DataFrame, time_period: Dict) -> pd.DataFrame:
        """Filter dataset based on specified time period."""
        if not time_period:
            return df
            
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if not date_cols:
            return df
            
        date_col = date_cols[0]
        filtered_df = df.copy()
        
        # Apply absolute time filters
        if 'month' in time_period:
            filtered_df = filtered_df[filtered_df[date_col].dt.month == time_period['month']]
        if 'year' in time_period:
            filtered_df = filtered_df[filtered_df[date_col].dt.year == time_period['year']]
            
        # Apply relative time filters
        if 'relative' in time_period:
            now = pd.Timestamp.now()
            if time_period['relative'] == 'last_year':
                start_date = now - pd.DateOffset(years=1)
            elif time_period['relative'] == 'last_month':
                start_date = now - pd.DateOffset(months=1)
            elif time_period['relative'] == 'last_week':
                start_date = now - pd.DateOffset(weeks=1)
            elif time_period['relative'] == 'ytd':
                start_date = pd.Timestamp(year=now.year, month=1, day=1)
            filtered_df = filtered_df[filtered_df[date_col] >= start_date]
            
        return filtered_df

    def _format_number(self, value: float) -> str:
        """Format numbers for better readability."""
        abs_value = abs(value)
        if abs_value >= 1e9:
            return f"{value/1e9:.1f}B"
        elif abs_value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs_value >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.2f}"

    def _get_correlation_insights(self, df: pd.DataFrame, metrics: List[str]) -> List[str]:
        """Generate insights about correlations between metrics."""
        insights = []
        
        if len(metrics) < 2:
            return insights
            
        correlation_matrix = df[metrics].corr()
        
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= 0.3:  # Only report meaningful correlations
                    strength = (
                        "strong" if abs(corr) >= 0.7 else
                        "moderate" if abs(corr) >= 0.5 else
                        "weak"
                    )
                    direction = "positive" if corr > 0 else "negative"
                    insights.append(
                        f"{metrics[i]} and {metrics[j]} show a {strength} {direction} "
                        f"correlation (r={corr:.2f})"
                    )
                    
        return insights

    def _create_summary_table(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Create a summary table with key statistics for each metric."""
        summary_stats = []
        
        for metric in metrics:
            if metric in df.columns:
                stats = df[metric].describe()
                summary_stats.append({
                    'Metric': metric,
                    'Mean': self._format_number(stats['mean']),
                    'Std Dev': self._format_number(stats['std']),
                    'Min': self._format_number(stats['min']),
                    'Max': self._format_number(stats['max']),
                    'Count': int(stats['count'])
                })
                
        return pd.DataFrame(summary_stats)

    def _get_outliers(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Identify outliers in a metric using IQR method."""
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return df[
            (df[metric] < lower_bound) |
            (df[metric] > upper_bound)
        ]


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


    def _create_general_visualizations(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        visualizations = []
        
        # Create appropriate visualization based on available data
        if metadata['metrics']:
            for metric in metadata['metrics']:
                if metric in df.columns:
                    if metadata['categories']:
                        fig = px.bar(df, x=metadata['categories'][0], y=metric,
                              title=f"{metric} by {metadata['categories'][0]}")
                    else:
                        fig = px.histogram(df, x=metric,
                                         title=f'Distribution of {metric}')
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


# if __name__ == "__main__":
#     # Example usage
#     analyzer = AIDataAnalyzer()
    
#     # Create sample data
#     dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
#     np.random.seed(42)
    
#     sample_data = pd.DataFrame({
#         'date': dates,
#         'satisfaction': np.random.normal(8, 1, len(dates)),
#         'sales': np.random.exponential(1000, len(dates)),
#         'customer_type': np.random.choice(['New', 'Returning'], len(dates)),
#         'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
#     })
    
#     # Example analysis
#     result = analyzer.analyze_question(
#         sample_data,
#         "How has customer satisfaction changed over time, and how does it vary by region?"
#     )
    
#     # Print results
#     print(result['answer'])
#     print(f"\nNumber of plots generated: {len(result['plots'])}")