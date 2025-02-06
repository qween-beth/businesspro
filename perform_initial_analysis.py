from flask import Flask, send_file, make_response, render_template, request, flash, jsonify, session, redirect, url_for, session, has_request_context
from business_intelligence import BusinessAnalyzer
from data_analyzer import AIDataAnalyzer
import pandas as pd
import numpy as np
import tempfile
import random
from datetime import datetime, timedelta

def perform_initial_analysis(df):
    try:
        print("DEBUG: Starting initial analysis...")

        if df.empty:
            print("DEBUG: Dataset is empty.")
            return {'error': 'The dataset is empty.'}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns

        if numeric_cols.empty:
            print("DEBUG: No numeric columns found.")
            return {'error': 'No numeric columns found for analysis.'}

        if date_cols.empty:
            print("DEBUG: No date columns found.")
            return {'error': 'No date columns found for time-based analysis.'}

        # Generate summary statistics
        print("DEBUG: Generating summary statistics...")
        summary_stats = BusinessAnalyzer.generate_summary_stats(df)

        # Detect outliers
        print("DEBUG: Detecting outliers...")
        outliers = BusinessAnalyzer.detect_outliers(df)

        # Generate visualizations
        print("DEBUG: Generating visualizations...")
        plots = BusinessAnalyzer.generate_plots(df)
        for key, plot_data in plots.items():
            if hasattr(plot_data, 'to_json'):
                plots[key] = plot_data.to_json()

        # AI insights
        print("DEBUG: Fetching AI-based insights from Groq API...")
        ai_data_analyzer = AIDataAnalyzer()
        context = ai_data_analyzer.prepare_data_context(df)
        question = "Provide key insights about the dataset."

        try:
            groq_insights = ai_data_analyzer.ask_groq(question, context)
            print("DEBUG: AI insights received.")
        except Exception as e:
            print(f"DEBUG: AI insights error - {e}")
            groq_insights = "Error retrieving AI insights"

        # Prepare response
        response = {
            'summary_statistics': summary_stats,
            'outliers': outliers,
            'visualizations': plots,
            'groq_insights': groq_insights
        }

        print("DEBUG: Initial analysis completed successfully.")
        return response

    except Exception as e:
        print(f"DEBUG: Initial Analysis Error - {e}")
        return {'error': str(e)}
