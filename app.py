from flask import Flask, render_template, request, flash, session, jsonify
from data_analyzer import AIDataAnalyzer
from business_intelligence import BusinessAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime
import json
import plotly
import plotly.express as px
from typing import Dict, List, Any
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
data_analyzer = AIDataAnalyzer()
business_intelligence = BusinessAnalyzer()

app.secret_key = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
                return render_template('index.html')
            
            if not file.filename.endswith('.csv'):
                flash('Please upload a CSV file')
                return render_template('index.html')
            
            try:
                df = pd.read_csv(file)
                
                # Convert date columns to datetime
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            continue
                
                # Generate business insights
                summary_stats = business_intelligence.generate_summary_stats(df)
                outliers = business_intelligence.detect_outliers(df)
                plots = business_intelligence.generate_plots(df)
                
                # Store in session
                session['df'] = df.to_json(date_format='iso')
                session['summary_stats'] = json.dumps(summary_stats)
                session['outliers'] = json.dumps(outliers)
                session['plots'] = plots
                
                return render_template('results.html',
                                     columns=df.columns.tolist(),
                                     sample_data=df.head().to_html(),
                                     summary_stats=summary_stats,
                                     outliers=outliers,
                                     plots=plots)
            
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return render_template('index.html')
    
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint for analyzing specific aspects of the data."""
    try:
        analysis_type = request.json.get('analysis_type')
        df = pd.read_json(session['df'])
        
        if analysis_type == 'correlation':
            numeric_df = df.select_dtypes(include=[np.number])
            correlation = numeric_df.corr().to_dict()
            return jsonify({'success': True, 'data': correlation})
        
        elif analysis_type == 'growth':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            growth_rates = {
                col: ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0] * 100)
                for col in numeric_cols
            }
            return jsonify({'success': True, 'data': growth_rates})
        
        return jsonify({'success': False, 'error': 'Invalid analysis type'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export/<format>', methods=['GET'])
def export_data(format):
    """Export analyzed data in various formats."""
    try:
        df = pd.read_json(session['df'])
        summary_stats = json.loads(session['summary_stats'])
        
        if format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Raw Data')
                pd.DataFrame([summary_stats]).to_excel(writer, sheet_name='Summary')
            
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='analysis_report.xlsx'
            )
        
        elif format == 'pdf':
            # Generate PDF report (requires additional PDF generation library)
            pass
        
    except Exception as e:
        flash(f'Error exporting data: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)