# AI Data Analyzer Flask App

A web application for analyzing CSV data using Groq AI.

## Setup

1. Install requirements:
```bash
pip install flask pandas groq python-dotenv
```

2. Create `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

3. Run the app:
```bash
python app.py
```

## Project Structure

```
ai-data-analyzer/
├── .env
├── app.py
├── analyzer.py
├── static/
│   └── style.css
└── templates/
    ├── index.html
    └── results.html
```

## Files


New business-focused features added:

Advanced Analytics:

Automatic trend detection
Growth rate calculations
Anomaly detection
Correlation analysis
Time series analysis


Business Intelligence Dashboard:

Key performance indicators
Quick stats cards
Growth metrics
Trend visualization


Data Export Options:

Excel export with multiple sheets
PDF report generation
Raw data export


Interactive Visualizations:

Time series plots
Distribution analysis
Correlation heatmaps
Interactive data tables


Data Quality Features:

Missing data analysis
Outlier detection
Data validation reports
Data quality scores


Business Reporting:

Automated insights
Custom metric calculations
Period-over-period comparisons
Executive summaries