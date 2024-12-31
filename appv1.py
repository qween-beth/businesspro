from groq import Groq
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIDataAnalyzer:
    def __init__(self):
        """Initialize the AI Data Analyzer with the Groq client."""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV file.")

    def create_sample_data(self) -> pd.DataFrame:
        """Create sample sales data for demonstration."""
        data = {
            'Date': pd.date_range(start='2024-01-01', periods=100),
            'Product': ['Laptop', 'Phone', 'Tablet', 'Watch'] * 25,
            'Region': ['North', 'South', 'East', 'West'] * 25,
            'Sales': [round(x, 2) for x in np.random.uniform(500, 2000, 100)],
            'Units': np.random.randint(1, 50, 100),
            'Customer_Satisfaction': np.random.randint(1, 6, 100)
        }
        return pd.DataFrame(data)

    def prepare_data_context(self, df: pd.DataFrame) -> str:
        """Prepare data context for the AI model."""
        context = f"""
            Data Summary:
            - Columns: {', '.join(df.columns.tolist())}
            - Number of rows: {len(df)}
            - Date range: {df['Date'].min()} to {df['Date'].max()}
            - Numerical summaries:
            {df.describe().to_string()}

            Sample data (first 5 rows):
            {df.head().to_string()}
        """
        return context

    def ask_groq(self, question: str, context: str) -> str:
        """Query Groq with a question about the data."""
        try:
            # Create a chat completion request
            final_prompt = f"""
                Here is some data I'd like you to analyze:
                {context}

                Question: {question}

                Please provide a clear and concise answer based on the data provided. If relevant, include simple statistical analysis.
            """
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Replace with the actual model ID you're using
                messages=[{"role": "system", "content": final_prompt}],
                max_tokens=350
            )
            # Extract and return the content
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error querying Groq: {str(e)}"

def main():
    # Example usage with sample data
    analyzer = AIDataAnalyzer()

    # Create sample data
    df = analyzer.create_sample_data()

    # Prepare the context
    context = analyzer.prepare_data_context(df)

    # Example questions to demonstrate capabilities
    example_questions = [
        "What is the average sales value by product category?",
        "Which region has the highest customer satisfaction rating?",
        "Is there any correlation between units sold and customer satisfaction?",
        "What are the top 3 performing products by total sales?",
        "Are there any notable trends in sales over time?"
    ]

    print("\nAI Data Analysis Tool - Interactive Mode")
    print("\nSample questions you can ask:")
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")

    print("\nType 'quit' to exit")

    while True:
        question = input("\nWhat would you like to know about your data? ")
        if question.lower() == 'quit':
            print("Exiting the tool. Goodbye!")
            break

        if question.isdigit() and 1 <= int(question) <= len(example_questions):
            question = example_questions[int(question) - 1]

        answer = analyzer.ask_groq(question, context)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
