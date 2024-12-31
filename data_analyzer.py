from groq import Groq
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class AIDataAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def prepare_data_context(self, df: pd.DataFrame) -> str:
        context = f"""
            Data Summary:
            - Columns: {', '.join(df.columns.tolist())}
            - Number of rows: {len(df)}
            - Numerical summaries:
            {df.describe(include='all').to_string()}
            
            Sample data (first 5 rows):
            {df.head().to_string()}
        """
        return context

    def ask_groq(self, question: str, context: str) -> str:
        try:
            final_prompt = f"""
                Here is some data I'd like you to analyze:
                {context}
                
                Question: {question}
                
                Please provide a clear and concise answer based on the data provided.
            """
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "system", "content": final_prompt}],
                max_tokens=350
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error querying Groq: {str(e)}"