import logging
import warnings
from langflow.load import run_flow_from_json
from dotenv import load_dotenv
import os
import json  # For pretty-printing JSON
import openai
import streamlit as st

# Configure logging to show only warnings and errors
logging.basicConfig(level=logging.WARNING)

# Suppress specific warnings from Langfuse
warnings.filterwarnings("ignore", message="Langfuse client is disabled*")

load_dotenv()  # This will load variables from .env into environment

openai_api_key = os.getenv("OPENAI_API_KEY")

TWEAKS = {
  "ChatInput-Vozl5": {},
  "AstraVectorStoreComponent-WHEl7": {},
  "ParseData-VBUIB": {},
  "Prompt-MuxZK": {},
  "ChatOutput-yPhbt": {},
  "SplitText-o6E43": {},
  "File-N86BN": {},
  "AstraVectorStoreComponent-xDqK4": {},
  "OpenAIEmbeddings-rsqYN": {},
  "OpenAIEmbeddings-FCGTL": {},
  "OpenAIModel-Wz1BC": {},
  "note-ehXB0": {},
  "note-g6SrD": {},
  "note-l3eAt": {}
}

def get_response(question: str) -> dict:
    """
    Processes the user's question and returns the AI's response.
    
    Args:
        question (str): The user's input question.
    
    Returns:
        dict: The AI's response.
    """
    try:
        result = run_flow_from_json(
            flow="Vector Store RAG (1).json",
            input_value=question,
            session_id="",  # provide a session id if you want to use session state
            fallback_to_env_vars=True,  # False by default
            tweaks=TWEAKS
        )
        
        # Extract the relevant part of the result
        return result[0].outputs[0].results
    except ValueError as e:
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

def get_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a legal assistant specializing in contract law."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    try:
        question = input("Enter a question: ")
        
        result = run_flow_from_json(
            flow="Vector Store RAG (1).json",
            input_value=question,
            session_id="",  # provide a session id if you want to use session state
            fallback_to_env_vars=True,  # False by default
            tweaks=TWEAKS
        )
        
        # Pretty-print the result using JSON
        print(result[0].outputs[0].results)
    except ValueError as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
