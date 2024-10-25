import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from logs import get_app_logger, get_error_logger

load_dotenv()

class Utilities:
    """
    A class that encapsulates the dependencies required for the application.

    It provides methods for initializing a language model client, 
    loading and writing chat history, loading and writing file maps, 
    and generating chat completions based on the conversation history.

    Attributes:
        app_logger (Logger): The application logger.
        error_logger (Logger): The error logger.
        OPENAI_API_KEY (str): The OpenAI API key.
        GENIE_ACCESS_TOKEN (str): The Genie access token.

    Methods:
        llm(model, service): Initializes and returns a language model client.
    """
    
    def __init__(self):
        """
        Initializes the Utilities class.

        It sets up the application and error loggers, loads the OpenAI API key, Genie access token, and Hugging Face API token.
        """
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        
    def llm(self, model: str = "gpt-4o", service: str = "OpenAI") -> ChatOpenAI:
        llm = ChatOpenAI(model = model, openai_api_key = self.OPENAI_API_KEY)
        return llm
        
    