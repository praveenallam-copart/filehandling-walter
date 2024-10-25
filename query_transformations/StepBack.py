import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from logs import get_app_logger, get_error_logger
from Utilities import Utilities
import prompts

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

class StepBackResponses(BaseModel):
    query: str = Field(description="A stepback (more generic/ relevant) question of the given query")

class StepBack:
    """
    A class to generate more generic questions based on specific user-provided questions.

    This class utilizes an AI assistant to extract broader questions that address the underlying principles or concepts needed to answer the specific queries provided by users. The aim is to reframe questions in a way that facilitates a deeper understanding of the topic at hand.

    Attributes:
        app_logger (Logger): Logger instance to log application-related activities.
        error_logger (Logger): Logger instance to log error-related activities.
        OPENAI_API_KEY (str): API key for OpenAI services, loaded from environment variables.
        GENIE_ACCESS_TOKEN (str): Access token for additional services, loaded from environment variables.
        llm (ChatOpenAI): The LLM instance initialized with the GPT-4 model.
        structured_llm (ChatOpenAI): The LLM instance with structured output in the format of `StepBackResponses`.

    Methods:
        __init__(): Initializes the Decomposition class by setting up loggers, loading environment variables, and initializing the OpenAI LLM with structured output.
    
        run(query: str) -> StepBackResponses:
            Extracts a more generic question from a specific user-provided question.

    Notes:
        - The `run` method is designed to provide insights by rephrasing specific questions into more general ones, focusing on broader implications or related concepts.
    """
    
    def __init__(self):
        """
        Initializes the AI assistant.

        This constructor sets up the necessary logging and API configurations
        for the AI assistant, including loading environment variables for API keys
        and initializing the language model with structured output capabilities.

        Attributes:
            app_logger: Logger for application-level logging.
            error_logger: Logger for error-level logging.
            OPENAI_API_KEY (str): The API key for OpenAI, retrieved from environment variables.
            GENIE_ACCESS_TOKEN (str): The access token for Genie, retrieved from environment variables.
            llm: Instance of ChatOpenAI configured to use the "gpt-4o" model.
            structured_llm: An instance of llm with structured output capabilities for StepBackResponses.
        """
        
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.llm = Utilities().llm()
        self.structured_llm = self.llm.with_structured_output(StepBackResponses)
        
    def run(self, query):
        """
        Extracts a more generic question from a specific user-provided question.

        This method takes a specific question and generates a broader question that addresses the underlying principles or concepts needed to answer the specific query. The goal is to reframe the question in a way that allows for a deeper exploration of the topic.

        Parameters:
            query : str
                The specific question provided by the user, which will be reframed into a more general question.

        Returns:
            StepBackResponses
                An object containing the generated more generic question derived from the user query.

        Notes:
            - This method aims to provide a more insightful perspective on the specific question, focusing on the broader implications or related concepts.
        """
        
        template = prompts.STEPBACK_SYSTEM_PROMPT
        
        prompt = ChatPromptTemplate.from_messages([
                                        ("system", template),
                                        ("human", "Question: {query}")
                                    ])
                                
        chain = prompt | self.structured_llm
        response = chain.invoke({"query" : query})
        
        return response
      
      