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

class DecompositionResponses(BaseModel):
    queries: List[str] = Field(description="List of decomposed quereies generated")

class Decomposition:
    """
    Decomposition Class

    This class, `Decomposition`, is responsible for taking a large user query and breaking it down into smaller, more manageable sub-queries. It uses OpenAI's GPT-4 model to achieve this task through structured output defined by the `DecompositionResponses` class.

    Attributes:
        app_logger (Logger): Logger instance to log application-related activities.
        error_logger (Logger): Logger instance to log error-related activities.
        OPENAI_API_KEY (str): API key for OpenAI services, loaded from environment variables.
        GENIE_ACCESS_TOKEN (str): Access token for additional services, loaded from environment variables.
        llm (ChatOpenAI): The LLM instance initialized with the GPT-4 model.
        structured_llm (ChatOpenAI): The LLM instance with structured output in the format of `DecompositionResponses`.

    Methods:
        __init__(): Initializes the Decomposition class by setting up loggers, loading environment variables, and initializing the OpenAI LLM with structured output.
        
        run(query: str) -> DecompositionResponses:
            Takes a user input query as an argument and generates a structured decomposition response.
            
            The query is broken down using a pre-defined template to generate relevant sub-queries, which can then be answered in isolation. If the query cannot be broken down, the response indicates that no sub-queries were generated.
            
            Args:
                query (str): The input query to be decomposed.
            
            Returns:
                DecompositionResponses: A Pydantic model that contains a list of sub-queries generated from the input query.

    Example:
        decomposition = Decomposition()
        response = decomposition.run("What's the difference between LangChain agents and LangGraph?")
        print(response)
        # Output: DecompositionResponses(queries=["What's the difference between LangChain agents and LangGraph?", "What are LangChain agents?", "What is LangGraph?"])
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
            structured_llm: An instance of llm with structured output capabilities for DecompositionResponses.
        """
        
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.llm = Utilities().llm()
        self.structured_llm = self.llm.with_structured_output(DecompositionResponses)
        self.app_logger.info(f"breaking down the large query info small sub queries...")
        

        
    def run(self, query):
        """
        Processes a given query to generate sub-queries.

        This method takes an input query and generates a set of sub-queries
        that break down the input into manageable parts. The assistant aims to
        decompose complex queries into simpler, isolated sub-questions based on
        the provided template.

        Parameters:
            query (str): The input query that needs to be decomposed into sub-queries.

        Returns:
            DecompositionResponses: An instance containing a list of generated sub-queries
            based on the input query. If no sub-queries can be generated, an appropriate
            message will be returned indicating that further decomposition is not possible.
        """
        
        template = prompts.DECOMPOSITION_SYSTEM_PROMPT
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                                            ("system", template),
                                            ("human", "Input: {query}")
                                        ])
                                    
            chain = prompt | self.structured_llm
            response = chain.invoke({"query" : query})
            self.app_logger.info(f"Decomposition done...")
            return response
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
      
