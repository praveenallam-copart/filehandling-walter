import os
from typing import List
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from logs import get_app_logger, get_error_logger
from Utilities import Utilities
import prompts

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

class MultiQueryResponses(BaseModel):
    queries: List[str] = Field(description="List of multiple quereies generated")


class MultiQuery:
    """
    A class to handle the generation of multiple differently worded questions based on user input.

    This class utilizes an AI assistant to take a user-provided query and generate a specified number of alternative questions that convey the same meaning but are rephrased or approached from different perspectives. It is designed to ensure that generated questions maintain context and meaning without introducing unrelated content.

    Attributes:
        app_logger (Logger): Logger instance to log application-related activities.
        error_logger (Logger): Logger instance to log error-related activities.
        OPENAI_API_KEY (str): API key for OpenAI services, loaded from environment variables.
        GENIE_ACCESS_TOKEN (str): Access token for additional services, loaded from environment variables.
        llm (ChatOpenAI): The LLM instance initialized with the GPT-4 model.
        structured_llm (ChatOpenAI): The LLM instance with structured output in the format of `MultiQueryResponses`.
        
    Methods:
        __init__(): Initializes the Decomposition class by setting up loggers, loading environment variables, and initializing the OpenAI LLM with structured output.
        
        run(query: str, number: int) -> MultiQueryResponses:
            Generates multiple alternative questions based on the provided user query.

    Notes:
        - The `run` method returns as many questions as possible if it cannot generate the specified number while maintaining the context and meaning of the original query.
        - If no questions can be generated, the method will indicate this.
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
            structured_llm: An instance of llm with structured output capabilities for MultiQueryResponses.
        """
        
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.llm = Utilities().llm()
        self.structured_llm = self.llm.with_structured_output(MultiQueryResponses)
        self.app_logger.info(f"waiting for different versions of given query...")
    
    
    def run(self, query, number = 5):
        """
        Generates multiple differently worded questions based on the provided user query.

        This method takes a user-provided question and generates a specified number of alternative questions that convey the same meaning. The generated questions should be rephrased or approached from different perspectives without introducing unrelated content.

        Parameters:
            query : str
                The original question provided by the user to generate alternatives for.
            number : int
                The number of alternative questions to generate. If fewer questions can be generated, the method will return as many as possible.

        Returns:
            MultiQueryResponses
                An object containing a list of generated questions. The list may contain fewer questions than requested if it is difficult to generate the specified number.

        Notes:
            - If the method encounters difficulty generating the requested number of questions, it will provide as many as possible while maintaining context and meaning.
            - If it is not feasible to generate any questions, the method will indicate this.
        """
        
        template = prompts.MULTIQUERY_SYSTEM_PROMPT
        try:
            prompt = ChatPromptTemplate.from_messages([
                                                        ("system", template),
                                                        ("human", "Query : {query}, Number : {number}"),
                                                    ])
            
            chain = prompt | self.structured_llm 
            response = chain.invoke({"query" : query, "number" : number})
            self.app_logger.info(f"MultiQuery done....")
            return response
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
      
