import os 
from dotenv import load_dotenv
from typing import Union, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from logs import get_app_logger, get_error_logger
from Utilities import Utilities
import prompts

load_dotenv()

class QueryTransformationRouterResponse(BaseModel):
    """
    Represents the response from the QueryTransformationRouter.

    Attributes:
        transformation (str): The best transformation technique for the given query.
        reason (str): The reason why the above transformation technique is chosen.
    """
    transformation: str = Field(description="The best transformation technique for the given query")
    reason: str = Field(description = "Reason why the above transformation technique is chosen")
    
class QueryTransformationRouter:
    """
    A class responsible for routing user queries to relevant files or providing helpful responses.

    The QueryTransformationRouter uses a large language model (LLM) to analyze the user's query and 
    determines the most appropriate query transformation technique.

    Attributes:
        app_logger (Logger): The application logger.
        error_logger (Logger): The error logger.
        OPENAI_API_KEY (str): The OpenAI API key.
        GENIE_ACCESS_TOKEN (str): The Genie access token.
        llm (ChatOpenAI): The large language model instance.
        structured_llm (ChatOpenAI): The structured large language model instance.

    Methods:
        run(query: str) -> QueryTransformationRouterResponse:
            Analyzes the user's query to determine the relevant response.
    """
    
    def __init__(self):
        """
        Initializes the KnowledgeRouter instance.

        Sets up the application and error loggers, retrieves the OpenAI API key and Genie access token,
        and initializes the large language model instances.
        """
        
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.llm =Utilities().llm()
        self.structured_llm = self.llm.with_structured_output(QueryTransformationRouterResponse)
        self.app_logger.info(f"Waiting for query transformation response...")
        
    def run(self, query):
        """
        Analyzes the user's query to determine the relevant response.

        Args:
            query (str): The user's query.

        Returns:
            QueryTransformationRouterResponse: The response from the large language model.
        """
        
        prompt = f"""You are an assistant that determines the most appropriate query transformation technique for a given user query.
 
        Analyze the query carefully and choose one of the following techniques:
        1. **Decomposition** – If the query contains multiple questions or has a complex structure, recommend this technique to break it into smaller, manageable parts.
        2. **MultiQuery** – If the query is vague, ambiguous, or uses indirect phrasing that could have multiple meanings, recommend generating semantic variants of the query.
        3. **CoreMeaning** – If the query contains unnecessary or redundant information, recommend this technique to remove irrelevant content and focus on the key intent.
        4. **None** – If the query is already concise, clear, and relevant, recommend no special transformation.
        
        Do not do any mistakes or hallucinate, just use the instructions given.

        Query: {{query}}
        Query transformation response : 
        
        Please analyze and provide the appropriate response."""
        
        try:
            prompt_template = ChatPromptTemplate.from_template(prompt)
            messages = prompt_template.format_messages(query = query)
            response = self.structured_llm.invoke(messages)
            self.app_logger.info(f"Query transformation technique found...")
            return response
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise