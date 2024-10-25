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

class CoreMeaningQuery(BaseModel):
    """Takes the query and gives the core meaning of the query"""
    
    core_meaning: str = Field(description="The core meaning of the query")

class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind, helpful and be precise with the answer with less hallucination."""

    response: str = Field(description="A conversational response to the user's query")

class InternalDataRequests(BaseModel):
    """Based on the chat history and the user's query, determine if the query is related to any of the files previously uploaded by the user, where the filenames and file summaries are recorded in the chat history. If the query is related to a file, return the filename, file type, and the action required to address the query"""
    
    filename: Optional[str] = Field(description = "The name of the file user is referring to, use the chat history for better contextual understanding", default = None)
    query: str = Field(description = "Query given by the user")
    filetype: Optional[str] = Field(description = "If a filename is founf then find the type of the file. For exmaple csv, pdf, jpeg, png, jpg", default = None)
    action: Optional[str] = Field(description = "If the file type is pdf/ image and the query needs to access vector database (you can use summary in chat history for better unserstanding) then give 'retrieve' as output or if the filetype is csv return 'CsvAgent'", default = None)
    
class KnowledgeRouterResponse(BaseModel):
    """
    Represents a response from the Knowledge Router.

    Attributes:
    response (Union[InternalDataRequests, ConversationalResponse]): 
        The response from the Knowledge Router, which can be either an internal data request or a conversational response.
    """
    response: Union[InternalDataRequests, ConversationalResponse]
    
class KnowledgeRouter:
    """
    A class responsible for routing user queries to relevant files or providing helpful responses.

    The KnowledgeRouter uses a large language model (LLM) to analyze the user's query and chat history.
    It determines if the query is related to any previously uploaded files and returns the filename, file type, and required action.
    If the query is not related to any file, it provides a friendly and helpful response.

    Attributes:
        app_logger (Logger): The application logger.
        error_logger (Logger): The error logger.
        OPENAI_API_KEY (str): The OpenAI API key.
        GENIE_ACCESS_TOKEN (str): The Genie access token.
        llm (ChatOpenAI): The large language model instance.
        structured_llm (ChatOpenAI): The structured large language model instance.

    Methods:
        run(query: str, chat_history: str) -> KnowledgeRouterResponse:
            Analyzes the user's query and chat history to determine the relevant response.
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
        self.structured_llm = self.llm.with_structured_output(KnowledgeRouterResponse)
        self.app_logger.info(f"In KnowledgeRouter....")
        
    def run(self, query, chat_history):
        """
        Analyzes the user's query and chat history to determine the relevant response.

        Args:
            query (str): The user's query.
            chat_history (str): The chat history.

        Returns:
            KnowledgeRouterResponse: The response from the large language model.
        """
        
        prompt = f"""
        Based on the chat history and the user's query, determine if the query is related (directly or semantically or aligned or in the context) to any of the files previously uploaded by the user, where the filenames and file summaries are recorded in the chat history. 
        If the query is related to a file, return the filename, file type, and the action required using the {InternalDataRequests} class. 
        If the query is not related to any file provide a friendly and helpful response using the {ConversationalResponse} class.
        If you're not sure whether the query is related to the previous files or if there is uncertainity provide a friendly and helpful response using {ConversationalResponse} class and ask if the query is realted to any of the uploaded files.
        
        Do not do any mistakes or hallucinate, just use the instructions given.

        Query: {{query}}
        Chat History: {{chat_history}}
        Knowledge Router Response:
        """
        try:
            prompt_template = ChatPromptTemplate.from_template(prompt)
            messages = prompt_template.format_messages(query = query, chat_history = chat_history)
            response = self.structured_llm.invoke(messages).response
            core_query = None
            if type(response) == InternalDataRequests:
                # self.app_logger.info(f"Internal Data Request: {response.response.query}")
                prompt = prompt = ChatPromptTemplate([
                    ("system", prompts.CORE_MEANING_PROMPT),
                    ("human", "Query: {query}")
                ])
                
                llm = self.llm.with_structured_output(CoreMeaningQuery)
                chain = prompt | llm
                query = chain.invoke({"query" : response.query})
                core_query = query.core_meaning
            self.app_logger.info(f"Got the knowledge about the {query=}...")
            return response, core_query
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise