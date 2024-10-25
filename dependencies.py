import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from logs import get_app_logger, get_error_logger

load_dotenv()

class Dependencies:
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
        get_timestamp(): Returns the current timestamp.
        get_model(llm_service, model): Initializes and returns a language model client.
        get_chat_history(): Loads and returns the complete chat history from a JSON file.
        write_chat_history(chat_history): Writes the given chat history to a JSON file.
        write_file_map(file_map): Writes the given file map to a JSON file.
        history(): Retrieves and transforms the chat history into a list of human-readable messages.
        get_file_map(): Loads and returns the file map from a JSON file.
        chat_completion(query, history): Generates a response to the user's query using an LLM, based on chat history.
    """
    
    def __init__(self):
        """
        Initializes the Dependencies class.

        It sets up the application and error loggers, loads the OpenAI API key, Genie access token, and Hugging Face API token.
        """
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")

    def get_timestamp(self):
        """gives the current timestamp"""
        self.app_logger.info("Accessed time....!")
        return str(datetime.now())

    def get_model(self, llm_service:str = "langchain", model:Union[str, None] = None):
        """
            Initializes and returns a language model (LLM) client based on the specified service.

            Args:
                llm_service (str): The service from which to select the model. Defaults to "langchain". 
                                Options include:
                                - "langchain"
                                - "groq"
                model (Union[str, None]): The specific model to be used. If None or empty, defaults to:
                                        - "gpt-4o" for the "langchain" service.
                                        - "google/gemini-1.5-pro-001" for groq.

            Returns:
                client: The initialized LLM client based on the given service and model.

            Raises:
                Exception: If any error occurs during client initialization, an error log entry is created 
                        and the exception is raised.
            
            Notes:
                - For the "groq" service, a client is created with a custom base URL and the `GENIE_ACCESS_TOKEN`.
                - For all other services (e.g., "langchain"), the client uses the `OPENAI_API_KEY`.

            Example:
                client = self.get_model(llm_service="groq")
        """
        
        # if model is None or model == "":
        #     model = "gpt-4o" if llm_service == "langchain" else "google/gemini-1.5-pro-001"
        try:
        #     client = ChatOpenAI(model=model, base_url="http://copartcodegenapi-ws.c-qa4.svc.rnq.k8s.copart.com/v1", api_key = self.GENIE_ACCESS_TOKEN, temperature = 0) \
        #             if llm_service == "groq" else \
        #                 ChatOpenAI(model = model, api_key = self.OPENAI_API_KEY, temperature = 0)
            client = ChatOpenAI(model = "gpt-4o", api_key = self.OPENAI_API_KEY, temperature = 0)
            self.app_logger.info(f"LLM initialised...{llm_service} @ {self.get_timestamp()}")
            return client
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def get_chat_history(self):
        """
            Loads and returns the complete chat history from a JSON file.

            This function attempts to open and read the chat history stored in 
            the file located at "/export/home/saallam/filehandling/chat_history.json". 
            If successful, the chat history is loaded into the `self.complete_chat_history` attribute.

            Raises:
                Exception: If any error occurs while reading the file (e.g., file not found, permission error),
                        an error log entry is created, and the exception is raised.

            Example:
                self.get_chat_history()
        """
        try:
            with open("/export/home/saallam/filehandling/chat_history.json", "r") as f:
                self.complete_chat_history = json.load(f)
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
            
    def write_chat_history(self, chat_history):
        """
            Writes the given chat history to a JSON file.

            Args:
                chat_history (dict): The chat history data to be written to the 
                                    file located at "/export/home/saallam/filehandling/chat_history.json".
            
            Raises:
                Exception: If an error occurs during the write process, an error log entry is created, 
                        and the exception is raised.

            Example:
                self.write_chat_history(chat_history)
        """
        try:
            with open("/export/home/saallam/filehandling/chat_history.json", "w") as f:
                f.write(json.dumps(chat_history, indent=4))
            self.app_logger.info(f"upadted chat history...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def write_file_map(self, file_map):
        """
            Writes the given file map to a JSON file.

            Args:
                file_map (dict): The file map data to be written to the file 
                                located at "/export/home/saallam/filehandling/file_map.json".

            Raises:
                Exception: If an error occurs during the write process, an error log entry is created,
                        and the exception is raised.

            Example:
                self.write_file_map(file_map)
        """
        try:
            with open("/export/home/saallam/filehandling/file_map.json", "w") as f:
                f.write(json.dumps(file_map))
            self.app_logger.info(f"updated file map...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def history(self):
        """
            Retrieves and transforms the chat history into a list of human-readable messages.

            This method loads the chat history using the `get_chat_history` method, 
            transforms each message based on its role ("user" or other), and returns 
            both the transformed history and the complete chat history.

            Returns:
                tuple: A tuple containing:
                    - history (list): The transformed chat history where user messages are represented as `HumanMessage`.
                    - complete_chat_history (dict): The original chat history loaded from the file.

            Raises:
                Exception: If any error occurs during processing, an error log entry is created, 
                        and the exception is raised.

            Example:
                history, complete_chat_history = self.history()
        """
        self.get_chat_history()
        history = []
        try:
            for chat in self.complete_chat_history:
                content = HumanMessage(content = chat["content"]) if chat["role"] == "user" else chat["content"]
                history.append(content)
            self.app_logger.info(f"chat history transformed and retrieved...")
            return history, self.complete_chat_history
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
            
    def get_file_map(self):
        
        """
            Loads and returns the file map from a JSON file.

            This method attempts to open and read the file map stored in the 
            file located at "/export/home/saallam/filehandling/file_map.json".

            Returns:
                file_map (dict): The file map loaded from the JSON file.

            Raises:
                Exception: If any error occurs while reading the file, an error log entry is created, 
                        and the exception is raised.

            Example:
                file_map = self.get_file_map()
        """
        try:
            with open("/export/home/saallam/filehandling/file_map.json", "r") as f:
                file_map = json.load(f)
            self.app_logger.info(f"file map loaded...")
            return file_map
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        
    async def chat_comlpletion(self, query, history):
        
        """
            Generates a response to the user's query using an LLM, based on chat history.

            Args:
                query (str): The user's current query.
                history (list): The conversation history provided for context.

            Returns:
                str: The LLM-generated response to the query.

            Raises:
                Exception: If an error occurs during the LLM invocation, an error log entry is created, 
                        and the exception is raised.

            Example:
                response = self.chat_completion(query="What is the status?", history=history)
        """
        llm = self.get_model(llm_service = "groq", llm = "groq/llama-3.1-70b-versatile")
        try:
            prompt = """
            You are an AI assistant that provides answers based on previous conversations and the user's current question. Consider the conversation history provided, and answer the current query in context.

            Conversation History: {history}
            Current Query: {query}
            """
            prompt_template = ChatPromptTemplate.from_template(prompt)
            message = prompt_template.format_messages(query = query, history = history)
            response = await llm.ainvoke(message)
            self.app_logger.info(f"chat completion done....got a response!!")
            return response.content
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise