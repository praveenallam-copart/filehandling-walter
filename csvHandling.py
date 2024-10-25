import os
from logs import get_app_logger, get_error_logger
from dotenv import load_dotenv
from typing import Dict
import httpx
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType  
from pathlib import Path
from Utilities import Utilities

load_dotenv()

class CSVHandling:
    """
    A class for handling CSV files.

    Attributes:
        app_logger (Logger): The application logger.
        error_logger (Logger): The error logger.
        OPENAI_API_KEY (str): The OpenAI API key.
        GENIE_ACCESS_TOKEN (str): The Genie access token.
        llm (LLM): The large language model.
    """
    
    def __init__(self):
        """
        Initializes the CSVHandling class.

        Initializes the application logger, error logger, OpenAI API key, Genie access token, and the large language model.
        """
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.llm = Utilities().llm()
    
    async def read_store_csv(self, csvUrl,name):
        """
        Reads and stores a CSV file from a URL.

        Args:
            csvUrl (str): The URL of the CSV file.
            name (str): The name of the CSV file.

        Returns:
            None
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(csvUrl)
                self.app_logger.info(f"csv response => {response.status_code}")
                with open(f"/export/home/saallam/filehandling/InputFiles/uploaded-csvs/{name}.csv", "wb") as file:
                    file.write(response.content)
                self.app_logger.info(f"csv file written...")
                self.app_logger.info(f"csv stored...")
            except Exception as e:
                self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                raise 
    
    def csv_agent(self, name = None):
        """
        Creates a CSV agent and gets a summary of the CSV file.

        Args:
            name (str, optional): The name of the CSV file. Defaults to None.

        Returns:
            tuple: A tuple containing the CSV agent and the summary of the CSV file.
        """
        try:
            folder_path = Path.cwd() / "uploaded-csvs"
            csv_files = []
            refined_summary = None
            for filename in os.listdir(folder_path.as_posix()):
                csv_files.append((folder_path / filename).as_posix())
            if csv_files:
                csv_agent = create_csv_agent(self.llm, csv_files, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True)
                if name:
                    refined_summary = csv_agent.invoke(f"get the brief summary fo the file {name}, include columns and distribution. use describe")["output"]
                self.app_logger.info(f"csv agent created, and got the summary...")
                return csv_agent, refined_summary
            else:   
                return None, None
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        
    def query_agent(self, query, csv_agent):
        """
        Queries the CSV agent.

        Args:
            query (str): The query to be executed.
            csv_agent: The CSV agent.

        Returns:
            Dict: A dictionary containing the response from the CSV agent.
        """
        try:
            csv_response = csv_agent.invoke(query)
            csv_agent_response = csv_response["output"]
            self.app_logger.info(f"There is an answer for the query!!!")
            return {"csv_agent_response" : csv_agent_response}
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise        