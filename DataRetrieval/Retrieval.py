import os
from typing import List, Dict
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from logs import get_app_logger, get_error_logger
from RetrievalComponent import RetrievalComponent
from ReRanker import ReRankerComponent

load_dotenv()

class Retrieval:
    """
    A class used to handle PDF retrieval and processing.

    It uses the Haystack library to manage document stores, retrievers, and generators.
    It also uses the OpenAI API for text embedding and generation.

    Attributes:
        app_logger (Logger): Logger instance to log application-related activities.
        error_logger (Logger): Logger instance to log error-related activities.
        OPENAI_API_KEY (str): API key for OpenAI services, loaded from environment variables.
        GENIE_ACCESS_TOKEN (str): Access token for additional services, loaded from environment variables.
        llm (OpenAIGenerator): The LLM instance initialized with the GPT-4 model.
        HF_API_TOKEN (str): API key for HuggingFace services, loaded from environment variables.
        vector_store_chroma_Politics (ChromaDocumentStore): Chroma document store for politics.
        vector_store_chroma_Education (ChromaDocumentStore): Chroma document store for education.
        vector_store_chroma_Sports (ChromaDocumentStore): Chroma document store for sports.
        vector_store_chroma_Environment (ChromaDocumentStore): Chroma document store for environment.
        vector_store_chroma_Others (ChromaDocumentStore): Chroma document store for others.
        vector_store_chroma_Images (ChromaDocumentStore): Chroma document store for images.
        db_map (dict): A dictionary mapping categories to their respective document stores and abbreviations.

    Methods:
        __init__(): Initializes the Decomposition class by setting up loggers, loading environment variables, and initializing the OpenAI LLM with structured output.
        
        run(query, category, name, userid) -> str
            Runs the pipeline to retrieve and process documents based on the query, category, name, and user ID.
    """
    
    def __init__(self):
        """
        Initializes the PdfRetrieval class.

        It sets up the application and error loggers, loads the OpenAI API key, Genie access token, and Hugging Face API token.
        It also initializes the OpenAI generator model and sets up the Chroma document stores for different categories.
        """
        
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.HF_API_TOKEN = os.getenv("HF_API_TOKEN")
        self.llm = OpenAIGenerator(model = "gpt-4o")
        
        try:
            self.persist_path = "/export/home/saallam/filehandling/walter-vector-storage"
            self.vector_store_chroma_Politics = ChromaDocumentStore(collection_name= "Politics", persist_path = self.persist_path, embedding_function="OpenAIEmbeddingFunction",api_key=  self.OPENAI_API_KEY)
            self.vector_store_chroma_Education = ChromaDocumentStore(collection_name= "Education", persist_path = self.persist_path, embedding_function="OpenAIEmbeddingFunction",api_key=  self.OPENAI_API_KEY)
            self.vector_store_chroma_Sports = ChromaDocumentStore(collection_name= "Sports", persist_path = self.persist_path, embedding_function="OpenAIEmbeddingFunction",api_key=  self.OPENAI_API_KEY)
            self.vector_store_chroma_Environment = ChromaDocumentStore(collection_name= "Environment", persist_path = self.persist_path, embedding_function="OpenAIEmbeddingFunction",api_key=  self.OPENAI_API_KEY)
            self.vector_store_chroma_Others = ChromaDocumentStore(collection_name= "Others", persist_path = self.persist_path, embedding_function="OpenAIEmbeddingFunction",api_key=  self.OPENAI_API_KEY)
            self.vector_store_chroma_Images =  ChromaDocumentStore(collection_name= "Images", persist_path = self.persist_path, embedding_function="OpenAIEmbeddingFunction",api_key=  self.OPENAI_API_KEY)

            self.db_map = {
                "Education" : [self.vector_store_chroma_Education, "Edu"],
                "Sports" : [self.vector_store_chroma_Sports, "Sports"],
                "Politics" : [self.vector_store_chroma_Politics, "politics"],
                "Environment" : [self.vector_store_chroma_Environment, "env"],
                "Others" : [self.vector_store_chroma_Others, "others"],
                "Images" : [self.vector_store_chroma_Images, "images"]
            }
            self.app_logger.info(f"Vector Databases are retrieved.....")  
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def run(self, query, category, name, userid):
        """
        Runs the pipeline to retrieve and process documents based on the query, category, name, and user ID.

        Args:
        ----
        query (str): The query to search for.
        category (str): The category of the documents.
        name (str): The name of the document.
        userid (str): The user ID.

        Returns:
        -------
        str: The response from the pipeline.
        """
        
        document_store = self.db_map[category][0]
        
        retriever = RetrievalComponent(retriever = ChromaQueryTextRetriever(document_store = document_store))
        ranker = ReRankerComponent() # BAAI/bge-reranker-base
        # ranker.warm_up()
        try:
            prompt = """Answer the question given the context.
                        Question: {{ query }}
                        Context:
                        {% for document in documents %}
                            {{ document.content }}
                        {% endfor %}
                        Answer:"""
            prompt_builder = PromptBuilder(template=prompt)
            
            pipeline = Pipeline()
            pipeline.add_component("retriever", retriever)
            pipeline.add_component("ranker", ranker)
            pipeline.add_component("prompt", prompt_builder)
            pipeline.add_component("generator", self.llm)
            
            pipeline.connect("retriever", "ranker")
            pipeline.connect("ranker", "prompt.documents")
            pipeline.connect("prompt", "generator")
            
            response = pipeline.run({"retriever" : {"queries" : query, "name" : name, "userid" : userid},
                                    "prompt" : {"query" : query}
                                    },
                                    include_outputs_from={"retriever", "ranker"})
            
            self.app_logger.info(f"Generated the answer using uploaded files...!!!")                        
            return response["generator"]["replies"][0], response["ranker"], response["retriever"]["question_context_pairs"]
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise