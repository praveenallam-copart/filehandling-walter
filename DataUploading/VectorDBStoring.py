import os
import fitz
import base64
import asyncio
import requests
from uuid import uuid4
from typing import List, Dict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import LLMChain 
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dependencies import Dependencies
from logs import get_app_logger, get_error_logger
from DataUploading.ImageHandling import ImageHandling

load_dotenv()

class VectorDBStroing:
    def __init__(self):
        """
        Initializes the PdfUploading class.

        Attributes:
            app_logger (Logger): The application logger instance.
            error_logger (Logger): The error logger instance.
            OPENAI_API_KEY (str): The OpenAI API key.
            GENIE_ACCESS_TOKEN (str): The Genie access token.
            llm (LLM): The language model instance.
            embeddings (OpenAIEmbeddings): The OpenAI embeddings instance.
            vector_store_chroma_Education (Chroma): The Chroma instance for the Education vector store.
            vector_store_chroma_Sports (Chroma): The Chroma instance for the Sports vector store.
            vector_store_chroma_Politics (Chroma): The Chroma instance for the Politics vector store.
            vector_store_chroma_Environment (Chroma): The Chroma instance for the Environment vector store.
            vector_store_chroma_Others (Chroma): The Chroma instance for the Others vector store.
            vector_store_chroma_Images (ChromaDocumentStore): Chroma document store for images.
            db_map (dict): A dictionary mapping categories to their corresponding vector stores.

        Raises:
            Exception: If an error occurs during initialization.
        """
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store_chroma_Education = Chroma(
                    collection_name="Education",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Sports = Chroma(
                    collection_name="Sports",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Politics = Chroma(
                    collection_name="Politics",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Environment = Chroma(
                    collection_name="Environment",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Others = Chroma(
                    collection_name="Others",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Images = Chroma(
                    collection_name="Images",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.db_map = {
                "Education" : self.vector_store_chroma_Education,
                "Sports" : self.vector_store_chroma_Sports,
                "Politics" : self.vector_store_chroma_Politics,
                "Environment" : self.vector_store_chroma_Environment,
                "Others" : self.vector_store_chroma_Others,
                "Images" : self.vector_store_chroma_Images
                
            }
            self.app_logger.info(f"Vector Databases are retrieved.....")  
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def load_vectorstore(self, vector_db, documents):
        """
        Stores the given documents into a vector database.

        Args:
            vector_db (Chroma): The vector database instance to store the documents.
            documents (List[Document]): A list of Langchain Documents to be stored.

        Raises:
            Exception: If an error occurs while adding documents to the vector store.
        """
        try:    
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_db.add_documents(documents=documents, ids=uuids)
            self.app_logger.info(f"Splits are stored into the vector databse...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
            
    def run(self, documents, category = None):
        """
        Stores the given documents into a vector database based on the specified category.

        Args:
            documents (List[Document]): A list of Langchain Documents to be stored.
            category (str, optional): The category of the documents. Defaults to None.

        Returns:
            str: A success message if the documents are stored successfully.

        Raises:
            Exception: If an error occurs during storage or if the category is not found in the database map.
        """
        vector_db = self.db_map[category]
        self.load_vectorstore(vector_db = vector_db, documents = documents)
        self.app_logger.info(f"stored data successfully....")
        return "Success"