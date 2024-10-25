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
from Utilities import Utilities
import prompts

load_dotenv()

class PdfUploading:
    """
    A class responsible for uploading PDF documents, extracting text and images, 
    summarizing the content, and storing the information in a vector database.

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
        db_map (dict): A dictionary mapping categories to their corresponding vector stores.

    Methods:
        encode_image(image_path): Encodes an image to base64 format.
        describe_image(base64_image): Sends a request to an LLM model to extract information from an image.
        extract_images_text(filepath, output_file, name, user_id): Extracts text and images from a PDF document.
        text_splitter(): Splits the combined text into smaller chunks.
        load_vectorstore(vector_db): Stores the split documents into a vector database.
        store_data(downloadUrl, name, user_id): Downloads a PDF, processes it, and stores the extracted information.
        summary(input_info): Generates a summary for the given input text.
        category(refined_summary): Classifies the given refined summary into one of five categories.
    """
    
    def __init__(self):
        """
        Initializes the PdfUploading class.

        Attributes:
            app_logger (Logger): The application logger instance.
            error_logger (Logger): The error logger instance.
            OPENAI_API_KEY (str): The OpenAI API key.
            GENIE_ACCESS_TOKEN (str): The Genie access token.
            llm (LLM): The language model instance.
        """
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.llm = Utilities().llm()

        
    async def extract_images_text(self, filepath, name, user_id):
        """
            Extracts text and images from a PDF document and stores the text and images, while summarizing the content.
            
            Args:
                filepath (str): The path to the PDF file.
                name (str): The name of the file (without extension).
                user_id (str): The user ID of the user who uploaded the file.

            Raises:
                Exception: If an error occurs during extraction or image processing.
        """
        try:
            total_summary = ""
            name = name.split(".")[0]
            self.app_logger.info(f"file {name=}...")
            output_folder = f"/export/home/saallam/filehandling/InputFiles/extracted_images/{name}" 
            document = fitz.open(filepath)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self.app_logger.info(f"{output_folder=} created/ found...")
            self.combined_text = []
            for page_number in range(len(document)):
                image_names, image_paths = [], []
                self.app_logger.info(f"processing {page_number=}...")
                page = document.load_page(page_number)
                text = page.get_text()
                images = page.get_images(full = True)
                metadata = {"source" : f"{name}.pdf", "Page": page_number + 1, "Image": "No", "userid" : user_id, "summary" : "no"}
                page_info = Document(metadata = metadata, page_content = text)
                self.combined_text.append(page_info)
                # total_summary += await self.summary(text)
                if images:
                    self.app_logger.info(f"found images in {page_number=}...")
                    for image_index, image in enumerate(images):
                        self.app_logger.info(f"processing {image_index=}...")
                        xref = image[0]
                        base_image = document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_extension = base_image["ext"]
                        image_filename = f"{name}_page_{page_number+1}_image_{image_index+1}.{image_extension}"
                        image_filepath = os.path.join(output_folder, image_filename)
                        image_names.append(image_filename)
                        image_paths.append(image_filepath)  
                        with open(image_filepath, "wb") as file:
                            file.write(image_bytes)
                        self.app_logger.info(f"image was written into {image_filepath=}...")
                image_documents, image_content = await ImageHandling().run(image_paths = image_paths, pagenumber = page_number + 1, filename = name, user_id = user_id, image_names = image_names, from_pdf = True)
                self.combined_text.extend(image_documents)
                text = text + "\n\n" + image_content
                total_summary += await self.summary(text)
                
            self.refined_summary = await self.summary(total_summary)
            metadata = {"source" : f"{name}", "Page": None, "Image": "no", "userid" : user_id, "summary" : "yes"}
            page_info = Document(metadata = metadata, page_content = self.refined_summary)
            self.app_logger.info("Processing done ==> extract_images_text")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
    
    def text_splitter(self):
        """
            Splits the combined text into smaller chunks using a RecursiveCharacterTextSplitter.
            
            Raises:
                Exception: If an error occurs during the text splitting process.
        """
        # hyper parameter tuning for better chunk_size and chunk_overlap, custom text splitter 
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 700,
                chunk_overlap = 100,
                separators = ["\n\n", "\n"]
            )
            
            self.splits = text_splitter.split_documents(self.combined_text)
            self.app_logger.info("splitting done using recursive character text splitter...")
            self.app_logger.info(f"The lenght of documents (total number of pages) : {len(self.combined_text)}")
            self.app_logger.info(f"The lenght of splits (total number of pages after splits) : {len(self.splits)}")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise

    
        
    async def summary(self, text : str):
        """
            Generates a summary for the given input text using a language model.

            Args:
                text (str): The text content to be summarized.

            Returns:
                str: The generated summary.

            Raises:
                Exception: If an error occurs during the summarization process.
        """
        try:
            prompt = ChatPromptTemplate([
                ("system", prompts.SUMMARY_PROMPT),
                ("human", "input: {input}")
            ])

            summary_llm = prompt | self.llm | StrOutputParser()
            response = await summary_llm.ainvoke({"input" : text})
            self.app_logger.info(f"got the summary for the given text....")
            return response
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
            
    
    async def category(self, refined_summary: str):
        
        """
            Classifies the given refined summary into one of five categories: Education, Sports, Politics, Environment, or Others.

            The function sends the provided input (refined summary) to a language model (LLM) prompt, which evaluates the content and assigns a category based on predefined descriptions and examples. 
            The output will be a single category string, such as "Education", "Sports", "Politics", "Environment", or "Others". The model uses examples for each category to make its decision and ensures the output strictly matches one of the categories.

            Args:
                refined_summary (str): The refined summary text of the document.

            Returns:
                str: The category that the document belongs to. Possible values: "Education", "Sports", "Politics", "Environment", or "Others".

            Raises:
                Exception: If any error occurs during the classification process.
        """
        try:
            prompt = ChatPromptTemplate([
                ("system", prompts.CATEGORY_PROMPT),
                ("human", "input: {input}")
            ])

            category_llm = prompt | self.llm | StrOutputParser()
            category = await category_llm.ainvoke({"input" : refined_summary})
            self.app_logger.info(f"the given document is of {category=}")
            return category
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
    
    async def run(self, downloadUrl, name, user_id):
        """
        Downloads a PDF file from the provided URL, extracts text and images, 
        summarizes the content, and stores the extracted information in a vector database.

        Args:
            downloadUrl (str): The URL of the PDF file to be downloaded.
            name (str): The name of the file (without extension).
            user_id (str): The user ID associated with the document.

        Returns:
            tuple: A tuple containing the split documents, refined summary, and category.

        Raises:
            Exception: If an error occurs during the download, extraction, summarization, or storage process.
        """

        try:
            response = requests.get(downloadUrl)
            filepath = f"/export/home/saallam/filehandling/InputFiles/uploaded-pdfs/{name}.pdf"
            with open(filepath, 'wb') as pdf_file:
                pdf_file.write(response.content)
            self.app_logger.info(f"file {name=} is written...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
        await self.extract_images_text(filepath, "extracted_images", name, user_id)
        self.text_splitter()
        category = await self.category(self.refined_summary)
        
        return self.splits, self.refined_summary, category