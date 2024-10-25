import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from typing import Dict
import base64
import httpx
import asyncio
import uuid
from Utilities import Utilities
from logs import get_app_logger, get_error_logger
from langchain_core.documents.base import Document
import prompts

load_dotenv()

class ImageHandling:
    """
    A class used to handle image processing tasks.

    Attributes:
    ----------
    app_logger : logger
        Application logger instance.
    error_logger : logger
        Error logger instance.
    OPENAI_API_KEY : str
        OpenAI API key.
    GENIE_ACCESS_TOKEN : str
        Genie access token.
    llm : object
        Language model instance.

    Methods:
    -------
    __init__()
        Initializes the ImageHandling instance.
    describe_image(encoded_image, image_path=None)
        Describes the given image using the language model.
    read_image_url(image_content, access_token, user_id)
        Reads the image from the given URL and saves it to a file.
    process_uploaded_image(image_content, access_token, user_id)
        Processes the uploaded image and generates its description.
    process_image_pdf(image_path)
        Processes the image from the given PDF file and generates its description.
    run(image_paths=None, pagenumber=None, filename=None, user_id=None, image_names=None, from_pdf=None, image_content=None, access_token=None)
        Runs the image processing tasks.
    """
    
    def __init__(self):
        """
        Initializes the ImageHandling instance.

        Initializes the application logger, error logger, OpenAI API key, Genie access token, and language model instance.
        """
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        self.llm = Utilities().llm()
    
    async def describe_image(self, encoded_image, image_path = None):
        """
        Describes the given image using the language model.

        Args:
        ----
        encoded_image : str
            Base64 encoded image string.
        image_path : str, optional
            Path to the image file (default is None).

        Returns:
        -------
        str
            Image description.
        """
        try:
            messages = [
                {"role" : "system", "content" : prompts.IMAGE_DESCRIPTION_PROMPT},
                {"role" : "user", "content" : [
                    {"type": "text", "text" : "extract the information from the image (text and everything), with same structure present in image and give me a summary as well. Do not miss anything"},
                    {"type" : "image_url",
                    "image_url" : {
                        "url" : f"data:image/png;base64,{encoded_image}",
                            },
                        }
                    ]
                }
            ]
            response = await self.llm.ainvoke(messages)
            image_description = response.content
            self.app_logger.info(f"Got the image description...")
            return image_description
        except Exception as e:
            self.error_logger.error(f"Error processing image {image_path}: {str(e)}")
            raise 
    
    async def read_image_url(self, image_content: Dict, access_token: str, user_id):
        """
        Reads the image from the given URL and saves it to a file.

        Args:
        ----
        image_content : Dict
            Dictionary containing image content information.
        access_token : str
            Access token for authentication.
        user_id : str
            User ID.

        Returns:
        -------
        tuple
            Image instance and image filename.
        """
        output_folder = f"/export/home/saallam/filehandling/InputFiles/uploaded_images/{user_id}" 
        image_filename = f"{uuid.uuid3(uuid.NAMESPACE_URL, image_content["downloadUrl"])}.png"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image_path = f"{output_folder}/{image_filename}"
        async with httpx.AsyncClient() as client:
            if image_content["contentType"] == "application/vnd.microsoft.teams.file.download.info":
                try: 
                    response = await client.get(image_content["downloadUrl"])
                    self.app_logger.info(f"Image response (application/vn.d.microsoft) => {response.status_code}")
                    image = Image.open(BytesIO(response.content))
                except Exception as e:
                    self.app_logger.info(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                    raise 
            if image_content["contentType"] == "image/*": # smba.trafficmanager.net
                headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json;odata=verbose"
                    }
                try:
                    response = await client.get(image_content["contentUrl"], headers = headers)
                    self.app_logger.info(f"Image response (image/*) => {response.status_code}")
                    image = Image.open(BytesIO(response.content))
                except Exception as e:
                    self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                    raise 
            try:
                image.save(image_path, quality=100, format = "PNG")
                self.app_logger.info(f"image saved, {image_path=}")
                return image, image_filename
            except Exception as e:
                self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                raise
    
    async def process_uploaded_image(self, image_content, access_token, user_id):
        """
        Processes the uploaded image and generates its description.

        Args:
        ----
        image_content : Dict
            Dictionary containing image content information.
        access_token : str
            Access token for authentication.
        user_id : str
            User ID.

        Returns:
        -------
        tuple
            Image description and image filename.
        """
        image, image_filename = await self.read_image_url(image_content, access_token, user_id)
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_description = f"The Image description of file {image_filename}:\n\n"
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
        image_description += await self.describe_image(encoded_image)
        return image_description, image_filename
                    
    async def process_image_pdf(self, image_path):
        """
        Processes the image from the given PDF file and generates its description.

        Args:
        ----
        image_path : str
            Path to the PDF file.

        Returns:
        -------
        str
            Image description.
        """
        try:
            image = Image.open(image_path)
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_description = await self.describe_image(encoded_image, image_path)
            return image_description
        except Exception as e:
            self.app_logger.info(f"Error processing image {image_path}: {str(e)}")
            return None, None, None

    async def run(self, image_paths = None, pagenumber = None, filename = None, user_id = None, image_names = None, from_pdf = None, image_content = None, access_token = None):
        """
        Runs the image processing tasks.

        Args:
        ----
        image_paths : list, optional
            List of image paths (default is None).
        pagenumber : int, optional
            Page number (default is None).
        filename : str, optional
            Filename (default is None).
        user_id : str, optional
            User ID (default is None).
        image_names : list, optional
            List of image names (default is None).
        from_pdf : bool, optional
            Flag indicating whether the images are from a PDF file (default is None).
        image_content : Dict, optional
            Dictionary containing image content information (default is None).
        access_token : str, optional
            Access token for authentication (default is None).

        Returns:
        -------
        tuple
            List of documents, image description, and image filename.
        """
        image_filename = None
        if from_pdf:
            tasks = [self.process_image_pdf(image_path) for image_path in image_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out any errors
            results = [result for result in results if not isinstance(result, Exception)]
            image_description = "\n\n".join([f"Image {i}\n{result}\n" for i, result in enumerate(results)])
            documents = [Document(metadata={"source": filename, "Page": pagenumber + 1, "Image": image_names[i], "userid": user_id, "summary": "no"}, page_content=result) for i, result in enumerate(results)]
        else:
            image_description, image_filename = await self.process_uploaded_image(image_content, access_token, user_id)
            documents = [Document(metadata={"source": image_filename, "type": "png", "userid": user_id}, page_content=image_description)]
        
        return documents, image_description, image_filename  