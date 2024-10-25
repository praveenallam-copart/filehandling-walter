from fastapi import FastAPI
from logs import get_app_logger, get_error_logger, setup_logging
from fastapi.middleware.cors import CORSMiddleware
import sys
from pydantic import BaseModel
from typing import Union, List, Dict
from DataUploading.ImageHandling import ImageHandling
from dependencies import Dependencies
from csvHandling import CSVHandling
from DataUploading.pdfAdding import PdfUploading
from DataUploading.VectorDBStoring import VectorDBStroing
from DataRetrieval.Retrieval import Retrieval
# from llmRouter import llm_router
from langchain_core.messages import HumanMessage
from langchain_core.documents.base import Document
from dependencies import Dependencies
from llm_routers.KnowledgeRouter import KnowledgeRouter
from llm_routers.QueryTransformationRouter import QueryTransformationRouter
from query_transformations.Decomposition import Decomposition
from query_transformations.MultiQuery import MultiQuery
from query_transformations.StepBack import StepBack

class ImageContents(BaseModel):
    image_content: Dict
    access_token: str
    user_id: str

class CSVContents(BaseModel):
    csvUrl: str
    name: str

class ChatCompletion(BaseModel):
    text: Union[str, None] = None
    image_description: Union[str, None] =  None

class UploadingFiles(BaseModel):
    downloadUrl: Union[str, None] =  None
    name: Union[str, None] = None
    message: Union[str, None] = None
    user_id: Union[str]

class InfoRetrievals(BaseModel):
    query: Union[str, List[str]]
    name: Union[str]
    userid: str

class StoreData(BaseModel):
    documents: List[Document]
    category: str
    
class Completion(BaseModel):
    query: str
    name: Union[str, None] = None
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

setup_logging()
app_logger = get_app_logger()
error_logger = get_error_logger()
history, chat_history = Dependencies().history()
file_map = Dependencies().get_file_map()
llm = Dependencies().get_model(llm_service = "groq", model = "groq/llama-3.1-70b-versatile")
csv_agent, refined_summary = CSVHandling().csv_agent()

@app.get("/time")     
def time():
    return {"time" : Dependencies().get_timestamp()}

@app.post("/image_description")
async def image_description(imageContents: ImageContents):
    try:
        documents, image_description, image_filename = await ImageHandling().run(image_content = imageContents.image_content, access_token = imageContents.access_token, user_id = imageContents.user_id)
        file_map[image_filename] = "Images"
        Dependencies().write_file_map(file_map)
        history.extend([HumanMessage(content=f"Upload the image: {image_filename}"), image_description])
        chat_history.extend([{"role" : "user", "content" : f"Upload the image: {image_filename}"}, {"role" : "assistant", "content" : image_description}])
        return {"image_description" : image_description, "documents" : documents, "image_filename" : image_filename}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise

@app.post("/csvHandling")
async def csv_handling(csvContents: CSVContents):
    try:
        name = csvContents.name.split(".")[0]
        await CSVHandling().read_store_csv(csvContents.csvUrl, name)
        csv_agent, refined_summary = CSVHandling().csv_agent(csvContents.name)
        history.extend([HumanMessage(content=f"Uploading a new file name = {csvContents.name}"), refined_summary])
        chat_history.extend([{"role" : "user", "content" : f"Uploading a new file name = {csvContents.name}"}, {"role" : "assistant", "content" : refined_summary}])
        return {"status" : 200, "response" : f"Read and stored the {name=}"}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise

@app.post("/queryCsv")
def query_csv(completion: Completion):
    try:
        query = f"{completion.query}\nuse {completion.name} for reference"
        response = CSVHandling().query_agent(query, csv_agent)
        history.extend([HumanMessage(content=query), response])
        chat_history.extend([{"role" : "user", "content" : query}, {"role" : "assistant", "content" : response}])
        return {"status" : 200, "response" : response}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise


   
@app.get("/sys-version")
def sys_version():
    return {"response" : sys.version}

@app.post("/UploadingFile")
async def uploading_file(uploadingFile: UploadingFiles):
    # filepath = "/export/home/saallam/image_handling/walter-image-handling/pdfs/All you need is attention.pdf"
    name = uploadingFile.name.split(".")[0]
    documents, refined_summary, category = await PdfUploading().run(uploadingFile.downloadUrl, name, uploadingFile.user_id)
    
    # category = await PdfUploading().category(refined_summary)
    history.extend([HumanMessage(content=f"{uploadingFile.message}\n {uploadingFile.name}"), refined_summary])
    chat_history.extend([{"role" : "user", "content" : f"Uploading a new file\n{uploadingFile.name}"}, {"role" : "assistant", "content" : refined_summary}])
    Dependencies().write_chat_history(chat_history)
    file_map[name] = category
    Dependencies().write_file_map(file_map)
    return {"documents" : documents, "summary" : refined_summary, "category" : category}

@app.post("/StoreData")
def store_data(storeData: StoreData):
    try:
        VectorDBStroing().run(storeData.documents, storeData.category)
        return {"Status" : "Success 200 OK!"}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise

@app.post("/retrieve")
def pdf_retrieval(infoRetrieval: InfoRetrievals):
    name = infoRetrieval.name.split(".")[0]
    category = file_map[name]
    response, retrieved_documents, question_context_pairs = Retrieval().run(infoRetrieval.query, category, infoRetrieval.name, infoRetrieval.userid)
    history.extend([HumanMessage(content=infoRetrieval.query), response])
    chat_history.extend([{"role" : "user", "content" : infoRetrieval.query}, {"role" : "assistant", "content" : response}])
    return {"response" : response, "knowledge_used" : retrieved_documents, "question_context_pairs" : question_context_pairs, "Status" : 200}

@app.post("/knowledgeRouter")
def llm_router_completion(chatCompletion: ChatCompletion):
    response, core_query = KnowledgeRouter().run(chatCompletion.text, history)
    if "ConversationalResponse" in str(type(response)):
        history.extend([HumanMessage(content=chatCompletion.text), response.response])
        chat_history.extend([{"role" : "user", "content" : chatCompletion.text}, {"role" : "assistant", "content" : response.response}])
    return {"response" : response, "core_query" : core_query, "response_type" : str(type(response))}

@app.post("/queryTransformationRouter")
def llm_router_completion(chatCompletion: ChatCompletion):
    response = QueryTransformationRouter().run(chatCompletion.text)
    transformerd_query = eval(f"{response.transformation}()").run(chatCompletion.text) if (response.transformation != "CoreMeaning" and response.transformation != "None") else None
    print(transformerd_query)
    return {"response" : response, "transformerd_query" : transformerd_query, "response_type" : str(type(response))}

@app.get("/chatHistory")
def get_chat_history():
    return {"response" : chat_history}

@app.get("/updateHistory")
def updateHistory():
    Dependencies().write_chat_history(chat_history)
    return {"response" : {"status" : 200, "chat_history" : chat_history}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host = "0.0.0.0", port = 8000, reload = True)