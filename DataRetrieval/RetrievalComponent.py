from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from typing import Dict, List, Union
from logs import get_app_logger, get_error_logger
from haystack import component, Document

@component
class RetrievalComponent:
    """
    A component responsible for retrieving relevant documents from a document store 
    based on a given query. It uses a ChromaQueryTextRetriever for querying the 
    document store and supports filtering by source and userid.

    Attributes:
        retriever (ChromaQueryTextRetriever): The retriever used for querying the document store.
        question_context_pairs (List[Dict]): A list of dictionaries containing the question and 
            the retrieved documents for each query.
        documents (List[Document]): A list of all retrieved documents.
        top_k (int): The number of documents to retrieve for each query.
    """

    def __init__(self, retriever: ChromaQueryTextRetriever, top_k: int = 10):
        """
        Initializes the RetrievalComponent.

        Args:
            retriever (ChromaQueryTextRetriever): The retriever used for querying the document store.
            top_k (int, optional): The number of documents to retrieve for each query. Defaults to 10.
        """
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.retriever = retriever
        self.question_context_pairs = []
        self.documents = []
        self.top_k = top_k
        self.app_logger.info(f"Retriver {retriever=} initialised...")

    @component.output_types(question_context_pairs=List[Dict], documents = List[Document])
    def run(self, queries: Union[str, List[str]], name: str, userid: str, top_k: int = None):
        """
        Runs the retrieval process for the given queries.

        Args:
            queries (Union[str, List[str]]): The query or list of queries to retrieve documents for.
            name (str): The name to filter by.
            userid (str): The userid to filter by.
            top_k (int, optional): The number of documents to retrieve for each query. Defaults to None.

        Returns:
            Dict: A dictionary containing the question_context_pairs and the retrieved documents.
        """
        if type(queries) == str:
            queries = [queries]
        if top_k != None:
            self.top_k = top_k
        try:
            for query in queries:
                result = self.retriever.run(query = query, top_k = self.top_k, filters =  {'operator': 'OR',
                                                    'conditions': [
                                                            {'field': 'source', 'operator': '==', 'value': name},
                                                            {'field': 'userid', 'operator': '==', 'value': userid}
                                                        ]
                                                    }
                                            )
                self.documents.extend(result["documents"])
                self.question_context_pairs.append({"question": query, "documents": result["documents"]})
            self.app_logger.info(f"relevant information retrieved....")
            return {"question_context_pairs": self.question_context_pairs, "documents" : self.documents}
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
