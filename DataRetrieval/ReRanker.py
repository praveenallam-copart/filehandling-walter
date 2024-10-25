from typing import Dict, List
from haystack import component, Document
from haystack.components.rankers import TransformersSimilarityRanker
from logs import get_app_logger, get_error_logger
import os

@component
class ReRankerComponent:
    """
    A component that re-ranks the retrieved documents based on their similarity to the query.

    Attributes:
        ranker (TransformersSimilarityRanker): The ranker model used to re-rank the documents.
        top_k (int): The number of top documents to return.
        documents (List[Document]): The list of re-ranked documents.
    """

    def __init__(self, ranker:TransformersSimilarityRanker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-12-v2"), top_k: int = 3):
        """
        Initializes the ReRankerComponent.

        Args:
            ranker (TransformersSimilarityRanker, optional): The ranker model to use. Defaults to TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-12-v2").
            top_k (int, optional): The number of top documents to return. Defaults to 3.
        """
        
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.HF_API_TOKEN = os.getenv("HF_API_TOKEN")
        self.ranker = ranker
        self.ranker.warm_up()
        self.documents = []
        self.top_k = top_k
        self.app_logger.info(f"Reranker {ranker=} initialised...")

    @component.output_types(documents = List[Document])
    def run(self, question_context_pairs: List[Dict], top_k: int = None):
        """
        Re-ranks the retrieved documents based on their similarity to the query.

        Args:
            question_context_pairs (List[Dict]): A list of dictionaries containing the query and the retrieved documents.
            top_k (int, optional): The number of top documents to return. Defaults to None.

        Returns:
            Dict: A dictionary containing the re-ranked documents.
        """
        if top_k != None:
            self.top_k = top_k
        try:
            self.app_logger.info(f"reranking process started...")
            for pair in question_context_pairs:
                query, docs = pair.items()
                self.app_logger.info(f"reranking for {query=}")
                result = self.ranker.run(query = query[1], documents=docs[1], top_k = self.top_k)
                self.documents.extend(result["documents"])
            self.app_logger.info(f"reranking done...!!")
            return {"documents" : self.documents}
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
