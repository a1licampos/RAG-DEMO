import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)

load_dotenv()

class AzureVectorStore:
    
    def __init__(self, embeddings, index_name: str = "langchain-vector-demo"):
        try:
            embedding_function = embeddings.embed_query

            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=len(embedding_function("Text")),
                    vector_search_profile_name="myHnswProfile",
                ),
                SearchableField(
                    name="metadata",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                # Additional field to store the title
                SearchableField(
                    name="title",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                # Additional field for filtering on document source
                SimpleField(
                    name="source",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
            ]

            vector_store: AzureSearch = AzureSearch(
                azure_search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                azure_search_key = os.getenv("AZURE_AI_SEARCH_KEY"),
                index_name = index_name,
                embedding_function = embedding_function,
            )
    
        except Exception as e:
            logging.error(f"Error initializing AzureVectorStore: {e}")
            raise

        self.vector_store = vector_store
    

    def _get_vector_store(self):
        return self.vector_store


#    _____
#   ( \/ @\____
#   /           O
#  /   (_|||||_/
# /____/  |||
#       kimba