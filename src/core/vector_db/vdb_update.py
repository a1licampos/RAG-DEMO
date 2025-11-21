import logging
from src.core.llm.llm_openai import LLMOpenAI
from src.core.vector_db.vdb_azure import AzureVectorStore
from langchain_text_splitters import CharacterTextSplitter
from src.core.database.azure_storage_blob import AzureStorageBlobDatabase

class UpdateVectorDB:
    
    def __init__(self, index_name: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Getting pdf documents from Azure Blob Storage
        self.azure_storage_blob_db = AzureStorageBlobDatabase()

        # Setting up embedding
        self.llmOpenAI = LLMOpenAI()
        self.embeddingOpenAI = self.llmOpenAI._get_embedding()

        # Setting up Azure Vector Store
        self.index_name = index_name
        self.azure_vector_store_model = AzureVectorStore(embedding=self.embeddingOpenAI, 
                                                         index_name=self.index_name)

        # Setting up text splitter
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)


    def _azure_upload_vector_store(self, name_folder: str):
        try:
            self.azure_vector_store_model._delete_vector_store_index()
            self.azure_vector_store_model._create_vector_store_index()

            docs_loaded = self.azure_storage_blob_db._load_pdfs_blob(name_folder=name_folder)

            if not docs_loaded:
                logging.warning(f"No se encontraron documentos en el folder '{name_folder}'")
                return
            
            docs = self.text_splitter.split_documents(docs_loaded)
            azure_vector_store = self.azure_vector_store_model._get_vector_store()

            azure_vector_store.add_texts(
                texts=[doc.page_content for doc in docs],
                metadatas = [{
                    "name": doc.metadata.get("name", ""),
                    "page_number": str(doc.metadata.get("page_number", "")),
                    "url": doc.metadata.get("url", ""),
                } for doc in docs])

            logging.info(f"Documents added to Azure Vector Store successfully. Total documents: {len(docs)}")


        except Exception as e:
            logging.error(f"Error uploading to Azure Vector Store: {e}")
            raise

#    _____
#   ( \/ @\____
#   /           O
#  /   (_|||||_/
# /____/  |||
#       kimba