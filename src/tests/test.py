from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from src.core.llm.llm_openai import LLMOpenAI
from src.core.vector_store.vs_azure import AzureVectorStore

from src.graph.state.graph_state import TaskRoute

llmOpenAI = LLMOpenAI()
embeddings = llmOpenAI._get_embedding()
vector_store_model = AzureVectorStore(embeddings=embeddings)
vector_store = vector_store_model._get_vector_store()

# pdf_path = "data/interim/test.pdf"

# try:
#     # loader = TextLoader("data/interim/test.txt", encoding="utf-8")
#     loader = PyPDFLoader(pdf_path)

#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     docs = text_splitter.split_documents(documents)

#     print(f"PDF cargado y dividido en {len(docs)} fragmentos.")
# except Exception as e:
#     print(f"Error loading document: {e}")
#     raise


# # vector_store.add_documents(documents=docs)

# vector_store.add_texts(texts=[doc.page_content for doc in docs], metadatas=[{"title": "Test Document", "source": pdf_path} for _ in docs])

# print(f"Documents added to Azure Vector Store successfully. Total documents: {len(docs)}")

# user_question = "¿Qué estudió Ali Campos?"
user_question = "Cuéntame un chiste sobre programadores."

# Perform a hybrid search using the hybrid_search method
docs = vector_store.hybrid_search(
    query=user_question, k=3
)

# print(docs[0].page_content)
# print(docs[0].metadata)


# chat_bot = llmOpenAI._get_chat_response(
#     system_message="Eres un asistente útil.",
#     developer_message="Proporciona respuestas claras y concisas.",
#     user_message=f"Basado en la siguiente información: {docs[0].page_content}, responde a la pregunta: {user_question}",
# )

# print("\n" + chat_bot.response + "\n")


chat_response = llmOpenAI._get_chat_response(
                system_message="Eres un descriminador de preguntas.",
                developer_message="Retornar true si la pregunta es educativa, false en caso contrario.",
                user_message=f"{user_question}",
                text_format=TaskRoute
            )

print(f"\n{chat_response.response}\n")
