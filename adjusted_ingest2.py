import os
import warnings
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

warnings.simplefilter("ignore")

def create_vector_database_from_url(url: str, db_dir: str):
    """
    Creates a vector database using the provided website URL with proper collection handling.
    """
    try:
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(data)

        embeddings = OllamaEmbeddings(model="mistral")
        
        # Create a new collection with a fixed name
        vector_database = Chroma(
            collection_name="document_collection",
            embedding_function=embeddings,
            persist_directory=db_dir
        )
        
        # Add documents to the collection
        vector_database.add_documents(docs)
        vector_database.persist()
        
        print(f"Vector database created and stored at: {db_dir}")
        return vector_database
    except Exception as e:
        print(f"Error creating vector database from URL: {e}")
        raise e

def create_vector_database_from_pdf(file_path: str, db_dir: str):
    """
    Creates a vector database using the provided PDF file with proper collection handling.
    """
    try:
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        pdf_loader = PyPDFLoader(file_path)
        loaded_documents = pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunked_documents = text_splitter.split_documents(loaded_documents)

        embeddings = OllamaEmbeddings(model="mistral")
        
        # Create a new collection with a fixed name
        vector_database = Chroma(
            collection_name="document_collection",
            embedding_function=embeddings,
            persist_directory=db_dir
        )
        
        # Add documents to the collection
        vector_database.add_documents(chunked_documents)
        vector_database.persist()
        
        print(f"Vector database created and stored at: {db_dir}")
        return vector_database
    except Exception as e:
        print(f"Error creating vector database from PDF: {e}")
        raise e