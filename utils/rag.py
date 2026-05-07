"""
utils/rag.py
RAG (Retrieval-Augmented Generation) utilities for knowledge base.
"""

import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import faiss
import pickle


VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_document(file_path: str) -> List[Document]:
    """Load a document from a file."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        # Try as text file
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
    
    return loader.load()


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


class LocalEmbeddings:
    """Local embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model only when needed."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return self.model.encode([text])[0].tolist()


def create_vector_store(documents: List[Document]) -> FAISS:
    """Create a FAISS vector store from documents."""
    embeddings = LocalEmbeddings()
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save to disk
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_DB_PATH)
    
    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """Load the vector store from disk."""
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    
    try:
        embeddings = LocalEmbeddings()
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def add_documents_to_store(documents: List[Document]):
    """Add documents to the existing vector store."""
    embeddings = LocalEmbeddings()
    
    # Load existing store or create new
    vector_store = load_vector_store()
    if vector_store is None:
        vector_store = create_vector_store(documents)
    else:
        vector_store.add_documents(documents)
        vector_store.save_local(VECTOR_DB_PATH)


def search_knowledge_base(query: str, k: int = 3) -> List[Document]:
    """Search the knowledge base for relevant documents."""
    vector_store = load_vector_store()
    if vector_store is None:
        return []
    
    embeddings = LocalEmbeddings()
    results = vector_store.similarity_search_by_vector(
        embeddings.embed_query(query),
        k=k
    )
    return results


def clear_knowledge_base():
    """Clear the knowledge base."""
    import shutil
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
