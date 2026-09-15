"""
Document indexing + retrieval backed by Chroma.
"""

import os
from functools import lru_cache
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def vectorstore() -> Chroma:
    """
    Open the persistent Chroma collection.

    Cached: every call used to build a fresh embeddings client and reopen the
    collection from disk, which happens on every single retrieval.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
    )


def index_docs(file_path: str, file_id: int) -> bool:
    """Load, split, and embed a document into the collection."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()
    if not documents:
        raise ValueError("No readable text found in this file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Document produced no text chunks.")

    for chunk in chunks:
        chunk.metadata["file_id"] = file_id

    vectorstore().add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {file_path} (file_id={file_id})")
    return True


def retrieve_docs(query: str, k: int = 4) -> List[Document]:
    """Similarity search across every indexed document."""
    return vectorstore().similarity_search(query, k=k)


def delete_document(file_id: int) -> bool:
    """Remove every chunk belonging to one uploaded file."""
    store = vectorstore()
    # Resolve the ids first: passing only a `where` filter is not reliably
    # supported across chromadb versions, but deleting by explicit id is.
    existing = store.get(where={"file_id": file_id}, include=[])
    ids = existing.get("ids", [])
    if ids:
        store.delete(ids=ids)
    print(f"Deleted {len(ids)} chunks for file_id={file_id}")
    return True


def has_documents() -> bool:
    """True if anything at all has been indexed — used for friendlier errors."""
    try:
        return vectorstore()._collection.count() > 0
    except Exception:
        return False
