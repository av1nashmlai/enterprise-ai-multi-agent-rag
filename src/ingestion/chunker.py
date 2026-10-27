"""
Task 4: Prepare Data for Semantic Search.
Splits documents into overlapping semantic chunks while preserving citation metadata.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits raw documents into smaller chunks suitable for vector embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks
