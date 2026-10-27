"""
Tasks 5 & 6: Vector-Based Knowledge Store & Intelligent Document Retrieval.
Manages embeddings and similarity search using ChromaDB.
Supports:
- NVIDIA NIM Embeddings (NVIDIAEmbeddings)
- OpenAI Embeddings (OpenAIEmbeddings)
- OpenRouter Embeddings (OpenAI-compatible)
- Built-in Local ONNX Embeddings (all-MiniLM-L6-v2 via ChromaDB DefaultEmbeddingFunction)
  for Groq, Gemini, OpenRouter, or offline/zero-cost fallback!
"""
import os
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from src.config import Config

# Verified active embedding models from https://integrate.api.nvidia.com/v1/models
NVIDIA_ACTIVE_EMBED_MODELS = [
    "nvidia/nemotron-3-embed-1b",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia/embed-qa-4",
    "snowflake/arctic-embed-l",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
]

DEPRECATED_NVIDIA_EMBED_MODELS = {
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/llama-nemotron-embed-1b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "nvidia/llama-3.2-nemoretriever-300m-embed-v2",
    "baai/bge-m3",
}


class LocalONNXEmbeddings(Embeddings):
    """
    LangChain Embeddings wrapper around ChromaDB's built-in DefaultEmbeddingFunction
    (all-MiniLM-L6-v2 via ONNX Runtime). Works 100% locally with zero API key required,
    enabling chat-only providers like Groq, Gemini, and OpenRouter to index documents seamlessly.
    """

    def __init__(self):
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        raw_vecs = self._ef(texts)
        return [[float(x) for x in v] for v in raw_vecs]

    def embed_query(self, text: str) -> List[float]:
        raw_vecs = self._ef([text])
        return [float(x) for x in raw_vecs[0]]


class VectorStoreManager:
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        provider_override: str = "auto",
    ):
        self.active_key = (api_key or Config.DEFAULT_API_KEY or "").strip()
        self.provider = Config.detect_provider(
            self.active_key, override_provider=provider_override
        )
        os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)

        # Determine requested embedding model
        req_embed = (embedding_model or "").strip()

        if req_embed.startswith("local/") or self.provider in ("groq", "gemini") or not self.active_key:
            self._init_local_store()
        elif self.provider == "openrouter":
            if req_embed and not req_embed.startswith("local/"):
                self._init_openrouter_store(req_embed)
            else:
                self._init_local_store()
        elif self.provider == "nvidia":
            model_name = req_embed or Config.NVIDIA_EMBEDDING_MODEL
            if model_name in DEPRECATED_NVIDIA_EMBED_MODELS:
                model_name = "nvidia/nemotron-3-embed-1b"
            self._init_nvidia_store(model_name)
        else:
            model_name = req_embed or Config.EMBEDDING_MODEL
            self._init_openai_store(model_name)

    def _init_local_store(self):
        self.model_name = "local/all-MiniLM-L6-v2 (ONNX)"
        self.collection_name = f"{Config.COLLECTION_NAME}_local_onnx"
        self.embeddings = LocalONNXEmbeddings()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )

    def _init_openrouter_store(self, model_name: str):
        self.model_name = model_name
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:36]
        self.collection_name = f"{Config.COLLECTION_NAME}_or_{slug}"
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=self.active_key,
            openai_api_base=Config.PROVIDER_BASE_URLS["openrouter"],
            check_ctx_length=False,
        )
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )

    def _init_openai_store(self, model_name: str):
        self.model_name = model_name
        self.collection_name = f"{Config.COLLECTION_NAME}_openai"
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=self.active_key,
        )
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )

    def _init_nvidia_store(self, model_name: str):
        self.model_name = model_name
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:36]
        self.collection_name = f"{Config.COLLECTION_NAME}_{slug}"
        self.embeddings = NVIDIAEmbeddings(
            model=model_name,
            nvidia_api_key=self.active_key,
            truncate="END",
        )
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )

    def get_chunk_count(self) -> int:
        """Returns the number of chunks currently indexed in the active collection."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def add_documents(self, chunks: List[Document]) -> int:
        """Task 5: Generate embeddings and store chunks in ChromaDB (with automatic cloud & local fallback)."""
        if not chunks:
            return 0
        try:
            self.vectorstore.add_documents(chunks)
            return len(chunks)
        except Exception as err:
            if self.provider == "nvidia":
                for candidate in NVIDIA_ACTIVE_EMBED_MODELS:
                    if candidate == self.model_name:
                        continue
                    try:
                        self._init_nvidia_store(candidate)
                        self.vectorstore.add_documents(chunks)
                        return len(chunks)
                    except Exception:
                        continue
            # Ultimate fail-safe fallback to built-in Local ONNX embeddings so ingestion never fails
            try:
                self._init_local_store()
                self.vectorstore.add_documents(chunks)
                return len(chunks)
            except Exception:
                raise err

    def retrieve_relevant_chunks(
        self, query: str, k: int = Config.TOP_K
    ) -> List[Document]:
        """Task 6: Retrieve top-k relevant document chunks using similarity search."""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as err:
            if self.provider == "nvidia":
                for candidate in NVIDIA_ACTIVE_EMBED_MODELS:
                    if candidate == self.model_name:
                        continue
                    try:
                        self._init_nvidia_store(candidate)
                        if self.get_chunk_count() > 0:
                            return self.vectorstore.similarity_search(query, k=k)
                    except Exception:
                        continue
            try:
                self._init_local_store()
                return self.vectorstore.similarity_search(query, k=k)
            except Exception:
                raise err

    def clear_store(self):
        """Resets the vector collection."""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )
