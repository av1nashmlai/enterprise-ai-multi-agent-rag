"""
Task 8: Agent Tools Module.
Defines explicit tools used by the autonomous agents in the decision support pipeline:
1. VectorSearchTool: Semantic similarity search across multi-format vector embeddings.
2. DocumentMetadataTool: Inspects ingested document catalog, formats, and chunk metadata.
3. HallucinationAuditTool: Evaluates draft responses against ground-truth retrieved context.
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.vectorstore.chroma_manager import VectorStoreManager


class VectorSearchTool:
    """Tool for retrieving semantically relevant document chunks from the vector database."""

    def __init__(self, vs_manager: VectorStoreManager):
        self.vs_manager = vs_manager

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Executes similarity search on the knowledge base."""
        return self.vs_manager.retrieve_relevant_chunks(query=query, k=k)


class DocumentMetadataTool:
    """Tool for extracting catalog information and provenance metadata from retrieved documents."""

    @staticmethod
    def extract_provenance(docs: List[Document]) -> List[Dict[str, Any]]:
        """Extracts unique source files, formats, and location pointers."""
        provenance = []
        seen = set()
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("location"))
            if key not in seen:
                seen.add(key)
                provenance.append({
                    "source": d.metadata.get("source", "Unknown"),
                    "type": d.metadata.get("type", "Document"),
                    "location": d.metadata.get("location", "N/A"),
                })
        return provenance


class HallucinationAuditTool:
    """Tool for verifying whether assertions in a draft response are grounded in source context."""

    @staticmethod
    def audit_grounding(context: str, answer: str) -> Dict[str, Any]:
        """Performs a deterministic heuristic check to assist the Validator Agent."""
        context_words = set(context.lower().split())
        answer_words = answer.lower().split()
        
        # Check if numbers/figures in answer are present in the context
        import re
        numbers_in_answer = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer)
        missing_numbers = [num for num in numbers_in_answer if num not in context]
        
        is_strictly_grounded = len(missing_numbers) == 0
        return {
            "missing_figures": missing_numbers,
            "has_unverified_numbers": not is_strictly_grounded,
            "total_assertions_checked": len(numbers_in_answer)
        }
