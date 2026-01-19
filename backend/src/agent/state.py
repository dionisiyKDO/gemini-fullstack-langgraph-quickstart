from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Annotated
from typing import TypedDict, List

from langgraph.graph import add_messages
import operator


class DocumentChunkState(TypedDict):
    """A chunk of text from a local document."""
    header_id: str          # e.g., "config.md > Setup > Installation"
    content: str
    file_path: str
    relevance_score: float  # BM25 score


class OverallState(TypedDict):
    messages: Annotated[List, add_messages]
    search_queries: List[str]
    retrieved_chunks: Annotated[List[DocumentChunkState], operator.add]  # Top-k chunks from BM25
    source_files: Annotated[List[str], operator.add]
    final_answer: str


class QueryGenerationState(TypedDict):
    """State after query generation step."""
    search_queries: list[str]
    query_rationale: str


class RetrievalState(TypedDict):
    """State for individual retrieval operations. Used when fanning out multiple queries in parallel."""
    query: str
    query_id: int
    chunks: List[DocumentChunkState]
