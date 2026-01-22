from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    """Multiple search queries to cover different aspects of the question."""
    queries: List[str] = Field(
        description="List of search queries to retrieve relevant documentation chunks."
    )
    rationale: str = Field(
        description="Brief explanation of the search strategy."
    )


class DocumentAnswer(BaseModel):
    """Structured answer with citations to source documents. This is what the final LLM call produces."""
    answer: str = Field(
        description="Complete answer to the user's question, with inline citations like [1], [2]."
    )
    citations: List[str] = Field(
        description="List of file paths referenced in the answer, in order of citation number."
    )
    confidence: str = Field(
        description="'high' if docs fully answer question, 'medium' if partial, 'low' if minimal info found."
    )
