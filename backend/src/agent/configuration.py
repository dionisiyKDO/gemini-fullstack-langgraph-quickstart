import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


BASE_DIR = Path(__file__).parent.parent.parent.resolve() # gemini-fullstack-langgraph-quickstart/backend
DATA_DIR = BASE_DIR / "data"


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    answer_model: str = Field(
        default="llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )
    
    local_filepath: str = Field(
        default=str(DATA_DIR),
        metadata={
            "description": "Path to local directory containing documentation files."
        },
    )

    top_k_chunks: int = Field(
        default=5,
        metadata={
            "description": "Number of top-ranked chunks to retrieve per query (BM25)."
        },
    )
    
    max_tokens: int = Field(
        default=20_000,
        metadata={
            "description": "Maximum tokens to include in context for final answer generation."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
