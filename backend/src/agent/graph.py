import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from langchain_groq import ChatGroq

from agent.configuration import Configuration
from agent.utils import get_research_topic
from agent.retrieval import Retrieval
from agent.tools_and_schemas import (
    SearchQueryList, 
    DocumentAnswer,
)
from agent.state import (
    OverallState,
    QueryGenerationState,
    RetrievalState,
)
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    answer_instructions,
)

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")


# Nodes
def generate_queries(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Groq (Llama 3.3) to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Groq LLM
    llm = ChatGroq(
        model=configurable.query_generator_model,
        temperature=0.7,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_queries": result.queries}


def fan_out_retrieval(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the retrieve_chunks node.
    
    This is used to spawn n number of retrieve_chunks nodes, one for each search query.
    """
    return [
        Send("retrieve_chunks", {"query": query, "query_id": idx})
        for idx, query in enumerate(state["search_queries"])
    ]


def retrieve_chunks(state: RetrievalState, config: RunnableConfig) -> OverallState:
    """Retrieve top-k relevant chunks using BM25.
    
    Args:
        state: Contains single query to search for
        config: Contains file path and top_k settings
        
    Returns:
        Updated state with retrieved chunks and source files
    """
    configurable = Configuration.from_runnable_config(config)
    
    retrieval = Retrieval()
    retrieved, source_files = retrieval.search(state["query"], top_k=configurable.top_k_chunks)
    
    # print(f"[Query {state['query_id']}] '{state['query_id']}' → {len(retrieved)} chunks from {len(source_files)} files")
    
    return {
        "retrieved_chunks": retrieved,
        "source_files": list(source_files),
    }


def generate_answer(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that synthesizes final answer from all retrieved chunks.
    
    Args:
        state: Contains all retrieved chunks from all queries
        config: Contains LLM model selection and max tokens limit
        
    Returns:
        Final answer and citation list
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Deduplicate chunks by header_id
    seen_chunks = {}
    for chunk in state["retrieved_chunks"]:
        key = chunk["header_id"]
        if key not in seen_chunks or chunk["relevance_score"] > seen_chunks[key]["relevance_score"]:
            seen_chunks[key] = chunk
    
    sorted_chunks = sorted(
        seen_chunks.values(),
        key=lambda c: c["relevance_score"],
        reverse=True
    )
    
    # Build context with rough token estimate
    context_parts = []
    total_tokens = 0
    max_tokens = configurable.max_tokens
    
    for i, chunk in enumerate(sorted_chunks):
        chunk_text = f"\n\n--- SOURCE [{i+1}]: {chunk['header_id']} ---\n{chunk['content']}"
        chunk_tokens = len(chunk_text.split()) * 1.3  # Rough estimate (1 word = 1.3 tokens)
        
        if total_tokens + chunk_tokens > max_tokens:
            break
        
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    context = "".join(context_parts)
    
    # Generate answer with citations
    llm = ChatGroq(
        model=configurable.answer_model,
        temperature=0,  # Deterministic for factual answers
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    structured_llm = llm.with_structured_output(DocumentAnswer)
    
    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    
    prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        context=context,
    )
    
    try:
        result = structured_llm.invoke(prompt)
        
        # Format final answer with sources
        final_answer = result.answer
        if result.citations:
            final_answer += "\n\n**Sources:**\n"
            for i, citation in enumerate(result.citations, 1):
                final_answer += f"{i}. {citation}\n"
        
        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
        }
        
    except Exception as e:
        error_msg = f"Error generating answer: {e}\n\nFound relevant info in: {state['source_files']}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "final_answer": error_msg,
        }

# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_queries", generate_queries)
builder.add_node("retrieve_chunks", retrieve_chunks)
builder.add_node("generate_answer", generate_answer)

# Define flow
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges("generate_queries", fan_out_retrieval, ["retrieve_chunks"])
builder.add_edge("retrieve_chunks", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile(name="local-doc-search-agent")
