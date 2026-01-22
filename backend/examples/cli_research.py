# cli_research.py
import argparse
import os
from langchain_core.messages import HumanMessage
from agent.graph import graph
from agent.retrieval import Retrieval


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Search local documentation with LangGraph + BM25")
    parser.add_argument("question", help="Question to answer from local documentation")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to directory containing documentation (.md, .txt, .ipynb files)",
    )
    
    # Optional search configuration
    parser.add_argument(
        "--num-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40000,
        help="Maximum tokens to include in final answer context",
    )
    
    
    # Model selection
    parser.add_argument(
        "--query-model",
        default="llama-3.3-70b-versatile",
        help="Model for the query generation",
    )
    parser.add_argument(
        "--answer-model",
        default="llama-3.3-70b-versatile",
        help="Model for the final answer",
    )
    
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist")
        return
    
    # Initialize Retrieval singletong and load files from local directory
    engine = Retrieval()
    engine.load_data(args.dir)
    
    # Build initial state
    state = {
        "messages": [HumanMessage(content=args.question)],
    }
    
    config = {
        "configurable": {
            "local_filepath": args.dir,
            "number_of_queries": args.num_queries,
            "top_k_chunks": args.top_k,
            "max_tokens": args.max_tokens,
            "query_generator_model": args.query_model,
            "answer_model": args.answer_model,
        }
    }
    
    # Run the graph
    result = graph.invoke(state, config=config)
    
    # Extract and print the answer
    messages = result.get("messages", [])
    if messages:
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(messages[-1].content)
        print()
    else:
        print("No answer generated. Check your documents and query.")


if __name__ == "__main__":
    main()