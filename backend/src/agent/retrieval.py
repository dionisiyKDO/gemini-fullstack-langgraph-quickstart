import os
from typing import List
from pathlib import Path
from rank_bm25 import BM25Okapi
from agent.utils import get_all_file_paths, read_file_content, split_markdown_by_headers
from agent.state import DocumentChunkState

# Singleton Retrieval class for retrieving document chunks
class Retrieval:
    _instance = None
    
    def __new__(cls):
        """Ensure we only have one instance of this engine."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
            cls._instance.chunks = []
            cls._instance.bm25 = None
        return cls._instance

    def load_data(self, directory: str):
        """Load all files, chunks them, and builds the BM25 index. Runs ONCE at startup."""
        if self.initialized:
            return # Retrieval already initialized, skipping reload

        file_paths = get_all_file_paths(directory)
        
        self.chunks: List[DocumentChunkState] = []
        tokenized_corpus = []

        for file_path in file_paths:
            content = read_file_content(file_path)
            if not content:
                continue
                
            file_name = os.path.basename(file_path)
            file_chunks = split_markdown_by_headers(content, file_name)
            
            for chunk in file_chunks:
                # Store the full chunk object
                full_chunk = {
                    "header_id": chunk["header_id"],
                    "content": chunk["content"],
                    "file_path": file_path,
                    "relevance_score": 0.0 
                }
                self.chunks.append(full_chunk)
                
                # Tokenize for BM25 
                tokens = f"{chunk['header_id']} {chunk['content']}".lower().split()
                tokenized_corpus.append(tokens)

        if not self.chunks:
            raise Exception("No document chunks found in the specified directory.")

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.initialized = True

    def search(self, query: str, top_k: int = 5):
        """Query the pre-built index."""
        if not self.initialized or not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        retrieved = []
        source_files = set()
        
        for idx in top_indices:
            # Create a copy so we don't modify the master list
            chunk = self.chunks[idx].copy()
            chunk["relevance_score"] = float(scores[idx])
            retrieved.append(chunk)
            source_files.add(chunk["file_path"])
            
        return [retrieved, source_files]