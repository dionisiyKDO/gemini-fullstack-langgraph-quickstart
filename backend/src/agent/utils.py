from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from pathlib import Path
import json
import re
import os

def get_research_topic(messages: List[AnyMessage]) -> str:
    """Get the research topic from the messages."""
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def get_all_file_paths(root_dir: str) -> list[str]:
    """Recursively find all .md, .ipynb and .txt files in the directory."""
    paths = []
    if not os.path.exists(root_dir):
        return []
        
    for dirpath, _, filenames in os.walk(root_dir):
        if any(excluded_dir in dirpath for excluded_dir in ["images", "assets"]):
            continue
        for f in filenames:
            if f.endswith(".md") or f.endswith(".txt") or f.endswith(".ipynb"):
                full_path = os.path.join(dirpath, f)
                paths.append(full_path)
    
    return paths


def read_file_content(file_path: str) -> str:
    """Read .md/.txt normally, and parses .ipynb to markdown."""
    try:
        if file_path.endswith(".ipynb"):
            with open(file_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)
            
            markdown_content = ""
            for cell in notebook.get("cells", []):
                cell_source = "".join(cell.get("source", []))
                if cell.get("cell_type") == "markdown":
                    markdown_content += f"\n{cell_source}\n"
                elif cell.get("cell_type") == "code":
                    markdown_content += f"\n```python\n{cell_source}\n```\n"
            return markdown_content
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def split_markdown_by_headers(markdown_text: str, filename: str) -> list[dict]:
    """Split markdown into blocks, ignoring headers inside code blocks."""
    lines = markdown_text.split('\n')
    blocks = []
    
    current_header = "Intro"
    current_content = []
    in_code_block = False
    
    # Regex for headers (1-6 hashes followed by space)
    header_pattern = re.compile(r'^(#{1,6})\s+(.*)')
    
    for line in lines:
        # Toggle code block state
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_content.append(line)
            continue

        # Check for header ONLY if not in code block
        match = header_pattern.match(line)
        if match and not in_code_block:
            # Save previous block
            if current_content:
                blocks.append({
                    "header_id": f"{filename} > {current_header}",
                    "content": "\n".join(current_content).strip()
                })
            
            # Reset for new block
            current_header = match.group(2).strip()
                
            current_content = [line] # Start content with the header itself
        else:
            current_content.append(line)
            
    # Append the last block
    if current_content:
        blocks.append({
            "header_id": f"{filename} > {current_header}",
            "content": "\n".join(current_content).strip()
        })
        
    return blocks