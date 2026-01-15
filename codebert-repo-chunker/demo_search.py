import logging
import torch
import numpy as np
from src.storage.storage_manager import StorageManager, StorageConfig, DeploymentEnvironment
from pathlib import Path
from src.embeddings.codebert_encoder import CodeBERTEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_search():
    print("Initializing Storage & Model...")
    
    # Initialize Storage
    # Pipeline uses 'data' directory
    config = StorageConfig(base_path=Path("data"))
    manager = StorageManager(config)
    
    # 2. Initialize Encoder (for query -> vector)
    encoder = CodeBERTEncoder()
    
    # 3. Embed Query
    query_text = "Find code that handles database connection"
    print(f"\nQuery: '{query_text}'")
    
    query_embedding = encoder.encode([query_text]).embeddings[0]
    
    # 4. Search
    print("Searching...")
    results = manager.search_by_vector(query_embedding, limit=3)
    
    # 6. Display Results
    print(f"\nFound {len(results)} matches:")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. [{res['file_path']}] (Score: {res['score']:.4f})")
        print(f"   Function: {res['function_name']}")
        print(f"   Snippet: {res['snippet'].replace(chr(10), ' ')[:100]}...")

if __name__ == "__main__":
    demo_search()
