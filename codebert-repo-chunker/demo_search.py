import logging
import torch
import numpy as np
from src.storage.storage_manager import StorageManager, StorageConfig, DeploymentEnvironment
from src.embeddings.codebert_encoder import CodeBERTEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_search():
    print("Initializing Storage & Model...")
    
    # 1. Initialize Storage
    config = StorageConfig(environment=DeploymentEnvironment.PRODUCTION)
    storage = StorageManager(config)
    
    # 2. Initialize Encoder (for query -> vector)
    encoder = CodeBERTEncoder()
    
    # 3. Define Query
    query = "Find code that handles database connection"
    print(f"\nQuery: '{query}'")
    
    # 4. Encode Query
    # Note: CodeBERT expects tokens, simplified here for demo
    # We use the same encoding logic as the pipeline
    with torch.no_grad():
        inputs = encoder.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = encoder.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy() # CLS token
        
    # 5. Search
    print("Searching...")
    results = storage.search_by_vector(embedding, limit=3)
    
    # 6. Display Results
    print(f"\nFound {len(results)} matches:")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. [{res['file_path']}] (Score: {res['score']:.4f})")
        print(f"   Function: {res['function_name']}")
        print(f"   Snippet: {res['snippet'].replace(chr(10), ' ')[:100]}...")

if __name__ == "__main__":
    demo_search()
