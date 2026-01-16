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
    # Load from config.json to ensure path consistency
    from src.utils.config_loader import ConfigLoader
    conf_data = ConfigLoader.load_config("config.json")
    base_path = Path(conf_data.get('storage', {}).get('base_path', 'data'))
    
    config = StorageConfig(base_path=base_path)
    manager = StorageManager(config)
    
    # 2. Initialize Encoder (for query -> vector)
    encoder = CodeBERTEncoder()
    
    # 3. Embed Query
    query_text = "database schema"
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

    # 7. Complex Scenario Verification
    print("\n--- Complex Scenario Verification ---")
    
    # Scenario: Hybrid Search check
    # We want to find 'Logger' using FTS
    target_file = "src/utils/logger.py"
    complex_query = "Logger"
    print(f"Verifying FTS can find definition: '{complex_query}'")
    
    fts_hits = manager.search_text(complex_query, limit=5)
    found = False
    for cid, rank in fts_hits:
        m = manager.metadata_store.get(cid)
        # Check if file path ends with logger.py (robust to absolute paths)
        if m and m.get('location', {}).get('file_path', '').endswith('logger.py'):
            found = True
            print(f"SUCCESS: Found {complex_query} in {m['location']['file_path']}")
            break
            
    if not found:
        print(f"FAILURE: Could not find {complex_query} in logger.py")
        print("Top FTS results were:")
        for cid, rank in fts_hits:
            m = manager.metadata_store.get(cid)
            print(f" - {m.get('location', {}).get('file_path')} (Rank: {rank})")
            
    # Verify Vector Search finds concepts
    concept_query = "dependency resolution mechanism"
    print(f"\nVerifying Vector Search for concept: '{concept_query}'")
    vec = encoder.encode([concept_query]).embeddings[0]
    vec_hits = manager.search_by_vector(vec, limit=5)
    
    concept_found = any('dependency_resolver' in r['file_path'] for r in vec_hits)
    if concept_found:
         print("SUCCESS: Vector search found dependency_resolver related files.")
    else:
         print("WARNING: Vector search did not prioritize dependency_resolver (might be acceptable depending on embeddings)")

if __name__ == "__main__":
    demo_search()
# Touch 3 for FTS update
