
from src.storage.storage_manager import StorageConfig, StorageManager, DeploymentEnvironment
from src.core.chunk_model import Chunk, ChunkLocation, ChunkType
from pathlib import Path
import numpy as np
import shutil
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup():
    # Clean verification dir
    path = Path("data_verify_diff")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    
    config = StorageConfig(base_path=path, environment=DeploymentEnvironment.DEVELOPMENT)
    return StorageManager(config)

def create_dummy_chunk(file_path, chunk_id):
    return Chunk(
        id=chunk_id,
        content=f"def foo(): pass # {chunk_id}",
        chunk_type=ChunkType.FUNCTION,
        location=ChunkLocation(file_path, 1, 10),
        language="python",
        embedding=np.random.rand(768).astype(np.float32),
        metadata={"repository": "test-repo", "file_checksum": "abc12345"}
    )

def test_diff_flow():
    manager = setup()
    
    print("\n--- 1. Adding Chunks ---")
    c1 = create_dummy_chunk("src/main.py", "c1")
    c2 = create_dummy_chunk("src/main.py", "c2")
    c3 = create_dummy_chunk("src/utils.py", "c3")
    
    manager.store_chunk(c1)
    manager.store_chunk(c2)
    manager.store_chunk(c3)
    
    print("Checking counts...")
    print("Checking counts...")
    assert len(manager.metadata_store.list_by_file("src/main.py")) == 2
    assert len(manager.metadata_store.list_by_file("src/utils.py")) == 1
    
    # STRICT CHECK: Vector Store must be active
    assert manager.vector_store is not None, "VectorStore failed to initialize"
    assert manager.vector_store.index.ntotal == 3
        
    print("\n--- 2. Simulating Deletion (Diff Update) ---")
    # Simulate: src/main.py was modified or deleted. We ensure we remove old chunks first.
    count = manager.delete_file_chunks("src/main.py")
    print(f"Deleted {count} chunks for src/main.py")
    
    assert count == 2
    assert len(manager.metadata_store.list_by_file("src/main.py")) == 0
    assert len(manager.metadata_store.list_by_file("src/utils.py")) == 1  # Should remain
    
    # STRICT CHECK
    print(f"Vector Store Total: {manager.vector_store.index.ntotal}")
    assert manager.vector_store.index.ntotal == 1
        
    print("\n--- 3. Persistence Check ---")
    manager.close()
    
    # Re-open
    config = StorageConfig(base_path=Path("data_verify_diff"))
    manager2 = StorageManager(config)
    
    # STRICT CHECK
    assert manager2.vector_store is not None, "Reloaded VectorStore failed to initialize"
    print(f"Reloaded Vector Store Total: {manager2.vector_store.index.ntotal}")
    assert manager2.vector_store.index.ntotal == 1
        
    print("\nSUCCESS: Diff Update Primitives Verified")

if __name__ == "__main__":
    try:
        test_diff_flow()
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)
