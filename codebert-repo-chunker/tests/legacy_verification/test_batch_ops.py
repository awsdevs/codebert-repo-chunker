import unittest
import shutil
import tempfile
import sqlite3
import numpy as np
import zlib
from pathlib import Path
from src.storage.storage_manager import StorageManager, StorageConfig, DeploymentEnvironment
from src.core.chunk_model import Chunk, ChunkLocation, ChunkType

class TestBatchOperations(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = StorageConfig(base_path=self.test_dir)
        self.manager = StorageManager(self.config)

    def tearDown(self):
        self.manager.close()
        shutil.rmtree(self.test_dir)

    def test_chunk_storage_batch(self):
        print("\nTesting ChunkStorage.store_batch...")
        chunks = []
        for i in range(100):
            chunks.append((f"c_{i}", f"content_{i}", "src/main.py", "python"))
        
        count = self.manager.chunk_storage.store_batch(chunks)
        self.assertEqual(count, 100)
        
        # Verify DB
        cursor = self.manager.chunk_storage.conn.execute("SELECT count(*) FROM chunks")
        self.assertEqual(cursor.fetchone()[0], 100)
        print("ChunkStorage batch insert SUCCESS")

    def test_metadata_store_batch(self):
        print("\nTesting MetadataStore.store_batch...")
        metadata_list = []
        for i in range(100):
            meta = {
                "file_path": "src/main.py",
                "repository": "test-repo",
                "function_name": f"func_{i}",
                "docstring": "test docstring"
            }
            metadata_list.append((f"c_{i}", meta))
            
        count = self.manager.metadata_store.store_batch(metadata_list)
        self.assertEqual(count, 100)
        
        # Verify Metadata DB
        cursor = self.manager.metadata_store.conn.execute("SELECT count(*) FROM metadata")
        self.assertEqual(cursor.fetchone()[0], 100)
        
        # Verify FTS
        cursor = self.manager.metadata_store.conn.execute("SELECT count(*) FROM search_index")
        self.assertEqual(cursor.fetchone()[0], 100)
        
        # Verify Match
        cursor = self.manager.metadata_store.conn.execute(
            "SELECT chunk_id FROM search_index WHERE rich_text MATCH 'func_50'"
        )
        self.assertEqual(cursor.fetchone()[0], "c_50")
        print("MetadataStore batch insert SUCCESS")

    def test_storage_manager_batch(self):
        print("\nTesting StorageManager.store_chunks_batch...")
        chunks = []
        for i in range(50):
            chunks.append(Chunk(
                id=f"sm_{i}",
                content=f"def func_{i}(): pass",
                chunk_type=ChunkType.FUNCTION,
                location=ChunkLocation("src/utils.py", i, i+1),
                language="python",
                metadata={"repository": "test-repo"},
                embedding=np.random.rand(768).astype(np.float32)
            ))
            
        count = self.manager.store_chunks_batch(chunks)
        self.assertEqual(count, 50)
        
        # Verify Vector Store
        if self.manager.vector_store:
            self.assertEqual(self.manager.vector_store.index.ntotal, 50)
            
        print("StorageManager batch insert SUCCESS")

if __name__ == '__main__':
    unittest.main()
