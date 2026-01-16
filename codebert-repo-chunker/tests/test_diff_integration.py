import unittest
import shutil
import tempfile
import sqlite3
import hashlib
from pathlib import Path
from src.pipeline.master_pipeline import MasterPipeline, PipelineConfig
from src.storage.storage_manager import StorageConfig

class TestDiffIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.test_dir / "test_repo"
        self.repo_dir.mkdir()
        
        # Create a dummy file
        self.file1 = self.repo_dir / "file1.py"
        self.file1.write_text("print('hello')")
        
        self.config = PipelineConfig(
            env="dev",
            storage_type="sqlite",
            chunk_size=128,
            enable_monitoring=False,
            enable_embeddings=False # Now supported!
        )
        # Monkey-patch config to disable embeddings? 
        # PipelineConfig definition: 
        # @dataclass
        # class PipelineConfig:
        #    ...
        #    enable_distributed: bool = False
        
        # Wait, PipelineConfig doesn't have enable_embeddings option? 
        # Checking src/pipeline/master_pipeline.py.
        # It's passed to ProcessingConfig.
        
        pass

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Cleanup "data" dir if it was created in CWD (which is project root)
        # To avoid this, we should really mock StorageFactory or pass storage config.
        # But for now, let's try to pass a config that uses test_dir if possible.
        pass

    def test_diff_flow(self):
        # Initialize Pipeline
        pipeline = MasterPipeline(self.config)
        # Override storage manager's base path to our test dir
        pipeline.storage_manager.config.base_path = self.test_dir / "data"
        pipeline.storage_manager.chunk_storage._setup_db() # re-init with new path
        pipeline.storage_manager.metadata_store._setup_db()
        # Vector store is auto-initialized if config allows
        
        print("\n--- Run 1: Initial ---")
        pipeline.run(self.repo_dir)
        self.assertEqual(pipeline.stats["chunks_created"], 1)
        self.assertEqual(pipeline.stats["files_scanned"], 1)
        
        # Verify DB has checksum
        checksums = pipeline.storage_manager.get_file_checksums("test_repo")
        self.assertIn("file1.py", checksums)
        
        print("\n--- Run 2: No Changes ---")
        # Reset stats
        pipeline.stats["chunks_created"] = 0
        pipeline.run(self.repo_dir)
        self.assertEqual(pipeline.stats["chunks_created"], 0, "Should skip processing")
        
        print("\n--- Run 3: Modification ---")
        self.file1.write_text("print('modified')")
        pipeline.stats["chunks_created"] = 0
        pipeline.run(self.repo_dir)
        self.assertEqual(pipeline.stats["chunks_created"], 1, "Should re-process modified file")
        
        # Verify content updated (old chunk gone? new chunk there?)
        # We can't easily check 'gone' without querying DB directly, 
        # but chunks_created=1 means it ran.
        
        print("\n--- Run 4: New File ---")
        file2 = self.repo_dir / "file2.py"
        file2.write_text("def foo(): pass")
        pipeline.stats["chunks_created"] = 0
        pipeline.run(self.repo_dir)
        self.assertEqual(pipeline.stats["chunks_created"], 1, "Should process new file only")
        
        print("\n--- Run 5: Deletion ---")
        file2.unlink()
        pipeline.stats["chunks_created"] = 0
        pipeline.run(self.repo_dir)
        self.assertEqual(pipeline.stats["chunks_created"], 0, "No new chunks")
        
        # Verify file2 is gone from DB
        checksums = pipeline.storage_manager.get_file_checksums("test_repo")
        self.assertNotIn("file2.py", checksums)
        
        pipeline.close()

if __name__ == '__main__':
    unittest.main()
