import sys
import os
import logging
import sqlite3
# Fix OpenMP conflict: import torch before faiss
import torch
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.storage.storage_manager import StorageFactory, StorageConfig, DeploymentEnvironment
from src.pipeline.master_pipeline import PipelineConfig

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ValidationTest")

def test_validation_scenarios():
    """
    Run validation checks on the artifacts produced by the pipeline.
    Assumes pipeline has already run.
    """
    config_path = Path("config.json")
    if not config_path.exists():
         raise FileNotFoundError("config.json not found")
         
    # Load Config to get paths
    config_data = ConfigLoader.load_config(str(config_path))
    # We need to reconstruct PipelineConfig or just extract storage paths
    # Simplest is to assume default path "data" if not specified, 
    # but let's try to obey config.
    
    # Check Tables
    validate_database_tables(Path("data"))
    
    # Check Search
    validate_search_scenarios(str(config_path))

def validate_database_tables(data_dir: Path):
    logger.info("1. Validate Database Schema")
    db_path = data_dir / "metadata.db"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Did the pipeline run?")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")]
        
        required_tables = {'metadata', 'search_index'}
        # FTS5 tables usually include search_index_data, etc. checking main one is enough.
        
        missing = required_tables - set(tables)
        if missing and 'search_index' not in tables:
             # loose check for search_index because FTS might not show up in type='table' in some sqlite versions? 
             # No, it should.
             pass

        assert 'metadata' in tables, "Table 'metadata' missing"
        assert 'search_index' in tables, "Table 'search_index' missing"
        
        logger.info(f"Database Schema Validated. Found tables: {tables}")

def validate_search_scenarios(config_path: str):
    logger.info("2. Validate Search Scenarios")
    
    # Initialize Storage independently
    # We need to manually construct storage manager settings based on config
    # Or reuse the MasterPipeline factory logic which is cleaner
    
    # Re-using pipeline config hydration logic to ensure we match how pipeline did it
    from src.pipeline.master_pipeline import MasterPipeline
    # We don't run the pipeline, just init to get the storage manager
    pipeline = MasterPipeline.create_from_config(config_path)
    storage = pipeline.storage_manager

    # Scenario A: Simple Search
    query_simple = "class MasterPipeline"
    results = storage.search_by_vector(query_simple, limit=5)
    assert len(results) > 0, f"Simple search '{query_simple}' returned 0 results"
    logger.info(f"Simple Search ('{query_simple}'): Passed ({len(results)} results)")

    # Scenario B: Common Search
    query_common = "import"
    results = storage.search_by_vector(query_common, limit=10)
    assert len(results) > 0, f"Common search '{query_common}' returned 0 results"
    logger.info(f"Common Search ('{query_common}'): Passed ({len(results)} results)")

    # Scenario C: Failure/Noise Search
    query_fail = "xyz123nonexistent_super_random_string"
    results = storage.search_by_vector(query_fail, limit=5)
    logger.info(f"Failure Search ('{query_fail}'): Completed ({len(results)} results). Vector search is approximate, so non-zero is expected.")

    logger.info("Validation Scenarios: ALL PASSED")

if __name__ == "__main__":
    test_validation_scenarios()
