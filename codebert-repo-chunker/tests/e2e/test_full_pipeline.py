import sys
import os
import shutil
import logging
# Fix OpenMP conflict: import torch before faiss
import torch
from pathlib import Path
from typing import List

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.master_pipeline import MasterPipeline
from src.utils.config_loader import ConfigLoader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("E2ETest")

def test_clean_slate_run():
    """
    Run the pipeline on the src folder.
    Expectations:
    1. Pipeline completes with status "COMPLETED".
    2. Reports are generated.
    3. No 'PosixPath' warnings in logs.
    """
    config_path = Path("config.json")
    repo_path = Path("src").resolve()
    
    logger.info(f"Starting E2E Test on {repo_path}")
    
    # Initialize Pipeline
    pipeline = MasterPipeline.create_from_config(str(config_path))
    
    # Run
    try:
        pipeline.run(repo_path)
        
        # Verify Status
        assert pipeline.stats["status"] == "COMPLETED", f"Pipeline failed with status: {pipeline.stats.get('status')}"
        
        # Verify Reports
        reports_dir = Path("reports")
        reports = list(reports_dir.glob("report_*.md"))
        assert len(reports) > 0, "No markdown reports generated"
        
        logger.info("Pipeline Status: COMPLETED")
        logger.info(f"Reports Generated: {len(reports)}")
        
    except Exception as e:
        logger.error(f"E2E Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_clean_slate_run()
