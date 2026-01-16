import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch # Pre-load to avoid segfault with faiss
import sys
import logging
import asyncio
from pathlib import Path
import uuid
from datetime import datetime

# Add src to path
sys.path.append(str(Path.cwd()))

from src.pipeline.master_pipeline import MasterPipeline, PipelineConfig
from src.pipeline.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("runner")

def run():
    print("Initializing Master Pipeline...")
    
    
    # Initialize pipeline
    # Try to load from config.json if it exists, otherwise use factory with default
    # Or just use factory which defaults to config.json
    pipeline = MasterPipeline.create_from_config("config.json")
    
    # Run on current directory
    repo_path = Path.cwd()
    session_id = str(uuid.uuid4())[:8]
    
    print(f"Starting scan of {repo_path} (Session: {session_id})...")
    
    try:
        # Run standard pipeline
        pipeline.run(repo_path)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pass # Pipeline handles its own cleanup now
        
if __name__ == "__main__":
    run()
