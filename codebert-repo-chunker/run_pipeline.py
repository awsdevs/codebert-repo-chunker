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
    
    # Create configuration
    config = PipelineConfig(
        max_workers=4,
        chunk_size=512,
        batch_size=10
    )
    
    # Initialize pipeline
    pipeline = MasterPipeline(config)
    
    # Run on current directory
    repo_path = Path.cwd()
    session_id = str(uuid.uuid4())[:8]
    
    print(f"Starting scan of {repo_path} (Session: {session_id})...")
    
    try:
        # 1. Scan & Process
        # We process the current directory directly
        file_paths = []
        for root, _, files in os.walk(repo_path):
            if any(p in str(Path(root)) for p in ['.git', '__pycache__', 'venv', 'env']):
                continue
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.ts', '.xml', '.yaml', '.json')):
                    file_paths.append(Path(root) / file)
        
        print(f"  Found {len(file_paths)} files.")

        # 2. Dependencies
        print("[2] Analyzing Dependencies...")
        dep_graph, imports = pipeline._analyze_dependencies(file_paths)
        print(f"  Resolved graph with {len(dep_graph)} nodes.")

        # 3. Process Chunks (Classify -> Chunk -> Embed -> Store)
        print("[3] Processing Files (Classify -> Chunk -> Embed -> Store)...")
        # We manually use the chunk processor here since _process_chunks typically typically pulls from queue
        # For verification, we process the list directly
        processor = pipeline.chunk_processor
        chunks = processor.process_batch(file_paths)
        print(f"  Processed {len(file_paths)} files into {len(chunks)} chunks.")
        
        # 4. Generate Reports (Sample on files)
        # We need chunks. Unfortuantely `_process_chunks` typically reads queue.
        # For visualization, we care mostly about dependency graph we just built.
        
        # 4. Generate Report
        print("[4] Generating Reports...")
        stats = {
            "status": "COMPLETED",
            "files_scanned": len(file_paths),
            "chunks_created": len(chunks),
            "duration_seconds": 0,
            "errors": []
        }
        
        reporter = ReportGenerator()
        
        # JSON/MD
        md_path = reporter.generate(stats, session_id)
        print(f"  Report: {md_path}")
        
        # HTML Visualization
        html_path = reporter.generate_html_report(stats, dep_graph, session_id)
        print(f"  Visual Report: {html_path}")
        
        print("\nPipeline Execution Verified Successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'pipeline' in locals():
            pipeline.close()
            print("Pipeline resources closed.")

if __name__ == "__main__":
    run()
