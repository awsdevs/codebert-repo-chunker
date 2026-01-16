import sys
import os

# CRITICAL: Prevent OpenMP conflicts between Torch and FAISS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# CRITICAL: Import torch before faiss/transformers to prevent OpenMP segfaults
import torch
import faiss

from pathlib import Path
import logging

# Add src to path
sys.path.append(os.getcwd())

from src.pipeline.master_pipeline import MasterPipeline, PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

def verify_fixes():
    logger.info("Starting Verification of Code Review Fixes")
    
    # 1. Config
    config = PipelineConfig(
        storage_type="sqlite",
        enable_embeddings=True, # Re-enabled with fix
        force_full_scan=True # Force run to generate relationships
    )
    
    pipeline = MasterPipeline(config)
    
    # 2. Run Pipeline on current repo (subset)
    # We'll run on 'src/utils' to be fast
    repo_path = Path("src/utils")
    
    try:
        pipeline.run(repo_path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # Print stack trace
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Verify Relationship Graph
    if hasattr(pipeline, 'relationship_graph'):
        graph = pipeline.relationship_graph
        logger.info(f"Relationship Graph built with {len(graph.edges)} edges")
        if not graph.edges:
            logger.warning("No relationships found! Check RelationshipBuilder integration.")
        else:
            logger.info("RelationshipBuilder integration verified.")
            
            # Check for Similarity relationships (Vector Search verification)
            sim_edges = [e for e in graph.edges if e.relation_type.value == "similar_to"]
            if sim_edges:
                logger.info(f"Found {len(sim_edges)} similarity edges")
            else:
                logger.warning("No similarity edges found (Expected if embeddings disabled)")
            
    else:
        logger.error("pipeline.relationship_graph not found! Integration failed.")
        
    # 4. Verify Quality Analysis
    quality_score = pipeline.stats.get('quality', 0)
    logger.info(f"Quality Score: {quality_score}")
    if quality_score > 0:
        logger.info("Quality Analysis verified.")
    else:
        logger.warning("Quality Analysis returned 0 score.")

    # 5. Check artifacts
    # StorageConfig default base_path is "data" relative to CWD if not specified? 
    # Actually StorageConfig default is "data".
    path = pipeline.storage_manager.config.base_path / "relationship_graph.pkl"
    if path.exists():
        logger.info(f"Graph persisted to {path}")
        
    logger.info("Verification Complete")

if __name__ == "__main__":
    verify_fixes()
