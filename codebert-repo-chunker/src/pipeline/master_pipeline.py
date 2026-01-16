
import os
import sys
import logging
import time
import shutil
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import json
import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Integration components
# Integration components
from src.pipeline.repository_scanner import RepositoryScanner, ScannerConfig
from src.pipeline.dependency_resolver import DependencyResolver
from src.pipeline.quality_analyzer import QualityAnalyzer
from src.pipeline.relationship_builder import RelationshipBuilder, RelationshipConfig
from src.pipeline.report_generator import ReportGenerator

# Imports for distributed processing (Celery)
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Imports for monitoring
try:
    from prometheus_client import Counter, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Imports for storage integration
# Imports for storage integration
from src.pipeline.chunk_processor import ChunkProcessor, ProcessingConfig
from src.storage.storage_manager import StorageFactory, StorageConfig, DeploymentEnvironment
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the Master Pipeline"""
    env: str = "dev"
    max_workers: int = 4
    batch_size: int = 10
    force_full_scan: bool = False
    chunk_size: int = 512
    overlap: int = 50
    storage_type: str = "sqlite"  # sqlite, postgres, mongo
    enable_monitoring: bool = False
    monitoring_port: int = 8000
    enable_distributed: bool = False
    redis_url: str = "redis://localhost:6379/0"
    enable_embeddings: bool = True # New flag to control embedding generation

class MasterPipeline:
    """
    Orchestrator for the entire CodeBERT Repo Chunker pipeline.
    Manages stages: Scan -> Resolve -> Chunk -> Embed -> Store.
    """
    
    @classmethod
    def create_from_config(cls, config_path: str = "config.json") -> 'MasterPipeline':
        """Factory: Create pipeline from external config file"""
        config_data = ConfigLoader.load_config(config_path)
        
        # Map JSON structure to PipelineConfig flat structure
        pipeline_settings = config_data.get('pipeline', {})
        processing_settings = config_data.get('processing', {})
        storage_settings = config_data.get('storage', {})
        monitoring_settings = config_data.get('monitoring', {})
        
        config = PipelineConfig(
            env=pipeline_settings.get('environment', 'dev'),
            max_workers=pipeline_settings.get('max_workers', 4),
            batch_size=pipeline_settings.get('batch_size', 10),
            force_full_scan=pipeline_settings.get('force_full_scan', False),
            
            chunk_size=processing_settings.get('chunk_size', 512),
            overlap=processing_settings.get('overlap', 50),
            enable_embeddings=processing_settings.get('enable_embeddings', True),
            
            storage_type=storage_settings.get('type', 'sqlite'),
            
            enable_monitoring=monitoring_settings.get('enabled', False),
            monitoring_port=monitoring_settings.get('port', 8000)
        )
        return cls(config)

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.running = False
        self.start_time = None
        self.components_initialized = False
        
        # Core components
        self.scanner = None
        self.dependency_resolver = None
        self.quality_analyzer = None
        self.report_generator = None
        self.chunk_processor = None
        self.storage_manager = None
        
        # State
        self.input_queue = asyncio.Queue()
        self.processed_files: Set[Path] = set()
        self.dependency_graph: Dict[str, Any] = {}
        self.dependency_data: Dict[str, Any] = {}
        self.stats = {
            "status": "IDLE",
            "files_scanned": 0,
            "chunks_created": 0,
            "errors": [],
            "start_time": None
        }
        
        self._init_components()

    def _init_components(self):
        """Initialize all pipeline components"""
        try:
            # 1. Storage
            storage_config = StorageConfig(
                environment=DeploymentEnvironment.DEVELOPMENT if self.config.env == "dev" else DeploymentEnvironment.PRODUCTION,
                primary_backend=self.config.storage_type
            )
            # Use factory to create storage manager
            self.storage_manager = StorageFactory.create(storage_config) 
            
            # 2. Processor
            proc_config = ProcessingConfig(
                chunk_size=self.config.chunk_size,
                overlap=self.config.overlap,
                max_workers=self.config.max_workers,
                enable_embeddings=self.config.enable_embeddings # Pass flag
            )
            self.chunk_processor = ChunkProcessor(proc_config, self.storage_manager)
            
            # 3. New Modules
            self.scanner = RepositoryScanner()
            self.dependency_resolver = DependencyResolver()
            # Analysis
            self.relationship_builder = RelationshipBuilder()
            self.quality_analyzer = QualityAnalyzer()
            self.report_generator = ReportGenerator()
                
            # 4. Monitoring
            if self.config.enable_monitoring and PROMETHEUS_AVAILABLE:
                self._setup_metrics()
                
            self.components_initialized = True
            logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            self.stats["errors"].append(str(e))
            raise

    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.metrics = {
            "files_processed": Counter('pipeline_files_processed_total', 'Total files processed'),
            "chunks_generated": Counter('pipeline_chunks_generated_total', 'Total chunks generated'),
            "processing_time": Histogram('pipeline_processing_seconds', 'Time spent processing files')
        }
        # start_http_server(self.config.monitoring_port)

    def _scan_repository(self, repo_path: Path) -> List[Path]:
        """Discovery phase"""
        if not self.scanner:
            logger.warning("Scanner not available, using simple walk")
            return list(repo_path.rglob("*.*"))
            
        logger.info(f"Scanning {repo_path}...")
        scanned_files = self.scanner.scan(repo_path)
        # Convert ScannedFile objects to Path objects
        return [f.path for f in scanned_files]

    def _analyze_dependencies(self, file_paths: List[Path]) -> tuple:
        """Dependency Resolution phase"""
        if not self.dependency_resolver:
            return {}, {}
            
        logger.info("Analyzing dependencies...")
        requests = self.dependency_resolver.analyze_repository(file_paths)
        # Store full result for report generator
        self.dependency_data = requests
        return requests.get("graph", {}), requests.get("imports", {})

    def _analyze_quality(self, chunks: List[Any]):
        """Quality Analysis phase"""
        if not self.quality_analyzer:
            return
        
        logger.info("Analyzing code quality...")
        # Analyze chunks in memory or from list
        results = self.quality_analyzer.analyze_chunks(chunks)
        # We could merge results into stats or existing metadata?
        # For now just logging summary or storing report data if needed
        self.stats['quality'] = results.get('overall_score', 0)

    def _generate_reports(self, session_id: str, dependency_graph: Dict[str, Any] = None, module_map: Dict[str, str] = None, repo_name: str = "unknown"):
        """Reporting phase"""
        if not self.report_generator:
            return
            
        logger.info("Generating reports...")
        self.report_generator.generate_report(self.stats, dependency_graph, session_id=session_id, module_map=module_map, repo_name=repo_name)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read in chunks to avoid memory issues
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash {file_path}: {e}")
            return ""

    def run(self, repo_path: Union[str, Path]):
        """Run the full pipeline"""
        repo_path = Path(repo_path).resolve()
        self.start_time = datetime.now(timezone.utc)
        self.stats["start_time"] = self.start_time
        self.stats["status"] = "RUNNING"
        
        try:
            # 1. Scan & Hash
            all_files = self._scan_repository(repo_path)
            self.stats["files_scanned"] = len(all_files)
            
            # 2. Diff Logic (Diff Logic Phase 2)
            repo_name = repo_path.name
            current_hashes = {}
            files_to_process = []
            
            logger.info("Computing file checksums for diff...")
            for f in all_files:
                try:
                    rel_path = f.relative_to(repo_path).as_posix()
                    current_hashes[rel_path] = self._compute_file_hash(f)
                except Exception as e:
                    logger.warning(f"Skipping hash for {f}: {e}")

            # Get previous state
            stored_hashes = self.storage_manager.get_file_checksums(repo_name)
            
            # Compare
            new_files = []
            modified_files = []
            deleted_files = []
            
            # Detect New & Modified
            for rel_path, current_hash in current_hashes.items():
                if rel_path not in stored_hashes:
                    new_files.append(rel_path)
                else:
                    # It exists in both.
                    if self.config.force_full_scan:
                        modified_files.append(rel_path)
                    elif stored_hashes[rel_path] != current_hash:
                        modified_files.append(rel_path)
                    
            # Detect Deleted
            for rel_path in stored_hashes:
                if rel_path not in current_hashes:
                    deleted_files.append(rel_path)
            
            logger.info(f"Diff Analysis: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted.")
            
            # Handle Deletions & Modifications (Cleanup old chunks)
            files_to_cleanup = deleted_files + modified_files
            if files_to_cleanup:
                logger.info(f"Cleaning up {len(files_to_cleanup)} stale files from DB...")
                # We need to cleanup by file_path. 
                # Note: get_file_checksums returns relative paths if that's what we stored?
                # Actually, in metadata_store.py we store "file_path" which might be absolute or relative depending on how it was stored.
                # To be safe, we should assume the DB keys match what we derived earlier.
                # However, our storage relies on 'file_path'.
                # Let's ensure we pass the correct path format expected by delete_file_chunks.
                # Ideally, delete_file_chunks expects the exact string stored in 'file_path' column.
                
                # IMPORTANT: If we stored absolute paths before, this might mismatch if we only used relative keys for diff.
                # But storage_manager.get_file_checksums returns the stored file_paths as keys.
                # So we can just use those keys.
                for rel_path in files_to_cleanup:
                    # In DB we might have stored normalized absolute paths or relative.
                    # We will use what was returned by get_file_checksums (which are the DB keys).
                    self.storage_manager.delete_file_chunks(rel_path)

            # Filter Processing List
            # We need to map relative paths back to absolute Path objects for processing
            files_to_process_rel = set(new_files + modified_files)
            files_to_process = [f for f in all_files if f.relative_to(repo_path).as_posix() in files_to_process_rel]
            
            if not files_to_process and not deleted_files:
                logger.info("No changes detected. Skipping processing.")
                self.stats["status"] = "COMPLETED"
                return

            logger.info(f"Processing {len(files_to_process)} files...")

            # 3. Dependencies (Analyze ALL files to ensure graph is complete, or just processed?)
            # Dependency graph usually needs the whole repo context. 
            # If we skip files, we might miss imports FROM them or TO them.
            # OPTIMIZATION CHOICE: We still run dependency analysis on ALL files to maintain graph integrity.
            # This is fast compared to chunking/embedding.
            dep_graph, imports = self._analyze_dependencies(all_files)
            self.dependency_graph = dep_graph
            self.imports_map = imports 
            
            # 4. Process (Chunk & Embed) - ONLY Changed Files
            chunks = self.chunk_processor.process_batch(files_to_process, repo_root=repo_path)
            self.stats["chunks_created"] += len(chunks)
            
            # 5. Relationships
            # Integrate RelationshipBuilder as requested in Code Review
            if chunks: # Only build if we have new chunks or force full scan? 
                # Ideally we need ALL chunks for relationships. 
                # For now, if incremental, we might be missing context. 
                # But typically we want to run this on the full set.
                # Let's try to get all chunks from storage if possible, or just build for new ones (partial).
                # Reviewer said: "Cross-file relationships... never computed".
                # We should instantiate and run it.
                logger.info("Building relationship graph...")
                graph = self.relationship_builder.build_relationships(chunks, storage=self.storage_manager.vector_store)
                self.relationship_graph = graph
                
                # Persist graph (simple pickle for now)
                try:
                    graph_path = self.storage_manager.config.base_path / "relationship_graph.pkl"
                    with open(graph_path, "wb") as f:
                        pickle.dump(graph, f)
                    logger.info(f"Persisted relationship graph to {graph_path}")
                except Exception as e:
                    logger.warning(f"Failed to persist relationship graph: {e}")

            # 6. Quality
            self._analyze_quality(chunks)
            
            # 5. Report
            session_id = f"run_{int(time.time())}"
            repo_name = repo_path.name if 'repo_path' in locals() else "unknown"
            module_map = self.dependency_data.get('module_map', {}) if hasattr(self, 'dependency_data') and self.dependency_data else {}
            self._generate_reports(session_id, self.dependency_graph, module_map=module_map, repo_name=repo_name)
            
            self.stats["status"] = "COMPLETED"
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            self.stats["status"] = "FAILED"
            self.stats["errors"].append(str(e))
        finally:
            duration = datetime.now(timezone.utc) - self.start_time
            self.stats["duration_seconds"] = duration.total_seconds()
            
            # 5. Report - Moved to finally block to ensure it runs even on failure
            if self.report_generator and self.stats["status"] != "COMPLETED":
                session_id = f"run_{int(time.time())}"
                # The imports_map from _analyze_dependencies serves as the module_map
                module_map = self.dependency_data.get('module_map', {}) if hasattr(self, 'dependency_data') and self.dependency_data else {}
                repo_name = repo_path.name if 'repo_path' in locals() else "unknown"
                self._generate_reports(session_id, self.dependency_graph, module_map=module_map, repo_name=repo_name)
            
            # Explicit Summary Log for User
            repo_name = repo_path.name if 'repo_path' in locals() else "Unknown"
            total_time = self.stats.get("duration_seconds", 0)
            
            logger.info("="*50)
            logger.info("BEYOND PIPELINE SUMMARY")
            logger.info("="*50)
            logger.info(f"Repository:    {repo_name}")
            logger.info(f"Total Time:    {total_time:.2f}s")
            logger.info(f"Files Scanned: {self.stats.get('files_scanned', 0)}")
            logger.info(f"Chunks Created: {self.stats.get('chunks_created', 0)}")
            logger.info("="*50)

    def close(self):
        """Close pipeline resources"""
        if self.chunk_processor:
            self.chunk_processor.close()