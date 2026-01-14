"""
Chunk processing pipeline for end-to-end code analysis and embedding generation
Orchestrates chunking, encoding, storage, and indexing with parallel processing
"""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import logging
import json
import yaml
import hashlib
from collections import defaultdict, deque
from queue import Queue, Empty
import threading
import time
import traceback
from tqdm import tqdm
import numpy as np

# Internal imports
from src.chunkers.registry import ChunkerRegistry
from src.core.base_chunker import Chunk, ChunkType
from src.core.file_context import FileContext
from src.classifiers.file_classifier import FileClassifier, FileClassification
from src.classifiers.content_analyzer import ContentAnalyzer
from src.classifiers.pattern_detector import PatternDetector
from src.embeddings.codebert_encoder import CodeBERTEncoder, EncoderConfig
#from src.embeddings.embedding_storage import EmbeddingStorage, StorageConfig
from src.storage.storage_manager import StorageManager
from src.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Processing pipeline stages"""
    FILE_DISCOVERY = "file_discovery"
    FILE_CLASSIFICATION = "file_classification"
    CONTENT_ANALYSIS = "content_analysis"
    CHUNKING = "chunking"
    PATTERN_DETECTION = "pattern_detection"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    INDEXING = "indexing"
    VALIDATION = "validation"
    COMPLETE = "complete"

class ProcessingStatus(Enum):
    """Processing status for files/chunks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

@dataclass
class ProcessingConfig:
    """Configuration for chunk processor"""
    # Paths
    repository_path: Path
    output_path: Path = Path("output")
    cache_path: Path = Path(".cache")
    
    # Processing options
    max_workers: int = mp.cpu_count()
    use_multiprocessing: bool = True
    batch_size: int = 100
    chunk_batch_size: int = 50
    
    # File filters
    include_patterns: List[str] = field(default_factory=lambda: ["*.py", "*.java", "*.js", "*.ts"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["*test*", "*node_modules*", "*venv*"])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    skip_binary: bool = True
    skip_generated: bool = True
    
    # Chunking options
    max_chunk_size: int = 450
    min_chunk_size: int = 50
    overlap_size: int = 50
    preserve_structure: bool = True
    
    # Embedding options
    model_name: str = "microsoft/codebert-base"
    embedding_batch_size: int = 32
    use_gpu: bool = True
    cache_embeddings: bool = True
    
    # Storage options
    storage_backend: str = "hdf5"
    enable_compression: bool = True
    enable_indexing: bool = True
    
    # Pattern detection
    enable_pattern_detection: bool = True
    pattern_confidence_threshold: float = 0.7
    
    # Progress and logging
    show_progress: bool = True
    verbose: bool = False
    save_intermediate: bool = True
    checkpoint_interval: int = 1000
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_queue_size: int = 1000

@dataclass
class FileProcessingResult:
    """Result of processing a single file"""
    file_path: Path
    status: ProcessingStatus
    classification: Optional[FileClassification] = None
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    patterns: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

@dataclass
class ProcessingStatistics:
    """Overall processing statistics"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_patterns: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    stage_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)

class ChunkProcessor:
    """
    Main chunk processing pipeline
    Orchestrates the entire processing workflow
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize chunk processor
        
        Args:
            config: Processing configuration
        """
        self.config = config
        
        # Initialize components
        self._init_components()
        
        # Processing state
        self.file_queue = Queue(maxsize=config.max_queue_size)
        self.chunk_queue = Queue(maxsize=config.max_queue_size)
        self.results = {}
        self.statistics = ProcessingStatistics()
        self.processing_lock = threading.Lock()
        
        # Checkpoint management
        self.checkpoint_path = config.output_path / "checkpoint.json"
        self.processed_files = set()
        self._load_checkpoint()
        
        # Thread pool for I/O operations
        self.io_executor = ThreadPoolExecutor(max_workers=config.max_workers // 2)
        
        # Process pool for CPU-intensive operations
        if config.use_multiprocessing:
            self.cpu_executor = ProcessPoolExecutor(max_workers=config.max_workers)
        else:
            self.cpu_executor = ThreadPoolExecutor(max_workers=config.max_workers)
    
    def _init_components(self):
        """Initialize processing components"""
        logger.info("Initializing processing components...")
        
        # File classifier
        self.file_classifier = FileClassifier()
        
        # Content analyzer
        self.content_analyzer = ContentAnalyzer()
        
        # Pattern detector
        if self.config.enable_pattern_detection:
            self.pattern_detector = PatternDetector()
        
        # Chunker registry
        self.chunker_registry = ChunkerRegistry()
        
        # CodeBERT encoder
        encoder_config = EncoderConfig(
            model_name=self.config.model_name,
            use_cuda=self.config.use_gpu,
            batch_size=self.config.embedding_batch_size,
            cache_embeddings=self.config.cache_embeddings,
            cache_dir=self.config.cache_path / "embeddings"
        )
        self.encoder = CodeBERTEncoder(encoder_config)
        
        # Embedding storage
        storage_config = StorageConfig(
            backend=self.config.storage_backend,
            storage_path=self.config.output_path / "embeddings",
            compression="gzip" if self.config.enable_compression else "none",
            enable_versioning=True
        )
        #self.storage = EmbeddingStorage(storage_config)
        self.storage = StorageManager(self.config.output_path)
        
        # Metrics collector
        self.metrics = MetricsCollector()
    
    def process_repository(self, 
                          repository_path: Optional[Path] = None,
                          resume: bool = True) -> ProcessingStatistics:
        """
        Process entire repository
        
        Args:
            repository_path: Path to repository (uses config if not provided)
            resume: Resume from checkpoint if available
            
        Returns:
            Processing statistics
        """
        repository_path = repository_path or self.config.repository_path
        
        logger.info(f"Processing repository: {repository_path}")
        start_time = time.time()
        
        try:
            # Stage 1: File discovery
            files = self._discover_files(repository_path, resume)
            self.statistics.total_files = len(files)
            
            # Stage 2: Process files in pipeline
            self._process_pipeline(files)
            
            # Stage 3: Build search index
            if self.config.enable_indexing:
                self._build_search_index()
            
            # Stage 4: Validation
            self._validate_results()
            
            # Calculate final statistics
            self.statistics.processing_time = time.time() - start_time
            
            # Save final results
            self._save_results()
            
            logger.info(f"Processing completed: {self.statistics.processed_files}/{self.statistics.total_files} files")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.statistics.errors.append(str(e))
            if not self.config.continue_on_error:
                raise
        
        finally:
            # Cleanup
            self._cleanup()
        
        return self.statistics
    
    def _discover_files(self, repository_path: Path, resume: bool) -> List[Path]:
        """Discover files to process"""
        logger.info("Discovering files...")
        
        files = []
        
        # Get all files matching patterns
        for pattern in self.config.include_patterns:
            for file_path in repository_path.rglob(pattern):
                if file_path.is_file():
                    # Apply exclusion patterns
                    if any(file_path.match(exclude) for exclude in self.config.exclude_patterns):
                        continue
                    
                    # Skip if already processed (when resuming)
                    if resume and str(file_path) in self.processed_files:
                        continue
                    
                    # Check file size
                    if file_path.stat().st_size > self.config.max_file_size:
                        logger.debug(f"Skipping large file: {file_path}")
                        continue
                    
                    files.append(file_path)
        
        logger.info(f"Discovered {len(files)} files to process")
        return files
    
    def _process_pipeline(self, files: List[Path]):
        """Process files through the pipeline"""
        
        # Create progress bar
        if self.config.show_progress:
            pbar = tqdm(total=len(files), desc="Processing files")
        
        # Process in batches
        for i in range(0, len(files), self.config.batch_size):
            batch = files[i:i+self.config.batch_size]
            
            # Process batch
            batch_results = self._process_batch(batch)
            
            # Update results
            for file_path, result in batch_results.items():
                self.results[str(file_path)] = result
                
                if result.status == ProcessingStatus.COMPLETED:
                    self.statistics.processed_files += 1
                elif result.status == ProcessingStatus.FAILED:
                    self.statistics.failed_files += 1
                elif result.status == ProcessingStatus.SKIPPED:
                    self.statistics.skipped_files += 1
            
            # Update progress
            if self.config.show_progress:
                pbar.update(len(batch))
            
            # Save checkpoint
            if (i + len(batch)) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
        
        if self.config.show_progress:
            pbar.close()
    
    def _process_batch(self, files: List[Path]) -> Dict[Path, FileProcessingResult]:
        """Process a batch of files"""
        results = {}
        
        # Stage 1: Classify files
        classifications = self._classify_files_batch(files)
        
        # Stage 2: Analyze content
        analyses = self._analyze_content_batch(files, classifications)
        
        # Stage 3: Chunk files
        chunks_map = self._chunk_files_batch(files, classifications, analyses)
        
        # Stage 4: Detect patterns (if enabled)
        patterns_map = {}
        if self.config.enable_pattern_detection:
            patterns_map = self._detect_patterns_batch(chunks_map)
        
        # Stage 5: Generate embeddings
        embeddings_map = self._generate_embeddings_batch(chunks_map)
        
        # Stage 6: Store embeddings
        self._store_embeddings_batch(chunks_map, embeddings_map)
        
        # Compile results
        for file_path in files:
            result = FileProcessingResult(
                file_path=file_path,
                status=ProcessingStatus.COMPLETED,
                classification=classifications.get(file_path),
                chunks=chunks_map.get(file_path, []),
                embeddings=embeddings_map.get(file_path, []),
                patterns=patterns_map.get(file_path, [])
            )
            
            results[file_path] = result
            self.processed_files.add(str(file_path))
        
        return results
    
    def _classify_files_batch(self, files: List[Path]) -> Dict[Path, FileClassification]:
        """Classify a batch of files"""
        classifications = {}
        
        with self.metrics.measure_time("file_classification"):
            futures = []
            
            for file_path in files:
                future = self.io_executor.submit(self._classify_file, file_path)
                futures.append((file_path, future))
            
            for file_path, future in futures:
                try:
                    classification = future.result(timeout=10)
                    classifications[file_path] = classification
                except Exception as e:
                    logger.error(f"Failed to classify {file_path}: {e}")
                    if not self.config.continue_on_error:
                        raise
        
        return classifications
    
    def _classify_file(self, file_path: Path) -> FileClassification:
        """Classify a single file"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1KB for classification
            
            # Classify
            classification = self.file_classifier.classify(file_path, content)
            
            # Skip if needed
            if self.config.skip_generated and classification.is_generated:
                raise ValueError("Generated file")
            
            if self.config.skip_binary and classification.category == "binary":
                raise ValueError("Binary file")
            
            return classification
            
        except Exception as e:
            logger.debug(f"Classification failed for {file_path}: {e}")
            raise
    
    def _analyze_content_batch(self, 
                              files: List[Path],
                              classifications: Dict[Path, FileClassification]) -> Dict[Path, Any]:
        """Analyze content of files"""
        analyses = {}
        
        with self.metrics.measure_time("content_analysis"):
            for file_path in files:
                if file_path not in classifications:
                    continue
                
                try:
                    # Read content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Analyze
                    analysis = self.content_analyzer.analyze(content, file_path)
                    analyses[file_path] = analysis
                    
                except Exception as e:
                    logger.debug(f"Content analysis failed for {file_path}: {e}")
                    if not self.config.continue_on_error:
                        raise
        
        return analyses
    
    def _chunk_files_batch(self,
                          files: List[Path],
                          classifications: Dict[Path, FileClassification],
                          analyses: Dict[Path, Any]) -> Dict[Path, List[Chunk]]:
        """Chunk files into semantic units"""
        chunks_map = {}
        
        with self.metrics.measure_time("chunking"):
            futures = []
            
            for file_path in files:
                if file_path not in classifications:
                    continue
                
                future = self.cpu_executor.submit(
                    self._chunk_file,
                    file_path,
                    classifications[file_path],
                    analyses.get(file_path)
                )
                futures.append((file_path, future))
            
            for file_path, future in futures:
                try:
                    chunks = future.result(timeout=30)
                    chunks_map[file_path] = chunks
                    self.statistics.total_chunks += len(chunks)
                except Exception as e:
                    logger.error(f"Chunking failed for {file_path}: {e}")
                    if not self.config.continue_on_error:
                        raise
        
        return chunks_map
    
    def _chunk_file(self,
                   file_path: Path,
                   classification: FileClassification,
                   analysis: Optional[Any]) -> List[Chunk]:
        """Chunk a single file"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Create file context
            file_context = FileContext.from_file(file_path, read_content=False)
            
            # Get appropriate chunker
            chunker = self.chunker_registry.get_chunker(
                file_context,
                force_chunker=analysis.suggested_chunker if analysis else None
            )
            
            # Chunk the file
            chunks = chunker.chunk(content, file_context)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata.language = classification.language
                chunk.metadata.framework = classification.framework
                chunk.metadata.importance = classification.importance.value
                chunk.metadata.annotations['file_type'] = classification.file_type
                chunk.metadata.annotations['project_type'] = classification.project_type
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk {file_path}: {e}")
            raise
    
    def _detect_patterns_batch(self,
                              chunks_map: Dict[Path, List[Chunk]]) -> Dict[Path, List[Any]]:
        """Detect patterns in chunks"""
        patterns_map = {}
        
        with self.metrics.measure_time("pattern_detection"):
            for file_path, chunks in chunks_map.items():
                patterns = []
                
                for chunk in chunks:
                    try:
                        # Detect patterns in chunk
                        analysis = self.pattern_detector.analyze(
                            chunk.content,
                            language=chunk.metadata.language
                        )
                        
                        # Filter by confidence
                        filtered_patterns = [
                            p for p in analysis.patterns_found
                            if p.confidence >= self.config.pattern_confidence_threshold
                        ]
                        
                        patterns.extend(filtered_patterns)
                        
                        # Add patterns to chunk metadata
                        chunk.metadata.patterns = [p.pattern_type.value for p in filtered_patterns]
                        
                    except Exception as e:
                        logger.debug(f"Pattern detection failed for chunk: {e}")
                
                patterns_map[file_path] = patterns
                self.statistics.total_patterns += len(patterns)
        
        return patterns_map
    
    def _generate_embeddings_batch(self,
                                  chunks_map: Dict[Path, List[Chunk]]) -> Dict[Path, List[np.ndarray]]:
        """Generate embeddings for chunks"""
        embeddings_map = {}
        
        with self.metrics.measure_time("embedding_generation"):
            # Flatten chunks for batch processing
            all_chunks = []
            chunk_sources = []
            
            for file_path, chunks in chunks_map.items():
                for chunk in chunks:
                    all_chunks.append(chunk.content)
                    chunk_sources.append((file_path, chunk))
            
            # Process in batches
            all_embeddings = []
            
            for i in range(0, len(all_chunks), self.config.chunk_batch_size):
                batch = all_chunks[i:i+self.config.chunk_batch_size]
                
                try:
                    # Generate embeddings
                    result = self.encoder.encode(
                        batch,
                        batch_size=self.config.embedding_batch_size,
                        show_progress=False
                    )
                    
                    all_embeddings.extend(result.embeddings)
                    
                except Exception as e:
                    logger.error(f"Embedding generation failed: {e}")
                    # Use zero embeddings as fallback
                    all_embeddings.extend([np.zeros(768) for _ in batch])
            
            # Map embeddings back to files
            idx = 0
            for file_path, chunks in chunks_map.items():
                file_embeddings = []
                
                for chunk in chunks:
                    if idx < len(all_embeddings):
                        embedding = all_embeddings[idx]
                        chunk.embedding = embedding
                        file_embeddings.append(embedding)
                        idx += 1
                
                embeddings_map[file_path] = file_embeddings
            
            self.statistics.total_embeddings += len(all_embeddings)
        
        return embeddings_map
    
#    def _store_embeddings_batch(self,
#                               chunks_map: Dict[Path, List[Chunk]],
#                               embeddings_map: Dict[Path, List[np.ndarray]]):
#        """Store embeddings in storage backend"""
#        with self.metrics.measure_time("embedding_storage"):
#            for file_path, chunks in chunks_map.items():
#                embeddings = embeddings_map.get(file_path, [])
#                
#                for chunk, embedding in zip(chunks, embeddings):
#                    try:
#                        # Store in database
#                        self.storage.store(
#                            chunk_id=chunk.id,
#                            embedding=embedding,
#                            file_path=str(file_path),
#                            model_name=self.config.model_name,
#                            language=chunk.metadata.language,
#                            chunk_type=chunk.chunk_type.value,
#                            chunk_size=len(chunk.content)
#                        )
#                    except Exception as e:
#                        logger.error(f"Failed to store embedding: {e}")
    def _store_embeddings_batch(self,
                                chunks_map: Dict[Path, List[Chunk]],
                                embeddings_map: Dict[Path, List[np.ndarray]]):
            """Store embeddings, content, and metadata using unified StorageManager"""
            with self.metrics.measure_time("embedding_storage"):
                all_chunks = []
                all_embeddings = []
                
                for file_path, chunks in chunks_map.items():
                    embeddings = embeddings_map.get(file_path, [])
                    
                    # Robust alignment check
                    if len(chunks) != len(embeddings):
                        logger.error(f"Alignment error for {file_path}: {len(chunks)} chunks vs {len(embeddings)} embeddings")
                        # Fallback: store what we can or skip
                        continue
                        
                    # Assign embeddings to chunks (in memory update)
                    for chunk, embedding in zip(chunks, embeddings):
                        chunk.embedding = embedding
                        
                    all_chunks.extend(chunks)
                    all_embeddings.extend(embeddings)

                if all_chunks:
                    try:
                        # Single atomic-like batch save
                        self.storage.batch_save(all_chunks, all_embeddings)
                    except Exception as e:
                        logger.error(f"Failed to store batch of {len(all_chunks)} chunks: {e}")
    
    def _build_search_index(self):
        """Build search index for embeddings"""
        logger.info("Building search index...")
        
        with self.metrics.measure_time("index_building"):
            try:
                self.storage.build_index(index_type="IVF1024,PQ64")
                logger.info("Search index built successfully")
            except Exception as e:
                logger.error(f"Failed to build search index: {e}")
    
    def _validate_results(self):
        """Validate processing results"""
        logger.info("Validating results...")
        
        with self.metrics.measure_time("validation"):
            # Check for missing embeddings
            missing_embeddings = 0
            for file_path, result in self.results.items():
                if result.status == ProcessingStatus.COMPLETED:
                    if len(result.chunks) != len(result.embeddings):
                        missing_embeddings += 1
                        logger.warning(f"Missing embeddings for {file_path}")
            
            if missing_embeddings > 0:
                logger.warning(f"Found {missing_embeddings} files with missing embeddings")
            
            # Validate storage
            storage_stats = self.storage.get_statistics()
            logger.info(f"Storage contains {storage_stats['total_embeddings']} embeddings")
    
    def _save_checkpoint(self):
        """Save processing checkpoint"""
        checkpoint = {
            'processed_files': list(self.processed_files),
            'statistics': asdict(self.statistics),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.debug(f"Checkpoint saved: {len(self.processed_files)} files processed")
    
    def _load_checkpoint(self):
        """Load processing checkpoint"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                
                self.processed_files = set(checkpoint.get('processed_files', []))
                
                # Restore statistics
                for key, value in checkpoint.get('statistics', {}).items():
                    if hasattr(self.statistics, key):
                        setattr(self.statistics, key, value)
                
                logger.info(f"Checkpoint loaded: {len(self.processed_files)} files already processed")
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
    
    def _save_results(self):
        """Save processing results"""
        logger.info("Saving results...")
        
        # Save statistics
        stats_path = self.config.output_path / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(asdict(self.statistics), f, indent=2, default=str)
        
        # Save detailed results
        if self.config.save_intermediate:
            results_path = self.config.output_path / "results.json"
            
            # Convert results to serializable format
            serializable_results = {}
            for file_path, result in self.results.items():
                serializable_results[file_path] = {
                    'status': result.status.value,
                    'chunks': len(result.chunks),
                    'embeddings': len(result.embeddings),
                    'patterns': len(result.patterns),
                    'processing_time': result.processing_time,
                    'errors': result.errors
                }
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_path}")
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        # Shutdown executors
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        
        # Close storage
        self.storage.close()
        
        # Save encoder cache
        self.encoder.save_cache()
        
        # Clear queues
        while not self.file_queue.empty():
            try:
                self.file_queue.get_nowait()
            except Empty:
                break
        
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except Empty:
                break
    
    def search_similar_code(self, 
                           query: str,
                           k: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar code chunks
        
        Args:
            query: Query code snippet
            k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of (chunk_id, similarity_score, metadata) tuples
        """
        # Generate query embedding
        query_result = self.encoder.encode(query)
        query_embedding = query_result.embeddings
        
        # Search in storage
        results = self.storage.search(query_embedding, k=k, filters=filters)
        
        return results
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get detailed processing metrics"""
        return {
            'statistics': asdict(self.statistics),
            'encoder_stats': self.encoder.get_statistics(),
            'storage_stats': self.storage.get_statistics(),
            'metrics': self.metrics.get_all_metrics()
        }

# Convenience functions
def process_repository(repository_path: Union[str, Path],
                      output_path: Union[str, Path] = "output",
                      **kwargs) -> ProcessingStatistics:
    """
    Process a repository with default settings
    
    Args:
        repository_path: Path to repository
        output_path: Output directory
        **kwargs: Additional configuration options
        
    Returns:
        Processing statistics
    """
    config = ProcessingConfig(
        repository_path=Path(repository_path),
        output_path=Path(output_path),
        **kwargs
    )
    
    processor = ChunkProcessor(config)
    return processor.process_repository()

def process_file(file_path: Union[str, Path],
                model_name: str = "microsoft/codebert-base") -> FileProcessingResult:
    """
    Process a single file
    
    Args:
        file_path: Path to file
        model_name: Embedding model name
        
    Returns:
        Processing result
    """
    config = ProcessingConfig(
        repository_path=Path(file_path).parent,
        output_path=Path("output"),
        model_name=model_name
    )
    
    processor = ChunkProcessor(config)
    
    # Process single file
    results = processor._process_batch([Path(file_path)])
    
    return results.get(Path(file_path))