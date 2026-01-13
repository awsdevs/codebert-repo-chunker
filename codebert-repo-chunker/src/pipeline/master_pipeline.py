"""
Master pipeline orchestrator for large-scale repository processing
Coordinates all processing stages with advanced scheduling and monitoring
"""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import json
import yaml
import hashlib
import pickle
import shutil
import os
import sys
import psutil
import signal
import time
from collections import defaultdict, deque
from queue import PriorityQueue, Queue, Empty
import threading
import subprocess
from contextlib import contextmanager
import tempfile
import traceback

# External dependencies
import schedule
import redis
from celery import Celery
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import numpy as np
from tqdm import tqdm
import git

# Internal imports
from src.pipeline.chunk_processor import ChunkProcessor, ProcessingConfig, ProcessingStatistics
from src.pipeline.repository_scanner import RepositoryScanner
from src.pipeline.dependency_resolver import DependencyResolver
from src.pipeline.quality_analyzer import QualityAnalyzer
from src.pipeline.report_generator import ReportGenerator
from src.utils.monitoring import MetricsExporter, HealthChecker
from src.utils.notifications import NotificationManager

logger = logging.getLogger(__name__)

# Prometheus metrics
FILES_PROCESSED = Counter('files_processed_total', 'Total number of files processed')
CHUNKS_CREATED = Counter('chunks_created_total', 'Total number of chunks created')
EMBEDDINGS_GENERATED = Counter('embeddings_generated_total', 'Total embeddings generated')
PROCESSING_TIME = Histogram('processing_duration_seconds', 'Processing time per file')
PIPELINE_STATUS = Gauge('pipeline_status', 'Pipeline status (0=stopped, 1=running, 2=error)')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
QUEUE_SIZE = Gauge('queue_size', 'Number of items in processing queue')

class PipelineStage(Enum):
    """Pipeline execution stages"""
    INITIALIZATION = "initialization"
    REPOSITORY_SCAN = "repository_scan"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    FILE_PRIORITIZATION = "file_prioritization"
    CHUNK_PROCESSING = "chunk_processing"
    QUALITY_ANALYSIS = "quality_analysis"
    EMBEDDING_GENERATION = "embedding_generation"
    INDEX_BUILDING = "index_building"
    REPORT_GENERATION = "report_generation"
    CLEANUP = "cleanup"
    COMPLETE = "complete"

class ExecutionMode(Enum):
    """Pipeline execution modes"""
    FULL = "full"              # Process entire repository
    INCREMENTAL = "incremental" # Process only changes
    SELECTIVE = "selective"     # Process selected files
    CONTINUOUS = "continuous"   # Continuous monitoring
    DISTRIBUTED = "distributed" # Distributed processing

class PipelineStatus(Enum):
    """Pipeline status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class MasterConfig:
    """Master pipeline configuration"""
    # Repository settings
    repository_path: Path
    repository_url: Optional[str] = None
    branch: str = "main"
    
    # Output settings
    output_base: Path = Path("pipeline_output")
    workspace: Path = Path(".pipeline_workspace")
    
    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.FULL
    max_workers: int = mp.cpu_count()
    distributed: bool = False
    
    # Processing settings
    chunk_size: int = 450
    batch_size: int = 100
    parallel_stages: bool = True
    
    # Model settings
    models: List[str] = field(default_factory=lambda: ["microsoft/codebert-base"])
    embedding_dimensions: int = 768
    
    # Quality settings
    quality_threshold: float = 0.7
    enable_quality_checks: bool = True
    enable_pattern_detection: bool = True
    
    # Performance settings
    memory_limit_gb: float = 16.0
    cpu_limit_percent: float = 80.0
    io_limit_mbps: float = 100.0
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 60
    
    # Notification settings
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Scheduling settings
    schedule_enabled: bool = False
    schedule_cron: str = "0 2 * * *"  # Daily at 2 AM
    
    # Caching settings
    enable_caching: bool = True
    cache_dir: Path = Path(".cache")
    cache_ttl_hours: int = 72
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0
    exponential_backoff: bool = True
    
    # Checkpointing
    checkpoint_interval: int = 1000
    enable_checkpointing: bool = True
    
    # Debug settings
    debug: bool = False
    profile: bool = False
    trace: bool = False

@dataclass
class PipelineState:
    """Current pipeline state"""
    status: PipelineStatus = PipelineStatus.IDLE
    current_stage: Optional[PipelineStage] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Progress tracking
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_gb: float = 0.0
    
    # Stage timings
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Results
    total_chunks: int = 0
    total_embeddings: int = 0
    total_patterns: int = 0
    quality_score: float = 0.0

class MasterPipeline:
    """
    Master pipeline orchestrator
    Manages the entire processing workflow with advanced features
    """
    
    def __init__(self, config: MasterConfig):
        """
        Initialize master pipeline
        
        Args:
            config: Master configuration
        """
        self.config = config
        self.state = PipelineState()
        
        # Initialize components
        self._init_workspace()
        self._init_components()
        self._init_monitoring()
        
        # Processing infrastructure
        self.task_queue = PriorityQueue()
        self.result_queue = Queue()
        self.worker_pool = []
        
        # Control flags
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Master pipeline initialized")
    
    def _init_workspace(self):
        """Initialize workspace directories"""
        self.config.workspace.mkdir(parents=True, exist_ok=True)
        self.config.output_base.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.paths = {
            'chunks': self.config.workspace / 'chunks',
            'embeddings': self.config.workspace / 'embeddings',
            'indices': self.config.workspace / 'indices',
            'reports': self.config.output_base / 'reports',
            'logs': self.config.output_base / 'logs',
            'checkpoints': self.config.workspace / 'checkpoints',
            'temp': self.config.workspace / 'temp'
        }
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _init_components(self):
        """Initialize pipeline components"""
        # Repository scanner
        self.scanner = RepositoryScanner(
            cache_dir=self.config.cache_dir,
            enable_git_analysis=True
        )
        
        # Dependency resolver
        self.dependency_resolver = DependencyResolver()
        
        # Quality analyzer
        self.quality_analyzer = QualityAnalyzer(
            threshold=self.config.quality_threshold
        )
        
        # Report generator
        self.report_generator = ReportGenerator(
            output_dir=self.paths['reports']
        )
        
        # Chunk processor
        processor_config = ProcessingConfig(
            repository_path=self.config.repository_path,
            output_path=self.paths['embeddings'],
            cache_path=self.config.cache_dir,
            max_workers=self.config.max_workers,
            enable_pattern_detection=self.config.enable_pattern_detection
        )
        self.chunk_processor = ChunkProcessor(processor_config)
        
        # Notification manager
        if self.config.enable_notifications:
            self.notifier = NotificationManager(
                channels=self.config.notification_channels
            )
        
        # Redis connection for distributed mode
        if self.config.distributed:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
            
            # Celery app for distributed tasks
            self.celery_app = Celery(
                'master_pipeline',
                broker='redis://localhost:6379',
                backend='redis://localhost:6379'
            )
    
    def _init_monitoring(self):
        """Initialize monitoring components"""
        if self.config.enable_monitoring:
            # Start Prometheus metrics server
            start_http_server(self.config.metrics_port)
            
            # Health checker
            self.health_checker = HealthChecker(
                check_interval=self.config.health_check_interval
            )
            
            # Metrics exporter
            self.metrics_exporter = MetricsExporter()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitor_thread.start()
    
    async def run(self) -> PipelineState:
        """
        Run the master pipeline
        
        Returns:
            Final pipeline state
        """
        logger.info("Starting master pipeline")
        self.state.status = PipelineStatus.RUNNING
        self.state.start_time = datetime.now(timezone.utc)
        PIPELINE_STATUS.set(1)
        
        try:
            # Execute pipeline stages
            if self.config.execution_mode == ExecutionMode.CONTINUOUS:
                await self._run_continuous()
            elif self.config.execution_mode == ExecutionMode.DISTRIBUTED:
                await self._run_distributed()
            else:
                await self._run_standard()
            
            self.state.status = PipelineStatus.COMPLETED
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.state.status = PipelineStatus.ERROR
            self.state.errors.append(str(e))
            PIPELINE_STATUS.set(2)
            
            if self.config.enable_notifications:
                self.notifier.send_error(f"Pipeline failed: {e}")
            
            raise
        
        finally:
            self.state.end_time = datetime.now(timezone.utc)
            await self._cleanup()
            
            # Generate final report
            self._generate_final_report()
            
            # Send completion notification
            if self.config.enable_notifications:
                self._send_completion_notification()
        
        return self.state
    
    async def _run_standard(self):
        """Run standard pipeline execution"""
        
        # Stage 1: Repository scan
        await self._execute_stage(
            PipelineStage.REPOSITORY_SCAN,
            self._scan_repository
        )
        
        # Stage 2: Dependency analysis
        await self._execute_stage(
            PipelineStage.DEPENDENCY_ANALYSIS,
            self._analyze_dependencies
        )
        
        # Stage 3: File prioritization
        await self._execute_stage(
            PipelineStage.FILE_PRIORITIZATION,
            self._prioritize_files
        )
        
        # Stage 4: Chunk processing
        await self._execute_stage(
            PipelineStage.CHUNK_PROCESSING,
            self._process_chunks
        )
        
        # Stage 5: Quality analysis
        if self.config.enable_quality_checks:
            await self._execute_stage(
                PipelineStage.QUALITY_ANALYSIS,
                self._analyze_quality
            )
        
        # Stage 6: Embedding generation
        await self._execute_stage(
            PipelineStage.EMBEDDING_GENERATION,
            self._generate_embeddings
        )
        
        # Stage 7: Index building
        await self._execute_stage(
            PipelineStage.INDEX_BUILDING,
            self._build_indices
        )
        
        # Stage 8: Report generation
        await self._execute_stage(
            PipelineStage.REPORT_GENERATION,
            self._generate_reports
        )
    
    async def _run_continuous(self):
        """Run continuous monitoring pipeline"""
        logger.info("Starting continuous monitoring mode")
        
        while not self._stop_flag.is_set():
            try:
                # Check for repository changes
                changes = await self._detect_changes()
                
                if changes:
                    logger.info(f"Detected {len(changes)} changes")
                    
                    # Process changes incrementally
                    await self._process_incremental(changes)
                    
                    # Update indices
                    await self._update_indices()
                
                # Wait for next check interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in continuous mode: {e}")
                await asyncio.sleep(10)
    
    async def _run_distributed(self):
        """Run distributed pipeline execution"""
        logger.info("Starting distributed processing mode")
        
        # Initialize distributed workers
        await self._init_distributed_workers()
        
        # Distribute tasks
        await self._distribute_tasks()
        
        # Monitor progress
        await self._monitor_distributed_progress()
        
        # Collect results
        await self._collect_distributed_results()
    
    async def _execute_stage(self, stage: PipelineStage, func: Callable):
        """Execute a pipeline stage with monitoring"""
        logger.info(f"Executing stage: {stage.value}")
        self.state.current_stage = stage
        
        start_time = time.time()
        
        try:
            # Check for pause/stop
            await self._check_control_flags()
            
            # Execute stage function
            result = await func()
            
            # Record timing
            duration = time.time() - start_time
            self.state.stage_times[stage.value] = duration
            
            logger.info(f"Stage {stage.value} completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")
            self.state.errors.append(f"{stage.value}: {str(e)}")
            
            if not self.config.execution_mode == ExecutionMode.CONTINUOUS:
                raise
    
    async def _scan_repository(self):
        """Scan repository for files"""
        logger.info(f"Scanning repository: {self.config.repository_path}")
        
        # Clone/update repository if URL provided
        if self.config.repository_url:
            await self._update_repository()
        
        # Scan for files
        scan_result = self.scanner.scan(
            self.config.repository_path,
            incremental=(self.config.execution_mode == ExecutionMode.INCREMENTAL)
        )
        
        self.state.total_files = scan_result['total_files']
        self.files_to_process = scan_result['files']
        
        logger.info(f"Found {self.state.total_files} files to process")
        
        # Save scan results
        scan_path = self.paths['checkpoints'] / 'repository_scan.json'
        with open(scan_path, 'w') as f:
            json.dump(scan_result, f, indent=2, default=str)
        
        FILES_PROCESSED.inc(0)  # Initialize counter
    
    async def _analyze_dependencies(self):
        """Analyze file dependencies"""
        logger.info("Analyzing dependencies")
        
        dependencies = self.dependency_resolver.analyze(
            self.files_to_process,
            self.config.repository_path
        )
        
        self.dependency_graph = dependencies['graph']
        self.import_map = dependencies['imports']
        
        # Save dependency analysis
        dep_path = self.paths['checkpoints'] / 'dependencies.json'
        with open(dep_path, 'w') as f:
            json.dump(dependencies, f, indent=2, default=str)
        
        logger.info(f"Analyzed dependencies for {len(self.dependency_graph)} files")
    
    async def _prioritize_files(self):
        """Prioritize files for processing"""
        logger.info("Prioritizing files")
        
        # Calculate priority scores
        priorities = []
        
        for file_path in self.files_to_process:
            score = self._calculate_priority(file_path)
            priorities.append((score, file_path))
        
        # Sort by priority (higher first)
        priorities.sort(reverse=True)
        
        # Update processing order
        self.processing_order = [p[1] for p in priorities]
        
        logger.info(f"Files prioritized, top priority: {priorities[0][1] if priorities else 'None'}")
    
    def _calculate_priority(self, file_path: Path) -> float:
        """Calculate file processing priority"""
        score = 0.0
        
        # File importance factors
        if 'main' in file_path.name.lower():
            score += 10.0
        if 'index' in file_path.name.lower():
            score += 8.0
        if 'app' in file_path.name.lower():
            score += 7.0
        if 'service' in file_path.name.lower():
            score += 6.0
        if 'model' in file_path.name.lower():
            score += 5.0
        
        # Test files have lower priority
        if 'test' in str(file_path).lower():
            score -= 5.0
        
        # Vendor/node_modules have lowest priority
        if 'vendor' in str(file_path).lower() or 'node_modules' in str(file_path):
            score -= 10.0
        
        # Dependency count (files with more dependents are higher priority)
        if hasattr(self, 'dependency_graph') and str(file_path) in self.dependency_graph:
            score += len(self.dependency_graph[str(file_path)]) * 2
        
        # File size (prefer smaller files for faster initial results)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb < 1:
                score += 2.0
            elif size_mb > 10:
                score -= 2.0
        
        return score
    
    async def _process_chunks(self):
        """Process files into chunks"""
        logger.info("Processing chunks")
        
        # Process files in batches
        batch_size = self.config.batch_size
        total_batches = (len(self.processing_order) + batch_size - 1) // batch_size
        
        with tqdm(total=len(self.processing_order), desc="Processing files") as pbar:
            for i in range(0, len(self.processing_order), batch_size):
                batch = self.processing_order[i:i+batch_size]
                
                # Check control flags
                await self._check_control_flags()
                
                # Process batch
                await self._process_batch(batch)
                
                # Update progress
                pbar.update(len(batch))
                self.state.processed_files += len(batch)
                FILES_PROCESSED.inc(len(batch))
                
                # Save checkpoint
                if self.state.processed_files % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint()
                
                # Check resource limits
                await self._check_resource_limits()
    
    async def _process_batch(self, batch: List[Path]):
        """Process a batch of files"""
        # Use chunk processor
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            self.chunk_processor._process_batch,
            batch
        )
        
        # Update statistics
        for file_path, result in results.items():
            self.state.total_chunks += len(result.chunks)
            CHUNKS_CREATED.inc(len(result.chunks))
            
            if result.status == "failed":
                self.state.failed_files += 1
                self.state.errors.append(f"Failed to process {file_path}")
    
    async def _analyze_quality(self):
        """Analyze code quality"""
        logger.info("Analyzing code quality")
        
        quality_results = self.quality_analyzer.analyze(
            self.paths['chunks'],
            self.paths['embeddings']
        )
        
        self.state.quality_score = quality_results['overall_score']
        
        # Save quality report
        quality_path = self.paths['reports'] / 'quality_analysis.json'
        with open(quality_path, 'w') as f:
            json.dump(quality_results, f, indent=2)
        
        logger.info(f"Quality score: {self.state.quality_score:.2f}")
        
        # Check quality threshold
        if self.state.quality_score < self.config.quality_threshold:
            self.state.warnings.append(
                f"Quality score {self.state.quality_score:.2f} below threshold {self.config.quality_threshold}"
            )
    
    async def _generate_embeddings(self):
        """Generate embeddings for chunks"""
        logger.info("Generating embeddings")
        
        # This is handled by chunk processor
        # Add any additional embedding processing here
        
        embedding_stats = self.chunk_processor.storage.get_statistics()
        self.state.total_embeddings = embedding_stats['total_embeddings']
        EMBEDDINGS_GENERATED.inc(self.state.total_embeddings)
        
        logger.info(f"Generated {self.state.total_embeddings} embeddings")
    
    async def _build_indices(self):
        """Build search indices"""
        logger.info("Building search indices")
        
        # Build FAISS index
        self.chunk_processor.storage.build_index(index_type="IVF1024,PQ64")
        
        # Build additional indices if needed
        # e.g., Elasticsearch, specialized indices
        
        logger.info("Search indices built successfully")
    
    async def _generate_reports(self):
        """Generate processing reports"""
        logger.info("Generating reports")
        
        report_data = {
            'state': asdict(self.state),
            'config': asdict(self.config),
            'processing_metrics': self.chunk_processor.get_processing_metrics(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Generate HTML report
        html_report = self.report_generator.generate_html_report(report_data)
        html_path = self.paths['reports'] / f"pipeline_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        with open(html_path, 'w') as f:
            f.write(html_report)
        
        # Generate JSON report
        json_path = self.paths['reports'] / f"pipeline_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Reports generated: {html_path.name}, {json_path.name}")
    
    async def _check_control_flags(self):
        """Check for pause/stop signals"""
        if self._stop_flag.is_set():
            logger.info("Stop signal received")
            raise KeyboardInterrupt("Pipeline stopped by user")
        
        while self._pause_flag.is_set():
            logger.info("Pipeline paused...")
            await asyncio.sleep(1)
    
    async def _check_resource_limits(self):
        """Check resource usage against limits"""
        process = psutil.Process(os.getpid())
        
        # Check memory usage
        memory_gb = process.memory_info().rss / (1024 ** 3)
        self.state.memory_usage_mb = memory_gb * 1024
        MEMORY_USAGE.set(process.memory_info().rss)
        
        if memory_gb > self.config.memory_limit_gb:
            logger.warning(f"Memory usage ({memory_gb:.2f}GB) exceeds limit ({self.config.memory_limit_gb}GB)")
            # Trigger garbage collection
            import gc
            gc.collect()
        
        # Check CPU usage
        cpu_percent = process.cpu_percent(interval=1)
        self.state.cpu_usage_percent = cpu_percent
        
        if cpu_percent > self.config.cpu_limit_percent:
            logger.warning(f"CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.config.cpu_limit_percent}%)")
            # Add small delay to reduce CPU usage
            await asyncio.sleep(0.1)
    
    async def _save_checkpoint(self):
        """Save processing checkpoint"""
        checkpoint = {
            'state': asdict(self.state),
            'processed_files': self.state.processed_files,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        checkpoint_path = self.paths['checkpoints'] / 'master_checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.debug(f"Checkpoint saved: {self.state.processed_files}/{self.state.total_files} files")
    
    async def _cleanup(self):
        """Cleanup resources and temporary files"""
        logger.info("Cleaning up")
        
        # Close storage
        self.chunk_processor.storage.close()
        
        # Clean temporary files
        if self.paths['temp'].exists():
            shutil.rmtree(self.paths['temp'], ignore_errors=True)
        
        # Stop monitoring
        if hasattr(self, 'monitor_thread'):
            self._stop_flag.set()
        
        PIPELINE_STATUS.set(0)
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_flag.is_set():
            try:
                # Update metrics
                QUEUE_SIZE.set(self.task_queue.qsize())
                
                # Health check
                if hasattr(self, 'health_checker'):
                    health_status = self.health_checker.check()
                    if not health_status['healthy']:
                        logger.warning(f"Health check failed: {health_status}")
                
                # Export metrics
                if hasattr(self, 'metrics_exporter'):
                    self.metrics_exporter.export()
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _generate_final_report(self):
        """Generate final execution report"""
        duration = (self.state.end_time - self.state.start_time).total_seconds()
        
        report = {
            'execution_summary': {
                'status': self.state.status.value,
                'duration_seconds': duration,
                'files_processed': self.state.processed_files,
                'files_failed': self.state.failed_files,
                'files_skipped': self.state.skipped_files,
                'total_chunks': self.state.total_chunks,
                'total_embeddings': self.state.total_embeddings,
                'quality_score': self.state.quality_score
            },
            'stage_timings': self.state.stage_times,
            'resource_usage': {
                'peak_memory_mb': self.state.memory_usage_mb,
                'peak_cpu_percent': self.state.cpu_usage_percent,
                'disk_usage_gb': self.state.disk_usage_gb
            },
            'errors': self.state.errors,
            'warnings': self.state.warnings
        }
        
        # Save summary
        summary_path = self.paths['reports'] / 'execution_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Status: {self.state.status.value}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Files Processed: {self.state.processed_files}/{self.state.total_files}")
        print(f"Chunks Created: {self.state.total_chunks}")
        print(f"Embeddings Generated: {self.state.total_embeddings}")
        print(f"Quality Score: {self.state.quality_score:.2f}")
        print("="*50)
    
    def _send_completion_notification(self):
        """Send completion notification"""
        if not hasattr(self, 'notifier'):
            return
        
        message = f"""
        Pipeline Execution Completed
        
        Status: {self.state.status.value}
        Files Processed: {self.state.processed_files}/{self.state.total_files}
        Chunks Created: {self.state.total_chunks}
        Embeddings Generated: {self.state.total_embeddings}
        Quality Score: {self.state.quality_score:.2f}
        
        Errors: {len(self.state.errors)}
        Warnings: {len(self.state.warnings)}
        """
        
        self.notifier.send(message, level="info")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logger.info(f"Received signal {signum}")
        self._stop_flag.set()
    
    # Control methods
    def pause(self):
        """Pause pipeline execution"""
        logger.info("Pausing pipeline")
        self._pause_flag.set()
        self.state.status = PipelineStatus.PAUSED
    
    def resume(self):
        """Resume pipeline execution"""
        logger.info("Resuming pipeline")
        self._pause_flag.clear()
        self.state.status = PipelineStatus.RUNNING
    
    def stop(self):
        """Stop pipeline execution"""
        logger.info("Stopping pipeline")
        self._stop_flag.set()
        self.state.status = PipelineStatus.CANCELLED

# CLI and convenience functions
def run_pipeline(repository_path: Union[str, Path],
                output_path: Union[str, Path] = "pipeline_output",
                mode: str = "full",
                **kwargs) -> PipelineState:
    """
    Run the master pipeline
    
    Args:
        repository_path: Path to repository
        output_path: Output directory
        mode: Execution mode (full, incremental, continuous)
        **kwargs: Additional configuration
        
    Returns:
        Pipeline execution state
    """
    config = MasterConfig(
        repository_path=Path(repository_path),
        output_base=Path(output_path),
        execution_mode=ExecutionMode(mode),
        **kwargs
    )
    
    pipeline = MasterPipeline(config)
    
    # Run pipeline
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(pipeline.run())

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Master Pipeline for Repository Processing")
    parser.add_argument("repository", help="Repository path or URL")
    parser.add_argument("-o", "--output", default="pipeline_output", help="Output directory")
    parser.add_argument("-m", "--mode", default="full", 
                       choices=["full", "incremental", "continuous", "distributed"],
                       help="Execution mode")
    parser.add_argument("-w", "--workers", type=int, default=mp.cpu_count(),
                       help="Number of workers")
    parser.add_argument("--model", default="microsoft/codebert-base",
                       help="Embedding model")
    parser.add_argument("--quality-threshold", type=float, default=0.7,
                       help="Quality threshold")
    parser.add_argument("--enable-monitoring", action="store_true",
                       help="Enable monitoring")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    state = run_pipeline(
        args.repository,
        args.output,
        args.mode,
        max_workers=args.workers,
        models=[args.model],
        quality_threshold=args.quality_threshold,
        enable_monitoring=args.enable_monitoring,
        debug=args.debug
    )
    
    # Exit with appropriate code
    sys.exit(0 if state.status == PipelineStatus.COMPLETED else 1)

if __name__ == "__main__":
    main()