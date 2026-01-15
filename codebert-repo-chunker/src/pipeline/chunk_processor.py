
import logging
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field

import yaml
import numpy as np

from src.core.chunk_model import Chunk, ChunkType, ChunkLocation
from src.storage.storage_manager import StorageManager
from src.classifiers.file_classifier import FileClassifier
from src.chunkers.registry import get_registry
from src.embeddings.codebert_encoder import CodeBERTEncoder, EncoderConfig

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for chunk processing"""
    chunk_size: int = 512
    overlap: int = 50
    min_chunk_size: int = 20
    max_workers: int = 4
    file_patterns: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_patterns: List[str] = field(default_factory=list)
    enable_embeddings: bool = True

@dataclass
class ProcessingStatistics:
    """Statistics for a processing run"""
    total_files: int = 0
    total_size: int = 0
    total_chunks: int = 0
    failed_files: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    errors: List[str] = field(default_factory=list)

class ChunkProcessor:
    """
    Main processor for converting files into code chunks.
    Manages the pipeline of: Read -> Parse -> Chunk -> Embed -> Store.
    """
    
    def __init__(self, config: ProcessingConfig, storage_manager: Optional[StorageManager] = None):
        self.config = config
        self.storage_manager = storage_manager
        self.stats = ProcessingStatistics()
        
        # Initialize components
        self.classifier = FileClassifier()
        
        # Initialize Encoder
        self.encoder = None
        self.tokenizer = None
        if self.config.enable_embeddings:
            try:
                # Use default config for now, maybe expose more via ProcessingConfig later
                enc_config = EncoderConfig(use_cuda=False) # Force CPU for safety in verify env unless confident
                self.encoder = CodeBERTEncoder(enc_config)
                self.tokenizer = self.encoder.tokenizer
                logger.info("CodeBERT Encoder initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize CodeBERT encoder: {e}")
        
        # Initialize Registry (pass tokenizer if available)
        self.registry = get_registry(self.tokenizer)
        logger.info("Chunker Registry initialized")

    def process_batch(self, file_paths: List[Path]) -> List[Chunk]:
        """Process a batch of files"""
        chunks = []
        for file_path in file_paths:
            try:
                file_chunks = self.process_file(file_path)
                chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.stats.failed_files += 1
                self.stats.errors.append(f"{file_path}: {e}")
                
        # Batch embedding if supported/needed
        # For now, process_file does embedding per file, which is simpler but less efficient.
        # Ideally we collect all chunks then embed in batch.
        
        if self.encoder and chunks:
             # Extract chunks that need embedding (if any weren't done in process_file)
            pass 
            
        return chunks

    def close(self):
        """Close resources"""
        if self.storage_manager:
            self.storage_manager.close()

    def process_file(self, file_path: Union[Path, str]) -> List[Chunk]:
        """Process a single file: Classify -> Chunk -> Embed -> Store"""
        file_path = Path(file_path)
        if not file_path.exists():
            return []
            
        self.stats.total_files += 1
        
        # 1. Classify
        # We read content once to pass to classifier and chunker
        try:
            content = file_path.read_text(errors='ignore')
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []

        print(f"DEBUG: Classifying {file_path}")
        classification = self.classifier.classify(file_path, content)
        
        # 2. Chunk
        # Use registry to Chunk
        try:
            print(f"DEBUG: Chunking {file_path}")
            chunks = self.registry.chunk_file(file_path, content=content)
        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")
            return []
            
        if not chunks:
            return []

        # 3. Embed
        if self.encoder and chunks:
            try:
                print(f"DEBUG: Embedding {len(chunks)} chunks for {file_path}")
                # Extract text for embedding
                texts = [c.content for c in chunks]
                results = self.encoder.encode(texts)
                
                # Assign embeddings back to chunks
                # results.embeddings is np.ndarray
                for i, chunk in enumerate(chunks):
                    chunk.embedding = results.embeddings[i]
            except Exception as e:
                logger.error(f"Embedding failed for {file_path}: {e}")
                # Continue without embeddings
        
        # 4. Store
        if self.storage_manager:
            for chunk in chunks:
                # Enrich chunk metadata from classification
                if chunk.metadata and hasattr(chunk.metadata, 'annotations'):
                    chunk.metadata.annotations['language'] = str(classification.technology_stack)
                    chunk.metadata.annotations['domain'] = classification.domain.value
                    chunk.metadata.annotations['purpose'] = classification.purpose.value
                elif isinstance(chunk.metadata, dict):
                    chunk.metadata['language'] = str(classification.technology_stack)
                    chunk.metadata['domain'] = classification.domain.value
                    chunk.metadata['purpose'] = classification.purpose.value
                
                self.storage_manager.store_chunk(chunk)
                self.stats.total_chunks += 1
                
        return chunks