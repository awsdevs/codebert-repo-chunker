
from typing import Dict, Any, List, Optional, Union, Generator, Tuple
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from src.utils.logger import get_logger
import json
import shutil

import numpy as np

from src.core.chunk_model import Chunk, ChunkLocation, ChunkType
from src.storage.chunk_storage import ChunkStorage, StorageConfig as ChunkStorageConfig
from src.storage.metadata_store import MetadataStore, MetadataConfig

from src.storage.vector_store import VectorStore, VectorConfig

logger = get_logger(__name__)

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"

@dataclass
class StorageConfig:
    """Configuration for storage backend"""
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    base_path: Path = Path("data")
    primary_backend: str = "sqlite"  # sqlite, postgres, mongo
    enable_caching: bool = True
    enable_vector_search: bool = True
    
    def __post_init__(self):
        self.base_path = Path(self.base_path)
        
class StorageFactory:
    """Factory for creating storage managers"""
    
    @staticmethod
    def create(config: StorageConfig) -> 'StorageManager':
        # Ensure base path exists
        config.base_path.mkdir(parents=True, exist_ok=True)
        return StorageManager(config)

class StorageManager:
    """
    Unified interface for all storage operations.
    Coordinating Chunk, Metadata, and Vector storage.
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self._init_backends()
        
    def _init_backends(self):
        """Initialize storage backends based on config"""
        # 1. Chunk Storage
        chunk_config = ChunkStorageConfig(storage_path=self.config.base_path)
        self.chunk_storage = ChunkStorage(chunk_config)
            
        # 2. Metadata Storage
        meta_config = MetadataConfig(storage_path=self.config.base_path)
        self.metadata_store = MetadataStore(meta_config)
        
        # 3. Vector Storage
        if self.config.enable_vector_search and VectorStore:
            try:
                vec_config = VectorConfig(storage_path=self.config.base_path / "vectors")
                self.vector_store = VectorStore(vec_config)
            except Exception as e:
                logger.warning(f"Vector storage disabled: {e}")
                self.vector_store = None
        else:
            self.vector_store = None
            
    def store_chunk(self, chunk: Chunk) -> bool:
        """Store a chunk across all backends"""
        try:
            # 1. Store Content
            self.chunk_storage.store(
                chunk.id, 
                chunk.content, 
                chunk.location.file_path,
                chunk.language
            )
            
            # 2. Store Metadata (Strip large fields)
            metadata_payload = chunk.to_dict()
            metadata_payload.pop('content', None)
            metadata_payload.pop('embedding', None)
            
            # Helper: Flatten essential location data for easier indexing
            if 'location' in metadata_payload and isinstance(metadata_payload['location'], dict):
                metadata_payload['file_path'] = metadata_payload['location'].get('file_path', '')
                metadata_payload['start_line'] = metadata_payload['location'].get('start_line', 0)
                metadata_payload['end_line'] = metadata_payload['location'].get('end_line', 0)
                
            self.metadata_store.store(chunk.id, metadata_payload)
            
            if self.vector_store and chunk.embedding is not None and np:
                 if isinstance(chunk.embedding, np.ndarray):
                    self.vector_store.add([chunk.id], chunk.embedding.reshape(1, -1))
            
            # Progress log (optional, maybe too noisy if called frequently, but good for debug)
            # logger.debug(f"Successfully stored chunk {chunk.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk.id}: {e}")
            return False
            
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a complete chunk"""
        content = self.chunk_storage.retrieve(chunk_id)
        
        metadata = self.metadata_store.get(chunk_id) or {}
        
        # Need to reconstruct ChunkLocation from metadata
        location = ChunkLocation(
            file_path=metadata.get('file_path', ''),
            start_line=metadata.get('start_line', 0),
            end_line=metadata.get('end_line', 0)
        )

        return Chunk(
            id=chunk_id,
            content=content,
            chunk_type=ChunkType(metadata.get('chunk_type', 'unknown')),
            location=location,
            language=metadata.get('language', 'unknown'),
            metadata=metadata
        )

    def search_by_vector(self, embedding: Any, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks using vector similarity
        Returns hydrated results with content and metadata
        """
        if not self.vector_store:
            logger.warning("Vector search disabled")
            return []
            
        # 1. Get similar chunk IDs
        results = self.vector_store.search(embedding, k=limit)
        
        hydrated_results = []
        for chunk_id, score in results:
            # 2. Get Metadata
            metadata = self.metadata_store.get(chunk_id) or {}
            
            # 3. Get Content (Optional? let's include snippet)
            content = self.chunk_storage.retrieve(chunk_id)
            snippet = content[:200] + "..." if content else ""
            
            hydrated_results.append({
                'id': chunk_id,
                'score': float(score),
                'file_path': metadata.get('file_path') or metadata.get('location', {}).get('file_path'),
                'function_name': metadata.get('function_name'),
                'snippet': snippet,
                'metadata': metadata
            })
            
        return hydrated_results

    def search_text(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Full text search returning (chunk_id, rank)"""
        return self.metadata_store.search_text(query, limit)

    def delete_file_chunks(self, file_path: str) -> int:
        """
        Delete all chunks for a file across all backends
        Returns number of chunks deleted (from chunk storage perspective)
        """
        # 1. Get chunk IDs first (needed for vector deletion)
        chunk_ids = self.metadata_store.list_by_file(file_path)
        
        # 2. Delete from vector store
        if self.vector_store and chunk_ids:
            self.vector_store.remove(chunk_ids)
        
        # 3. Delete from metadata
        self.metadata_store.delete_by_file(file_path)
        
        # 4. Delete from chunk storage
        count = self.chunk_storage.delete_by_file(file_path)
        
        logger.info(f"Deleted {count} chunks for {file_path}")
        return count
        
    def get_file_checksums(self, repository: str) -> Dict[str, str]:
        """Get checksums for all files in a repo for diff updates"""
        return self.metadata_store.get_file_checksums(repository)

    def save(self):
        """Save all storage backends"""
        logger.info(f"Saving storage to {self.config.base_path.absolute()}...")
        if self.vector_store:
            self.vector_store.save()
            
        if hasattr(self.chunk_storage, 'save'):
            self.chunk_storage.save()
            
        # Metadata store saves on write, but maybe close/commit
        if hasattr(self.metadata_store, 'save'):
            self.metadata_store.save()
            
    def close(self):
        """Close connections"""
        self.save()
        
        if hasattr(self.metadata_store, 'close'):
            self.metadata_store.close()
            
        if hasattr(self.chunk_storage, 'close'):
            self.chunk_storage.close()