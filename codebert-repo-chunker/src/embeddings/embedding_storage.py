"""
Embedding storage system for managing code embeddings at scale
Provides efficient storage, retrieval, versioning, and search capabilities
"""

import numpy as np
import faiss
import pickle
import json
import sqlite3
import h5py
import zarr
import lmdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import logging
import shutil
import threading
from collections import defaultdict
from contextlib import contextmanager
import mmap
import struct
import zlib
import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class StorageBackend(Enum):
    """Available storage backends"""
    NUMPY = "numpy"
    HDF5 = "hdf5"
    ZARR = "zarr"
    LMDB = "lmdb"
    SQLITE = "sqlite"
    PARQUET = "parquet"
    FAISS = "faiss"
    MEMORY = "memory"
    HYBRID = "hybrid"

class CompressionType(Enum):
    """Compression types for storage"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"

@dataclass
class EmbeddingMetadata:
    """Metadata for stored embeddings"""
    id: str
    chunk_id: str
    file_path: str
    model_name: str
    model_version: str
    embedding_dim: int
    created_at: datetime
    updated_at: datetime
    version: int = 1
    language: Optional[str] = None
    framework: Optional[str] = None
    chunk_type: Optional[str] = None
    chunk_size: Optional[int] = None
    hash: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StorageConfig:
    """Configuration for embedding storage"""
    backend: StorageBackend = StorageBackend.HDF5
    storage_path: Path = Path(".embeddings")
    compression: CompressionType = CompressionType.GZIP
    chunk_size: int = 1000  # Number of embeddings per chunk
    cache_size: int = 1000  # Number of embeddings to cache in memory
    index_type: str = "IVF1024,PQ64"  # FAISS index type
    enable_versioning: bool = True
    enable_sharding: bool = True
    shard_size: int = 100000  # Embeddings per shard
    enable_wal: bool = True  # Write-ahead logging for SQLite
    sync_interval: int = 100  # Sync to disk every N operations
    metadata_backend: str = "sqlite"  # sqlite or json
    
    def __post_init__(self):
        """Create storage directory"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

class BaseStorage(ABC):
    """Abstract base class for embedding storage"""
    
    @abstractmethod
    def store(self, embedding_id: str, embedding: np.ndarray, metadata: EmbeddingMetadata):
        """Store an embedding"""
        pass
    
    @abstractmethod
    def retrieve(self, embedding_id: str) -> Tuple[Optional[np.ndarray], Optional[EmbeddingMetadata]]:
        """Retrieve an embedding"""
        pass
    
    @abstractmethod
    def delete(self, embedding_id: str) -> bool:
        """Delete an embedding"""
        pass
    
    @abstractmethod
    def exists(self, embedding_id: str) -> bool:
        """Check if embedding exists"""
        pass
    
    @abstractmethod
    def list_ids(self) -> List[str]:
        """List all embedding IDs"""
        pass
    
    @abstractmethod
    def close(self):
        """Close storage connection"""
        pass

class MemoryStorage(BaseStorage):
    """In-memory embedding storage"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, EmbeddingMetadata] = {}
        self._lock = threading.RLock()
    
    def store(self, embedding_id: str, embedding: np.ndarray, metadata: EmbeddingMetadata):
        with self._lock:
            self.embeddings[embedding_id] = embedding.copy()
            self.metadata[embedding_id] = metadata
    
    def retrieve(self, embedding_id: str) -> Tuple[Optional[np.ndarray], Optional[EmbeddingMetadata]]:
        with self._lock:
            embedding = self.embeddings.get(embedding_id)
            metadata = self.metadata.get(embedding_id)
            return (embedding.copy() if embedding is not None else None, metadata)
    
    def delete(self, embedding_id: str) -> bool:
        with self._lock:
            if embedding_id in self.embeddings:
                del self.embeddings[embedding_id]
                del self.metadata[embedding_id]
                return True
            return False
    
    def exists(self, embedding_id: str) -> bool:
        return embedding_id in self.embeddings
    
    def list_ids(self) -> List[str]:
        return list(self.embeddings.keys())
    
    def close(self):
        self.embeddings.clear()
        self.metadata.clear()

class HDF5Storage(BaseStorage):
    """HDF5-based embedding storage"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.file_path = config.storage_path / "embeddings.h5"
        self._lock = threading.RLock()
        self._init_storage()
    
    def _init_storage(self):
        """Initialize HDF5 file"""
        with h5py.File(self.file_path, 'a') as f:
            if 'embeddings' not in f:
                f.create_group('embeddings')
            if 'metadata' not in f:
                f.create_group('metadata')
    
    def store(self, embedding_id: str, embedding: np.ndarray, metadata: EmbeddingMetadata):
        with self._lock:
            with h5py.File(self.file_path, 'a') as f:
                # Store embedding
                if embedding_id in f['embeddings']:
                    del f['embeddings'][embedding_id]
                
                f['embeddings'].create_dataset(
                    embedding_id,
                    data=embedding,
                    compression=self.config.compression.value if self.config.compression != CompressionType.NONE else None
                )
                
                # Store metadata
                metadata_json = json.dumps(asdict(metadata), default=str)
                if embedding_id in f['metadata']:
                    del f['metadata'][embedding_id]
                f['metadata'].create_dataset(embedding_id, data=metadata_json)
    
    def retrieve(self, embedding_id: str) -> Tuple[Optional[np.ndarray], Optional[EmbeddingMetadata]]:
        with self._lock:
            try:
                with h5py.File(self.file_path, 'r') as f:
                    if embedding_id not in f['embeddings']:
                        return None, None
                    
                    embedding = f['embeddings'][embedding_id][:]
                    metadata_json = f['metadata'][embedding_id][()].decode('utf-8')
                    metadata_dict = json.loads(metadata_json)
                    
                    # Convert datetime strings
                    metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                    metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                    
                    metadata = EmbeddingMetadata(**metadata_dict)
                    return embedding, metadata
            except Exception as e:
                logger.error(f"Error retrieving embedding {embedding_id}: {e}")
                return None, None
    
    def delete(self, embedding_id: str) -> bool:
        with self._lock:
            try:
                with h5py.File(self.file_path, 'a') as f:
                    if embedding_id in f['embeddings']:
                        del f['embeddings'][embedding_id]
                        del f['metadata'][embedding_id]
                        return True
                return False
            except Exception as e:
                logger.error(f"Error deleting embedding {embedding_id}: {e}")
                return False
    
    def exists(self, embedding_id: str) -> bool:
        with h5py.File(self.file_path, 'r') as f:
            return embedding_id in f['embeddings']
    
    def list_ids(self) -> List[str]:
        with h5py.File(self.file_path, 'r') as f:
            return list(f['embeddings'].keys())
    
    def close(self):
        pass  # HDF5 files are opened/closed per operation

class LMDBStorage(BaseStorage):
    """LMDB-based embedding storage for high performance"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.db_path = config.storage_path / "embeddings.lmdb"
        self.env = lmdb.open(
            str(self.db_path),
            map_size=10 * 1024 * 1024 * 1024,  # 10GB
            max_dbs=2,
            sync=False,
            writemap=True
        )
        
        # Create sub-databases
        self.embedding_db = self.env.open_db(b'embeddings')
        self.metadata_db = self.env.open_db(b'metadata')
    
    def store(self, embedding_id: str, embedding: np.ndarray, metadata: EmbeddingMetadata):
        key = embedding_id.encode('utf-8')
        
        # Serialize embedding
        embedding_bytes = embedding.tobytes()
        embedding_shape = struct.pack('II', *embedding.shape) if embedding.ndim == 2 else struct.pack('I', embedding.shape[0])
        embedding_data = embedding_shape + embedding_bytes
        
        # Serialize metadata
        metadata_bytes = msgpack.packb(asdict(metadata), default=str)
        
        with self.env.begin(write=True) as txn:
            txn.put(key, embedding_data, db=self.embedding_db)
            txn.put(key, metadata_bytes, db=self.metadata_db)
    
    def retrieve(self, embedding_id: str) -> Tuple[Optional[np.ndarray], Optional[EmbeddingMetadata]]:
        key = embedding_id.encode('utf-8')
        
        with self.env.begin() as txn:
            # Retrieve embedding
            embedding_data = txn.get(key, db=self.embedding_db)
            if not embedding_data:
                return None, None
            
            # Deserialize embedding
            if len(embedding_data) > 8:  # 2D array
                shape = struct.unpack('II', embedding_data[:8])
                embedding_bytes = embedding_data[8:]
            else:  # 1D array
                shape = (struct.unpack('I', embedding_data[:4])[0],)
                embedding_bytes = embedding_data[4:]
            
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(shape)
            
            # Retrieve and deserialize metadata
            metadata_bytes = txn.get(key, db=self.metadata_db)
            metadata_dict = msgpack.unpackb(metadata_bytes, raw=False)
            
            # Convert datetime strings
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
            
            metadata = EmbeddingMetadata(**metadata_dict)
            
            return embedding, metadata
    
    def delete(self, embedding_id: str) -> bool:
        key = embedding_id.encode('utf-8')
        
        with self.env.begin(write=True) as txn:
            success1 = txn.delete(key, db=self.embedding_db)
            success2 = txn.delete(key, db=self.metadata_db)
            return success1 and success2
    
    def exists(self, embedding_id: str) -> bool:
        key = embedding_id.encode('utf-8')
        with self.env.begin() as txn:
            return txn.get(key, db=self.embedding_db) is not None
    
    def list_ids(self) -> List[str]:
        ids = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.embedding_db)
            for key, _ in cursor:
                ids.append(key.decode('utf-8'))
        return ids
    
    def close(self):
        self.env.close()

class SQLiteMetadataStore:
    """SQLite-based metadata storage"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT,
                    file_path TEXT,
                    model_name TEXT,
                    model_version TEXT,
                    embedding_dim INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    version INTEGER,
                    language TEXT,
                    framework TEXT,
                    chunk_type TEXT,
                    chunk_size INTEGER,
                    hash TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON embeddings(chunk_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON embeddings(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)")
            
            self.conn.commit()
    
    def store(self, metadata: EmbeddingMetadata):
        """Store metadata"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                metadata.id,
                metadata.chunk_id,
                metadata.file_path,
                metadata.model_name,
                metadata.model_version,
                metadata.embedding_dim,
                metadata.created_at.isoformat(),
                metadata.updated_at.isoformat(),
                metadata.version,
                metadata.language,
                metadata.framework,
                metadata.chunk_type,
                metadata.chunk_size,
                metadata.hash,
                json.dumps(metadata.tags),
                # Enforce no content in shadow writer
                json.dumps({k: v for k, v in metadata.metadata.items() if k not in ['content', 'embedding']})
            ))
            self.conn.commit()
    
    def retrieve(self, embedding_id: str) -> Optional[EmbeddingMetadata]:
        """Retrieve metadata"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM embeddings WHERE id = ?", (embedding_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return EmbeddingMetadata(
                id=row[0],
                chunk_id=row[1],
                file_path=row[2],
                model_name=row[3],
                model_version=row[4],
                embedding_dim=row[5],
                created_at=datetime.fromisoformat(row[6]),
                updated_at=datetime.fromisoformat(row[7]),
                version=row[8],
                language=row[9],
                framework=row[10],
                chunk_type=row[11],
                chunk_size=row[12],
                hash=row[13],
                tags=json.loads(row[14]) if row[14] else [],
                metadata=json.loads(row[15]) if row[15] else {}
            )
    
    def delete(self, embedding_id: str) -> bool:
        """Delete metadata"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM embeddings WHERE id = ?", (embedding_id,))
            self.conn.commit()
            return cursor.rowcount > 0
    
    def search(self, **kwargs) -> List[EmbeddingMetadata]:
        """Search metadata by criteria"""
        with self._lock:
            query = "SELECT * FROM embeddings WHERE 1=1"
            params = []
            
            if 'file_path' in kwargs:
                query += " AND file_path = ?"
                params.append(kwargs['file_path'])
            
            if 'model_name' in kwargs:
                query += " AND model_name = ?"
                params.append(kwargs['model_name'])
            
            if 'language' in kwargs:
                query += " AND language = ?"
                params.append(kwargs['language'])
            
            if 'chunk_type' in kwargs:
                query += " AND chunk_type = ?"
                params.append(kwargs['chunk_type'])
            
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append(EmbeddingMetadata(
                    id=row[0],
                    chunk_id=row[1],
                    file_path=row[2],
                    model_name=row[3],
                    model_version=row[4],
                    embedding_dim=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    version=row[8],
                    language=row[9],
                    framework=row[10],
                    chunk_type=row[11],
                    chunk_size=row[12],
                    hash=row[13],
                    tags=json.loads(row[14]) if row[14] else [],
                    metadata=json.loads(row[15]) if row[15] else {}
                ))
            
            return results
    
    def close(self):
        """Close database connection"""
        self.conn.close()

class EmbeddingStorage:
    """
    Main embedding storage system with caching and indexing
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize embedding storage
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        
        # Initialize storage backend
        self._init_backend()
        
        # Initialize metadata store
        self.metadata_store = SQLiteMetadataStore(
            self.config.storage_path / "metadata.db"
        )
        
        # Initialize cache
        self.cache = MemoryStorage(self.config)
        self.cache_order = []
        
        # Initialize FAISS index
        self.index = None
        self.index_to_id = {}
        self.id_to_index = {}
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Sync counter
        self.operations_since_sync = 0
    
    def _init_backend(self):
        """Initialize storage backend"""
        if self.config.backend == StorageBackend.MEMORY:
            self.backend = MemoryStorage(self.config)
        elif self.config.backend == StorageBackend.HDF5:
            self.backend = HDF5Storage(self.config)
        elif self.config.backend == StorageBackend.LMDB:
            self.backend = LMDBStorage(self.config)
        else:
            # Default to HDF5
            self.backend = HDF5Storage(self.config)
    
    def store(self, 
             chunk_id: str,
             embedding: np.ndarray,
             file_path: str,
             model_name: str,
             model_version: str = "1.0.0",
             **kwargs) -> str:
        """
        Store an embedding
        
        Args:
            chunk_id: Chunk identifier
            embedding: Embedding vector
            file_path: Source file path
            model_name: Model used for embedding
            model_version: Model version
            **kwargs: Additional metadata
            
        Returns:
            Embedding ID
        """
        # Generate embedding ID
        embedding_id = self._generate_id(chunk_id, file_path, model_name)
        
        # Create metadata
        metadata = EmbeddingMetadata(
            id=embedding_id,
            chunk_id=chunk_id,
            file_path=file_path,
            model_name=model_name,
            model_version=model_version,
            embedding_dim=embedding.shape[-1],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            **kwargs
        )
        
        # Check if exists and handle versioning
        if self.config.enable_versioning and self.backend.exists(embedding_id):
            existing_embedding, existing_metadata = self.backend.retrieve(embedding_id)
            if existing_metadata:
                metadata.version = existing_metadata.version + 1
        
        # Store in backend
        self.backend.store(embedding_id, embedding, metadata)
        
        # Store metadata
        self.metadata_store.store(metadata)
        
        # Update cache
        self._update_cache(embedding_id, embedding, metadata)
        
        # Update index
        if self.index is not None:
            self._add_to_index(embedding_id, embedding)
        
        # Update statistics
        self.stats['total_stored'] += 1
        self.operations_since_sync += 1
        
        # Sync if needed
        if self.operations_since_sync >= self.config.sync_interval:
            self.sync()
        
        return embedding_id
    
    def retrieve(self, embedding_id: str) -> Tuple[Optional[np.ndarray], Optional[EmbeddingMetadata]]:
        """
        Retrieve an embedding
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Embedding and metadata
        """
        # Check cache first
        cached = self.cache.retrieve(embedding_id)
        if cached[0] is not None:
            self.stats['cache_hits'] += 1
            return cached
        
        self.stats['cache_misses'] += 1
        
        # Retrieve from backend
        embedding, metadata = self.backend.retrieve(embedding_id)
        
        if embedding is not None:
            # Update cache
            self._update_cache(embedding_id, embedding, metadata)
        
        return embedding, metadata
    
    def batch_store(self, 
                   embeddings: List[Tuple[str, np.ndarray, Dict[str, Any]]],
                   show_progress: bool = False) -> List[str]:
        """
        Store multiple embeddings
        
        Args:
            embeddings: List of (chunk_id, embedding, metadata) tuples
            show_progress: Show progress bar
            
        Returns:
            List of embedding IDs
        """
        from tqdm import tqdm
        
        ids = []
        
        if show_progress:
            embeddings = tqdm(embeddings, desc="Storing embeddings")
        
        for chunk_id, embedding, metadata in embeddings:
            embedding_id = self.store(
                chunk_id=chunk_id,
                embedding=embedding,
                **metadata
            )
            ids.append(embedding_id)
        
        return ids
    
    def batch_retrieve(self, embedding_ids: List[str]) -> List[Tuple[Optional[np.ndarray], Optional[EmbeddingMetadata]]]:
        """Retrieve multiple embeddings"""
        results = []
        for embedding_id in embedding_ids:
            results.append(self.retrieve(embedding_id))
        return results
    
    def delete(self, embedding_id: str) -> bool:
        """Delete an embedding"""
        # Delete from backend
        success = self.backend.delete(embedding_id)
        
        if success:
            # Delete from metadata store
            self.metadata_store.delete(embedding_id)
            
            # Delete from cache
            self.cache.delete(embedding_id)
            
            # Remove from index
            if self.index is not None and embedding_id in self.id_to_index:
                # Note: FAISS doesn't support deletion, need to rebuild
                self._rebuild_index_excluding(embedding_id)
            
            self.stats['total_deleted'] += 1
        
        return success
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, EmbeddingMetadata]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of (embedding_id, distance, metadata) tuples
        """
        if self.index is None:
            self.build_index()
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Search in index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(k * 3, self.index.ntotal)  # Get more results for filtering
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:  # Invalid index
                continue
            
            embedding_id = self.index_to_id.get(idx)
            if not embedding_id:
                continue
            
            # Get metadata
            metadata = self.metadata_store.retrieve(embedding_id)
            if not metadata:
                continue
            
            # Apply filters
            if filters:
                if not self._match_filters(metadata, filters):
                    continue
            
            results.append((embedding_id, float(distance), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def build_index(self, index_type: Optional[str] = None):
        """Build FAISS index for all embeddings"""
        index_type = index_type or self.config.index_type
        
        # Get all embeddings
        all_ids = self.backend.list_ids()
        if not all_ids:
            logger.warning("No embeddings to index")
            return
        
        # Get first embedding to determine dimension
        first_embedding, _ = self.backend.retrieve(all_ids[0])
        if first_embedding is None:
            return
        
        dimension = first_embedding.shape[-1]
        
        # Create FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            # Use factory to create complex index
            self.index = faiss.index_factory(dimension, index_type)
        
        # Add embeddings to index
        embeddings = []
        valid_ids = []
        
        for i, embedding_id in enumerate(all_ids):
            embedding, _ = self.backend.retrieve(embedding_id)
            if embedding is not None:
                embeddings.append(embedding)
                valid_ids.append(embedding_id)
                self.index_to_id[len(embeddings) - 1] = embedding_id
                self.id_to_index[embedding_id] = len(embeddings) - 1
        
        if embeddings:
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            
            # Train index if needed
            if hasattr(self.index, 'train'):
                self.index.train(embeddings_array)
            
            # Add to index
            self.index.add(embeddings_array)
            
            logger.info(f"Built index with {self.index.ntotal} embeddings")
    
    def _update_cache(self, embedding_id: str, embedding: np.ndarray, metadata: EmbeddingMetadata):
        """Update cache with LRU eviction"""
        # Add to cache
        self.cache.store(embedding_id, embedding, metadata)
        
        # Track order
        if embedding_id in self.cache_order:
            self.cache_order.remove(embedding_id)
        self.cache_order.append(embedding_id)
        
        # Evict if cache is full
        while len(self.cache_order) > self.config.cache_size:
            oldest_id = self.cache_order.pop(0)
            self.cache.delete(oldest_id)
    
    def _add_to_index(self, embedding_id: str, embedding: np.ndarray):
        """Add embedding to index"""
        if self.index is None:
            return
        
        idx = self.index.ntotal
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        self.index_to_id[idx] = embedding_id
        self.id_to_index[embedding_id] = idx
    
    def _rebuild_index_excluding(self, exclude_id: str):
        """Rebuild index excluding specific embedding"""
        # This is inefficient but necessary for FAISS
        logger.info(f"Rebuilding index excluding {exclude_id}")
        
        # Clear current index
        self.index = None
        self.index_to_id = {}
        self.id_to_index = {}
        
        # Rebuild
        self.build_index()
    
    def _match_filters(self, metadata: EmbeddingMetadata, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if hasattr(metadata, key):
                if getattr(metadata, key) != value:
                    return False
            elif key in metadata.metadata:
                if metadata.metadata[key] != value:
                    return False
            else:
                return False
        return True
    
    def _generate_id(self, chunk_id: str, file_path: str, model_name: str) -> str:
        """Generate unique embedding ID"""
        data = f"{chunk_id}_{file_path}_{model_name}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def sync(self):
        """Sync storage to disk"""
        self.operations_since_sync = 0
        # Most backends handle syncing internally
        logger.debug("Storage synced")
    
    def export(self, output_path: Path, format: str = "parquet"):
        """Export embeddings to file"""
        all_ids = self.backend.list_ids()
        
        if format == "parquet":
            # Export as Parquet
            data = []
            for embedding_id in all_ids:
                embedding, metadata = self.backend.retrieve(embedding_id)
                if embedding is not None:
                    data.append({
                        'id': embedding_id,
                        'embedding': embedding.tolist(),
                        'chunk_id': metadata.chunk_id,
                        'file_path': metadata.file_path,
                        'model_name': metadata.model_name,
                        'created_at': metadata.created_at.isoformat()
                    })
            
            df = pa.Table.from_pylist(data)
            pq.write_table(df, output_path)
        
        elif format == "numpy":
            # Export as NumPy arrays
            embeddings = []
            metadata_list = []
            
            for embedding_id in all_ids:
                embedding, metadata = self.backend.retrieve(embedding_id)
                if embedding is not None:
                    embeddings.append(embedding)
                    metadata_list.append(asdict(metadata))
            
            np.savez(
                output_path,
                embeddings=np.vstack(embeddings),
                metadata=metadata_list
            )
    
    def import_embeddings(self, input_path: Path, format: str = "parquet"):
        """Import embeddings from file"""
        if format == "parquet":
            table = pq.read_table(input_path)
            
            for row in table.to_pylist():
                self.store(
                    chunk_id=row['chunk_id'],
                    embedding=np.array(row['embedding']),
                    file_path=row['file_path'],
                    model_name=row['model_name']
                )
        
        elif format == "numpy":
            data = np.load(input_path, allow_pickle=True)
            embeddings = data['embeddings']
            metadata_list = data['metadata']
            
            for embedding, metadata in zip(embeddings, metadata_list):
                self.store(
                    chunk_id=metadata['chunk_id'],
                    embedding=embedding,
                    file_path=metadata['file_path'],
                    model_name=metadata['model_name']
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_embeddings = len(self.backend.list_ids())
        
        return {
            'total_embeddings': total_embeddings,
            'total_stored': self.stats['total_stored'],
            'total_deleted': self.stats['total_deleted'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1),
            'cache_size': len(self.cache.list_ids()),
            'index_size': self.index.ntotal if self.index else 0,
            'backend': self.config.backend.value,
            'compression': self.config.compression.value,
            'storage_path': str(self.config.storage_path)
        }
    
    def close(self):
        """Close storage connections"""
        self.sync()
        self.backend.close()
        self.metadata_store.close()
        self.cache.close()

# Convenience functions
def create_storage(backend: str = "hdf5", 
                  storage_path: str = ".embeddings") -> EmbeddingStorage:
    """Create embedding storage with common settings"""
    config = StorageConfig(
        backend=StorageBackend(backend),
        storage_path=Path(storage_path)
    )
    return EmbeddingStorage(config)

def store_embedding(chunk_id: str, 
                   embedding: np.ndarray,
                   file_path: str,
                   model_name: str = "microsoft/codebert-base") -> str:
    """Quick function to store an embedding"""
    storage = create_storage()
    embedding_id = storage.store(chunk_id, embedding, file_path, model_name)
    storage.close()
    return embedding_id