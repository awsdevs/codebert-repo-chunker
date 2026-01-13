"""
Chunk storage system for efficient storage and retrieval of code chunks
Provides multiple backend options with compression, versioning, and caching
"""

import json
import pickle
import hashlib
import zlib
import lz4.frame
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import logging
import sqlite3
import threading
from contextlib import contextmanager
import shutil
import tempfile
import mmap
import struct
from collections import defaultdict, OrderedDict
import numpy as np

# Storage backends
import h5py
import zarr
import lmdb
import rocksdb
import leveldb
from pymongo import MongoClient
from redis import Redis
import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Internal imports
from src.core.chunk_model import Chunk, ChunkType, ChunkMetadata
from src.utils.serialization import ChunkSerializer

logger = logging.getLogger(__name__)

class StorageBackend(Enum):
    """Available storage backends"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
    ROCKSDB = "rocksdb"
    LEVELDB = "leveldb"
    LMDB = "lmdb"
    HDF5 = "hdf5"
    ZARR = "zarr"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"

class CompressionType(Enum):
    """Compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"
    BROTLI = "brotli"

class SerializationFormat(Enum):
    """Serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    PARQUET = "parquet"

@dataclass
class StorageConfig:
    """Configuration for chunk storage"""
    backend: StorageBackend = StorageBackend.SQLITE
    storage_path: Path = Path("storage/chunks")
    
    # Connection settings
    connection_string: Optional[str] = None
    host: str = "localhost"
    port: Optional[int] = None
    database: str = "chunks"
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Storage settings
    compression: CompressionType = CompressionType.LZ4
    compression_level: int = 3
    serialization: SerializationFormat = SerializationFormat.JSON
    
    # Performance settings
    cache_enabled: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    batch_size: int = 1000
    connection_pool_size: int = 10
    
    # Versioning
    enable_versioning: bool = True
    max_versions: int = 10
    
    # Sharding
    enable_sharding: bool = False
    shard_count: int = 16
    shard_by: str = "chunk_id"  # chunk_id, file_path, date
    
    # Reliability
    enable_wal: bool = True  # Write-ahead logging
    sync_interval: int = 100
    backup_enabled: bool = False
    backup_path: Optional[Path] = None
    
    # Limits
    max_chunk_size_mb: float = 10.0
    max_storage_size_gb: Optional[float] = None
    
    def __post_init__(self):
        """Initialize paths"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if self.backup_enabled and self.backup_path:
            self.backup_path.mkdir(parents=True, exist_ok=True)

@dataclass
class ChunkVersion:
    """Version information for a chunk"""
    version: int
    chunk_id: str
    content_hash: str
    created_at: datetime
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StorageStats:
    """Storage statistics"""
    total_chunks: int = 0
    total_size_bytes: int = 0
    total_versions: int = 0
    compression_ratio: float = 1.0
    cache_hits: int = 0
    cache_misses: int = 0
    read_operations: int = 0
    write_operations: int = 0
    errors: List[str] = field(default_factory=list)

class ChunkCache:
    """LRU cache for chunks"""
    
    def __init__(self, max_size_mb: int = 512):
        """Initialize cache with size limit"""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.size_bytes = 0
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Chunk]:
        """Get chunk from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, chunk: Chunk):
        """Add chunk to cache"""
        with self.lock:
            # Estimate chunk size
            chunk_size = len(json.dumps(chunk.to_dict(), default=str).encode())
            
            # Remove items if cache is full
            while self.size_bytes + chunk_size > self.max_size_bytes and self.cache:
                oldest_key = next(iter(self.cache))
                removed = self.cache.pop(oldest_key)
                removed_size = len(json.dumps(removed.to_dict(), default=str).encode())
                self.size_bytes -= removed_size
            
            # Add new item
            self.cache[key] = chunk
            self.size_bytes += chunk_size
            
            # Move to end if already exists
            if key in self.cache:
                self.cache.move_to_end(key)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            return {
                'size_bytes': self.size_bytes,
                'size_mb': self.size_bytes / (1024 * 1024),
                'items': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hits / total if total > 0 else 0
            }

class BaseChunkStorage:
    """Base class for chunk storage implementations"""
    
    def __init__(self, config: StorageConfig):
        """Initialize storage"""
        self.config = config
        self.serializer = ChunkSerializer(format=config.serialization)
        self.stats = StorageStats()
        self.lock = threading.RLock()
        
        # Initialize cache
        if config.cache_enabled:
            self.cache = ChunkCache(config.cache_size_mb)
        else:
            self.cache = None
    
    def store(self, chunk: Chunk, version: bool = True) -> bool:
        """Store a chunk"""
        raise NotImplementedError
    
    def retrieve(self, chunk_id: str, version: Optional[int] = None) -> Optional[Chunk]:
        """Retrieve a chunk"""
        raise NotImplementedError
    
    def delete(self, chunk_id: str, version: Optional[int] = None) -> bool:
        """Delete a chunk"""
        raise NotImplementedError
    
    def exists(self, chunk_id: str) -> bool:
        """Check if chunk exists"""
        raise NotImplementedError
    
    def list_chunks(self, limit: Optional[int] = None, offset: int = 0) -> List[str]:
        """List chunk IDs"""
        raise NotImplementedError
    
    def get_versions(self, chunk_id: str) -> List[ChunkVersion]:
        """Get version history"""
        raise NotImplementedError
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data"""
        if self.config.compression == CompressionType.NONE:
            return data
        elif self.config.compression == CompressionType.GZIP:
            return zlib.compress(data, level=self.config.compression_level)
        elif self.config.compression == CompressionType.LZ4:
            return lz4.frame.compress(data, compression_level=self.config.compression_level)
        elif self.config.compression == CompressionType.ZSTD:
            cctx = zstd.ZstdCompressor(level=self.config.compression_level)
            return cctx.compress(data)
        else:
            return data
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        if self.config.compression == CompressionType.NONE:
            return data
        elif self.config.compression == CompressionType.GZIP:
            return zlib.decompress(data)
        elif self.config.compression == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        elif self.config.compression == CompressionType.ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        else:
            return data

class SQLiteChunkStorage(BaseChunkStorage):
    """SQLite-based chunk storage"""
    
    def __init__(self, config: StorageConfig):
        """Initialize SQLite storage"""
        super().__init__(config)
        
        self.db_path = config.storage_path / "chunks.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        
        # Enable WAL mode for better concurrency
        if config.enable_wal:
            self.conn.execute("PRAGMA journal_mode=WAL")
        
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.lock:
            cursor = self.conn.cursor()
            
            # Main chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content BLOB NOT NULL,
                    chunk_type TEXT,
                    file_path TEXT,
                    language TEXT,
                    size_bytes INTEGER,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Versions table
            if self.config.enable_versioning:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunk_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chunk_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        content BLOB NOT NULL,
                        content_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        size_bytes INTEGER,
                        metadata TEXT,
                        UNIQUE(chunk_id, version),
                        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
                    )
                """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_language ON chunks(language)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON chunks(created_at)")
            
            self.conn.commit()
    
    def store(self, chunk: Chunk, version: bool = True) -> bool:
        """Store a chunk in SQLite"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                # Serialize chunk
                chunk_dict = chunk.to_dict()
                content = self.serializer.serialize(chunk_dict)
                
                # Compress
                compressed = self._compress(content)
                
                # Calculate hash
                content_hash = hashlib.sha256(content).hexdigest()
                
                # Check if exists
                cursor.execute("SELECT chunk_id, content_hash FROM chunks WHERE chunk_id = ?", (chunk.id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing chunk
                    if version and self.config.enable_versioning and existing[1] != content_hash:
                        # Save current version
                        cursor.execute("""
                            INSERT INTO chunk_versions (chunk_id, version, content, content_hash, size_bytes, metadata)
                            SELECT chunk_id, 
                                   COALESCE((SELECT MAX(version) FROM chunk_versions WHERE chunk_id = ?), 0) + 1,
                                   content, content_hash, size_bytes, metadata
                            FROM chunks WHERE chunk_id = ?
                        """, (chunk.id, chunk.id))
                    
                    # Update chunk
                    cursor.execute("""
                        UPDATE chunks SET 
                            content = ?, chunk_type = ?, file_path = ?, language = ?,
                            size_bytes = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP,
                            metadata = ?
                        WHERE chunk_id = ?
                    """, (
                        compressed,
                        chunk.chunk_type.value,
                        chunk.file_path,
                        chunk.metadata.language,
                        len(compressed),
                        content_hash,
                        json.dumps(chunk.metadata.annotations),
                        chunk.id
                    ))
                else:
                    # Insert new chunk
                    cursor.execute("""
                        INSERT INTO chunks (
                            chunk_id, content, chunk_type, file_path, language,
                            size_bytes, content_hash, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.id,
                        compressed,
                        chunk.chunk_type.value,
                        chunk.file_path,
                        chunk.metadata.language,
                        len(compressed),
                        content_hash,
                        json.dumps(chunk.metadata.annotations)
                    ))
                
                self.conn.commit()
                
                # Update cache
                if self.cache:
                    self.cache.put(chunk.id, chunk)
                
                # Update stats
                self.stats.write_operations += 1
                self.stats.total_chunks = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk.id}: {e}")
            self.stats.errors.append(str(e))
            return False
    
    def retrieve(self, chunk_id: str, version: Optional[int] = None) -> Optional[Chunk]:
        """Retrieve a chunk from SQLite"""
        try:
            # Check cache first
            if self.cache:
                cached = self.cache.get(chunk_id)
                if cached:
                    self.stats.cache_hits += 1
                    return cached
                self.stats.cache_misses += 1
            
            with self.lock:
                cursor = self.conn.cursor()
                
                if version is not None and self.config.enable_versioning:
                    # Retrieve specific version
                    cursor.execute("""
                        SELECT content FROM chunk_versions 
                        WHERE chunk_id = ? AND version = ?
                    """, (chunk_id, version))
                else:
                    # Retrieve current version
                    cursor.execute("SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,))
                
                row = cursor.fetchone()
                
                if row:
                    # Decompress
                    decompressed = self._decompress(row[0])
                    
                    # Deserialize
                    chunk_dict = self.serializer.deserialize(decompressed)
                    
                    # Create chunk object
                    chunk = Chunk.from_dict(chunk_dict)
                    
                    # Update cache
                    if self.cache:
                        self.cache.put(chunk_id, chunk)
                    
                    # Update stats
                    self.stats.read_operations += 1
                    
                    return chunk
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {e}")
            self.stats.errors.append(str(e))
            return None
    
    def delete(self, chunk_id: str, version: Optional[int] = None) -> bool:
        """Delete a chunk from SQLite"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                if version is not None and self.config.enable_versioning:
                    # Delete specific version
                    cursor.execute("""
                        DELETE FROM chunk_versions 
                        WHERE chunk_id = ? AND version = ?
                    """, (chunk_id, version))
                else:
                    # Delete chunk and all versions
                    if self.config.enable_versioning:
                        cursor.execute("DELETE FROM chunk_versions WHERE chunk_id = ?", (chunk_id,))
                    cursor.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
                
                self.conn.commit()
                
                # Remove from cache
                if self.cache and chunk_id in self.cache.cache:
                    del self.cache.cache[chunk_id]
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            self.stats.errors.append(str(e))
            return False
    
    def exists(self, chunk_id: str) -> bool:
        """Check if chunk exists"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM chunks WHERE chunk_id = ? LIMIT 1", (chunk_id,))
            return cursor.fetchone() is not None
    
    def list_chunks(self, limit: Optional[int] = None, offset: int = 0) -> List[str]:
        """List chunk IDs"""
        with self.lock:
            cursor = self.conn.cursor()
            
            query = "SELECT chunk_id FROM chunks ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall()]
    
    def get_versions(self, chunk_id: str) -> List[ChunkVersion]:
        """Get version history"""
        if not self.config.enable_versioning:
            return []
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT version, content_hash, created_at, size_bytes, metadata
                FROM chunk_versions
                WHERE chunk_id = ?
                ORDER BY version DESC
            """, (chunk_id,))
            
            versions = []
            for row in cursor.fetchall():
                versions.append(ChunkVersion(
                    version=row[0],
                    chunk_id=chunk_id,
                    content_hash=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    size_bytes=row[3],
                    metadata=json.loads(row[4]) if row[4] else {}
                ))
            
            return versions
    
    def search(self, **criteria) -> List[Chunk]:
        """Search chunks by criteria"""
        with self.lock:
            cursor = self.conn.cursor()
            
            query = "SELECT content FROM chunks WHERE 1=1"
            params = []
            
            if 'file_path' in criteria:
                query += " AND file_path = ?"
                params.append(criteria['file_path'])
            
            if 'language' in criteria:
                query += " AND language = ?"
                params.append(criteria['language'])
            
            if 'chunk_type' in criteria:
                query += " AND chunk_type = ?"
                params.append(criteria['chunk_type'])
            
            if 'after_date' in criteria:
                query += " AND created_at >= ?"
                params.append(criteria['after_date'])
            
            cursor.execute(query, params)
            
            chunks = []
            for row in cursor.fetchall():
                decompressed = self._decompress(row[0])
                chunk_dict = self.serializer.deserialize(decompressed)
                chunks.append(Chunk.from_dict(chunk_dict))
            
            return chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self.lock:
            cursor = self.conn.cursor()
            
            stats = {
                'total_chunks': cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                'total_size_bytes': cursor.execute("SELECT SUM(size_bytes) FROM chunks").fetchone()[0] or 0,
                'unique_files': cursor.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0],
                'languages': cursor.execute("SELECT DISTINCT language FROM chunks").fetchall(),
                'chunk_types': cursor.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type").fetchall()
            }
            
            if self.config.enable_versioning:
                stats['total_versions'] = cursor.execute("SELECT COUNT(*) FROM chunk_versions").fetchone()[0]
            
            if self.cache:
                stats['cache'] = self.cache.get_stats()
            
            stats['operations'] = {
                'reads': self.stats.read_operations,
                'writes': self.stats.write_operations,
                'errors': len(self.stats.errors)
            }
            
            return stats
    
    def optimize(self):
        """Optimize storage"""
        with self.lock:
            self.conn.execute("VACUUM")
            self.conn.execute("ANALYZE")
    
    def backup(self, backup_path: Optional[Path] = None):
        """Create backup"""
        backup_path = backup_path or self.config.backup_path
        if not backup_path:
            return
        
        backup_file = backup_path / f"chunks_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
        
        with self.lock:
            # Use SQLite backup API
            with sqlite3.connect(str(backup_file)) as backup_conn:
                self.conn.backup(backup_conn)
        
        logger.info(f"Backup created: {backup_file}")
    
    def close(self):
        """Close storage connection"""
        if self.cache:
            cache_stats = self.cache.get_stats()
            logger.info(f"Cache stats: {cache_stats}")
        
        self.conn.close()

class LMDBChunkStorage(BaseChunkStorage):
    """LMDB-based chunk storage for high performance"""
    
    def __init__(self, config: StorageConfig):
        """Initialize LMDB storage"""
        super().__init__(config)
        
        self.db_path = config.storage_path / "chunks.lmdb"
        self.env = lmdb.open(
            str(self.db_path),
            map_size=50 * 1024 * 1024 * 1024,  # 50GB
            max_dbs=10,
            sync=True,
            writemap=True
        )
        
        # Create sub-databases
        self.chunks_db = self.env.open_db(b'chunks')
        self.metadata_db = self.env.open_db(b'metadata')
        
        if config.enable_versioning:
            self.versions_db = self.env.open_db(b'versions')
    
    def store(self, chunk: Chunk, version: bool = True) -> bool:
        """Store chunk in LMDB"""
        try:
            # Serialize chunk
            chunk_dict = chunk.to_dict()
            content = self.serializer.serialize(chunk_dict)
            compressed = self._compress(content)
            
            # Create metadata
            metadata = {
                'chunk_type': chunk.chunk_type.value,
                'file_path': chunk.file_path,
                'language': chunk.metadata.language,
                'size_bytes': len(compressed),
                'content_hash': hashlib.sha256(content).hexdigest(),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            with self.env.begin(write=True) as txn:
                # Store chunk
                txn.put(
                    chunk.id.encode(),
                    compressed,
                    db=self.chunks_db
                )
                
                # Store metadata
                txn.put(
                    chunk.id.encode(),
                    json.dumps(metadata).encode(),
                    db=self.metadata_db
                )
                
                # Handle versioning
                if version and self.config.enable_versioning:
                    version_key = f"{chunk.id}:v{datetime.now():%Y%m%d%H%M%S}".encode()
                    txn.put(version_key, compressed, db=self.versions_db)
            
            # Update cache
            if self.cache:
                self.cache.put(chunk.id, chunk)
            
            self.stats.write_operations += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk.id}: {e}")
            self.stats.errors.append(str(e))
            return False
    
    def retrieve(self, chunk_id: str, version: Optional[int] = None) -> Optional[Chunk]:
        """Retrieve chunk from LMDB"""
        try:
            # Check cache
            if self.cache:
                cached = self.cache.get(chunk_id)
                if cached:
                    return cached
            
            with self.env.begin() as txn:
                compressed = txn.get(chunk_id.encode(), db=self.chunks_db)
                
                if compressed:
                    decompressed = self._decompress(compressed)
                    chunk_dict = self.serializer.deserialize(decompressed)
                    chunk = Chunk.from_dict(chunk_dict)
                    
                    # Update cache
                    if self.cache:
                        self.cache.put(chunk_id, chunk)
                    
                    self.stats.read_operations += 1
                    return chunk
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {e}")
            return None
    
    def delete(self, chunk_id: str, version: Optional[int] = None) -> bool:
        """Delete chunk from LMDB"""
        try:
            with self.env.begin(write=True) as txn:
                success = txn.delete(chunk_id.encode(), db=self.chunks_db)
                txn.delete(chunk_id.encode(), db=self.metadata_db)
                
                # Delete versions if enabled
                if self.config.enable_versioning:
                    cursor = txn.cursor(db=self.versions_db)
                    prefix = f"{chunk_id}:v".encode()
                    cursor.set_range(prefix)
                    
                    while cursor.key().startswith(prefix):
                        cursor.delete()
                        if not cursor.next():
                            break
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False
    
    def exists(self, chunk_id: str) -> bool:
        """Check if chunk exists in LMDB"""
        with self.env.begin() as txn:
            return txn.get(chunk_id.encode(), db=self.chunks_db) is not None
    
    def list_chunks(self, limit: Optional[int] = None, offset: int = 0) -> List[str]:
        """List chunk IDs from LMDB"""
        chunk_ids = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.chunks_db)
            
            # Skip to offset
            for _ in range(offset):
                if not cursor.next():
                    break
            
            # Collect chunk IDs
            count = 0
            for key, _ in cursor:
                chunk_ids.append(key.decode())
                count += 1
                
                if limit and count >= limit:
                    break
        
        return chunk_ids
    
    def close(self):
        """Close LMDB environment"""
        self.env.close()

class ChunkStorageFactory:
    """Factory for creating chunk storage instances"""
    
    @staticmethod
    def create(config: StorageConfig) -> BaseChunkStorage:
        """Create storage instance based on backend"""
        if config.backend == StorageBackend.SQLITE:
            return SQLiteChunkStorage(config)
        elif config.backend == StorageBackend.LMDB:
            return LMDBChunkStorage(config)
        # Add other backends as needed
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

# Convenience functions
def create_chunk_storage(backend: str = "sqlite", 
                        path: str = "storage/chunks",
                        **kwargs) -> BaseChunkStorage:
    """Create chunk storage with common settings"""
    config = StorageConfig(
        backend=StorageBackend(backend),
        storage_path=Path(path),
        **kwargs
    )
    return ChunkStorageFactory.create(config)