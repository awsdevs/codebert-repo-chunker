"""
Metadata storage system for managing chunk and file metadata
Provides rich querying, indexing, and relationship tracking capabilities
"""

import json
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import threading
from contextlib import contextmanager
import hashlib
from collections import defaultdict, Counter
import uuid

# Database backends
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from elasticsearch import Elasticsearch
from redis import Redis
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class MetadataBackend(Enum):
    """Available metadata storage backends"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"
    PARQUET = "parquet"

class IndexType(Enum):
    """Types of indices for metadata"""
    BTREE = "btree"
    HASH = "hash"
    FULLTEXT = "fulltext"
    SPATIAL = "spatial"
    BITMAP = "bitmap"

@dataclass
class MetadataConfig:
    """Configuration for metadata storage"""
    backend: MetadataBackend = MetadataBackend.SQLITE
    storage_path: Path = Path("storage/metadata")
    
    # Connection settings
    connection_string: Optional[str] = None
    host: str = "localhost"
    port: Optional[int] = None
    database: str = "metadata"
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Performance settings
    cache_enabled: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: int = 3600
    batch_size: int = 1000
    connection_pool_size: int = 10
    
    # Indexing
    auto_index: bool = True
    index_fields: List[str] = field(default_factory=lambda: [
        "chunk_id", "file_path", "language", "chunk_type",
        "created_at", "updated_at", "project", "repository"
    ])
    fulltext_fields: List[str] = field(default_factory=lambda: [
        "description", "tags", "comments"
    ])
    
    # Partitioning
    enable_partitioning: bool = False
    partition_by: str = "created_at"  # created_at, project, repository
    partition_interval: str = "monthly"  # daily, weekly, monthly, yearly
    
    # Maintenance
    enable_vacuum: bool = True
    vacuum_interval_hours: int = 24
    enable_archiving: bool = False
    archive_after_days: int = 90
    
    # Limits
    max_query_results: int = 10000
    query_timeout_seconds: int = 30
    
    def __post_init__(self):
        """Initialize storage path"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

@dataclass
class ChunkMetadata:
    """Comprehensive metadata for a chunk"""
    # Identity
    chunk_id: str
    chunk_hash: str
    version: int = 1
    
    # Source information
    file_path: str
    file_hash: Optional[str] = None
    repository: Optional[str] = None
    project: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    
    # Chunk properties
    chunk_type: str = "unknown"
    language: Optional[str] = None
    framework: Optional[str] = None
    
    # Location
    start_line: int = 0
    end_line: int = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Size metrics
    size_bytes: int = 0
    line_count: int = 0
    token_count: int = 0
    
    # Quality metrics
    complexity_score: float = 0.0
    quality_score: float = 0.0
    importance_score: float = 0.0
    
    # Relationships
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    dependency_ids: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    
    # Classifications
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    
    # Descriptive
    description: Optional[str] = None
    comments: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: Optional[datetime] = None
    
    # Processing metadata
    processed: bool = False
    processing_time_ms: Optional[float] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Flags
    is_test: bool = False
    is_generated: bool = False
    is_deprecated: bool = False
    is_public: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key in ['created_at', 'updated_at', 'accessed_at']:
            if data[key] and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary"""
        # Convert ISO format to datetime objects
        for key in ['created_at', 'updated_at', 'accessed_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)

@dataclass
class FileMetadata:
    """Metadata for source files"""
    file_path: str
    file_hash: str
    
    # File properties
    size_bytes: int = 0
    line_count: int = 0
    language: Optional[str] = None
    encoding: str = "utf-8"
    mime_type: Optional[str] = None
    
    # Repository info
    repository: Optional[str] = None
    project: Optional[str] = None
    branch: Optional[str] = None
    
    # Statistics
    chunk_count: int = 0
    total_complexity: float = 0.0
    average_quality: float = 0.0
    
    # Dependencies
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_processed: Optional[datetime] = None
    
    # Classification
    file_type: str = "unknown"
    is_test: bool = False
    is_generated: bool = False
    
    # Custom attributes
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetadataCache:
    """LRU cache for metadata"""
    
    def __init__(self, max_size: int = 10000):
        """Initialize cache"""
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str, ttl_seconds: int = 3600) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check TTL
                if (datetime.now(timezone.utc) - timestamp).total_seconds() > ttl_seconds:
                    del self.cache[key]
                    self.access_order.remove(key)
                    self.misses += 1
                    return None
                
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Add item to cache"""
        with self.lock:
            # Remove oldest if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            # Add/update item
            self.cache[key] = (value, datetime.now(timezone.utc))
            
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

class BaseMetadataStore:
    """Base class for metadata storage implementations"""
    
    def __init__(self, config: MetadataConfig):
        """Initialize metadata store"""
        self.config = config
        self.lock = threading.RLock()
        
        # Initialize cache
        if config.cache_enabled:
            self.cache = MetadataCache(config.cache_size)
        else:
            self.cache = None
        
        # Statistics
        self.stats = defaultdict(int)
    
    def store_chunk_metadata(self, metadata: ChunkMetadata) -> bool:
        """Store chunk metadata"""
        raise NotImplementedError
    
    def retrieve_chunk_metadata(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Retrieve chunk metadata"""
        raise NotImplementedError
    
    def update_chunk_metadata(self, chunk_id: str, updates: Dict[str, Any]) -> bool:
        """Update chunk metadata"""
        raise NotImplementedError
    
    def delete_chunk_metadata(self, chunk_id: str) -> bool:
        """Delete chunk metadata"""
        raise NotImplementedError
    
    def search_chunks(self, **criteria) -> List[ChunkMetadata]:
        """Search chunks by criteria"""
        raise NotImplementedError
    
    def store_file_metadata(self, metadata: FileMetadata) -> bool:
        """Store file metadata"""
        raise NotImplementedError
    
    def retrieve_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Retrieve file metadata"""
        raise NotImplementedError

class SQLiteMetadataStore(BaseMetadataStore):
    """SQLite-based metadata storage"""
    
    def __init__(self, config: MetadataConfig):
        """Initialize SQLite metadata store"""
        super().__init__(config)
        
        self.db_path = config.storage_path / "metadata.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Enable optimizations
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.lock:
            cursor = self.conn.cursor()
            
            # Chunk metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_hash TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    file_path TEXT NOT NULL,
                    file_hash TEXT,
                    repository TEXT,
                    project TEXT,
                    branch TEXT,
                    commit_hash TEXT,
                    chunk_type TEXT,
                    language TEXT,
                    framework TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    start_char INTEGER,
                    end_char INTEGER,
                    size_bytes INTEGER,
                    line_count INTEGER,
                    token_count INTEGER,
                    complexity_score REAL,
                    quality_score REAL,
                    importance_score REAL,
                    parent_chunk_id TEXT,
                    child_chunk_ids TEXT,
                    dependency_ids TEXT,
                    imports TEXT,
                    exports TEXT,
                    tags TEXT,
                    categories TEXT,
                    patterns TEXT,
                    description TEXT,
                    comments TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    processed BOOLEAN DEFAULT 0,
                    processing_time_ms REAL,
                    embedding_model TEXT,
                    embedding_dimension INTEGER,
                    custom_attributes TEXT,
                    is_test BOOLEAN DEFAULT 0,
                    is_generated BOOLEAN DEFAULT 0,
                    is_deprecated BOOLEAN DEFAULT 0,
                    is_public BOOLEAN DEFAULT 1
                )
            """)
            
            # File metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    size_bytes INTEGER,
                    line_count INTEGER,
                    language TEXT,
                    encoding TEXT,
                    mime_type TEXT,
                    repository TEXT,
                    project TEXT,
                    branch TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    total_complexity REAL DEFAULT 0,
                    average_quality REAL DEFAULT 0,
                    imports TEXT,
                    exports TEXT,
                    dependencies TEXT,
                    created_at TIMESTAMP,
                    modified_at TIMESTAMP,
                    last_processed TIMESTAMP,
                    file_type TEXT,
                    is_test BOOLEAN DEFAULT 0,
                    is_generated BOOLEAN DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # Relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_chunk_id TEXT NOT NULL,
                    target_chunk_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_chunk_id, target_chunk_id, relationship_type),
                    FOREIGN KEY (source_chunk_id) REFERENCES chunk_metadata(chunk_id),
                    FOREIGN KEY (target_chunk_id) REFERENCES chunk_metadata(chunk_id)
                )
            """)
            
            # Tags table (normalized)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_name TEXT UNIQUE NOT NULL,
                    category TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Chunk-tags junction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_tags (
                    chunk_id TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chunk_id, tag_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunk_metadata(chunk_id),
                    FOREIGN KEY (tag_id) REFERENCES tags(id)
                )
            """)
            
            # Processing history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration_ms REAL,
                    error_message TEXT,
                    metadata TEXT,
                    FOREIGN KEY (chunk_id) REFERENCES chunk_metadata(chunk_id)
                )
            """)
            
            # Create indices
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_chunk_file_path ON chunk_metadata(file_path)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_language ON chunk_metadata(language)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunk_metadata(chunk_type)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_repository ON chunk_metadata(repository)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_project ON chunk_metadata(project)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_created_at ON chunk_metadata(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_updated_at ON chunk_metadata(updated_at)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_quality ON chunk_metadata(quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_chunk_complexity ON chunk_metadata(complexity_score)",
                "CREATE INDEX IF NOT EXISTS idx_rel_source ON chunk_relationships(source_chunk_id)",
                "CREATE INDEX IF NOT EXISTS idx_rel_target ON chunk_relationships(target_chunk_id)",
                "CREATE INDEX IF NOT EXISTS idx_file_language ON file_metadata(language)",
                "CREATE INDEX IF NOT EXISTS idx_file_repository ON file_metadata(repository)",
                "CREATE INDEX IF NOT EXISTS idx_processing_chunk ON processing_history(chunk_id)"
            ]
            
            for index in indices:
                cursor.execute(index)
            
            # Create full-text search virtual table
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search 
                USING fts5(
                    chunk_id UNINDEXED,
                    description,
                    comments,
                    tags,
                    content='chunk_metadata',
                    content_rowid='rowid'
                )
            """)
            
            self.conn.commit()
    
    def store_chunk_metadata(self, metadata: ChunkMetadata) -> bool:
        """Store chunk metadata in SQLite"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                # Convert lists and dicts to JSON strings
                data = metadata.to_dict()
                for field in ['child_chunk_ids', 'dependency_ids', 'imports', 'exports', 
                            'tags', 'categories', 'patterns', 'custom_attributes']:
                    if field in data and data[field]:
                        data[field] = json.dumps(data[field])
                
                # Insert or replace
                cursor.execute("""
                    INSERT OR REPLACE INTO chunk_metadata (
                        chunk_id, chunk_hash, version, file_path, file_hash,
                        repository, project, branch, commit_hash, chunk_type,
                        language, framework, start_line, end_line, start_char, end_char,
                        size_bytes, line_count, token_count, complexity_score,
                        quality_score, importance_score, parent_chunk_id,
                        child_chunk_ids, dependency_ids, imports, exports,
                        tags, categories, patterns, description, comments,
                        created_at, updated_at, accessed_at, processed,
                        processing_time_ms, embedding_model, embedding_dimension,
                        custom_attributes, is_test, is_generated, is_deprecated, is_public
                    ) VALUES (
                        :chunk_id, :chunk_hash, :version, :file_path, :file_hash,
                        :repository, :project, :branch, :commit_hash, :chunk_type,
                        :language, :framework, :start_line, :end_line, :start_char, :end_char,
                        :size_bytes, :line_count, :token_count, :complexity_score,
                        :quality_score, :importance_score, :parent_chunk_id,
                        :child_chunk_ids, :dependency_ids, :imports, :exports,
                        :tags, :categories, :patterns, :description, :comments,
                        :created_at, :updated_at, :accessed_at, :processed,
                        :processing_time_ms, :embedding_model, :embedding_dimension,
                        :custom_attributes, :is_test, :is_generated, :is_deprecated, :is_public
                    )
                """, data)
                
                # Update full-text search
                if metadata.description or metadata.comments or metadata.tags:
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunk_search (chunk_id, description, comments, tags)
                        VALUES (?, ?, ?, ?)
                    """, (
                        metadata.chunk_id,
                        metadata.description,
                        metadata.comments,
                        ' '.join(metadata.tags) if metadata.tags else None
                    ))
                
                # Handle tags
                if metadata.tags:
                    for tag in metadata.tags:
                        # Insert tag if not exists
                        cursor.execute("""
                            INSERT OR IGNORE INTO tags (tag_name) VALUES (?)
                        """, (tag,))
                        
                        # Link chunk to tag
                        cursor.execute("""
                            INSERT OR IGNORE INTO chunk_tags (chunk_id, tag_id)
                            SELECT ?, id FROM tags WHERE tag_name = ?
                        """, (metadata.chunk_id, tag))
                
                self.conn.commit()
                
                # Update cache
                if self.cache:
                    self.cache.put(f"chunk:{metadata.chunk_id}", metadata)
                
                self.stats['chunks_stored'] += 1
                return True
                
        except Exception as e:
            logger.error(f"Failed to store chunk metadata {metadata.chunk_id}: {e}")
            self.stats['store_errors'] += 1
            return False
    
    def retrieve_chunk_metadata(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Retrieve chunk metadata from SQLite"""
        try:
            # Check cache
            if self.cache:
                cached = self.cache.get(f"chunk:{chunk_id}")
                if cached:
                    self.stats['cache_hits'] += 1
                    return cached
                self.stats['cache_misses'] += 1
            
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute("""
                    SELECT * FROM chunk_metadata WHERE chunk_id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                
                if row:
                    # Convert row to dict
                    data = dict(row)
                    
                    # Parse JSON fields
                    for field in ['child_chunk_ids', 'dependency_ids', 'imports', 'exports',
                                'tags', 'categories', 'patterns', 'custom_attributes']:
                        if data.get(field):
                            data[field] = json.loads(data[field])
                    
                    # Create metadata object
                    metadata = ChunkMetadata.from_dict(data)
                    
                    # Update access time
                    cursor.execute("""
                        UPDATE chunk_metadata SET accessed_at = ? WHERE chunk_id = ?
                    """, (datetime.now(timezone.utc), chunk_id))
                    self.conn.commit()
                    
                    # Update cache
                    if self.cache:
                        self.cache.put(f"chunk:{chunk_id}", metadata)
                    
                    self.stats['chunks_retrieved'] += 1
                    return metadata
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve chunk metadata {chunk_id}: {e}")
            self.stats['retrieve_errors'] += 1
            return None
    
    def update_chunk_metadata(self, chunk_id: str, updates: Dict[str, Any]) -> bool:
        """Update chunk metadata"""
        try:
            with self.lock:
                # Build update query
                set_clauses = []
                params = []
                
                for key, value in updates.items():
                    # Skip non-updateable fields
                    if key in ['chunk_id', 'created_at']:
                        continue
                    
                    # Handle list/dict fields
                    if key in ['child_chunk_ids', 'dependency_ids', 'imports', 'exports',
                              'tags', 'categories', 'patterns', 'custom_attributes']:
                        value = json.dumps(value) if value else None
                    
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                
                # Add updated_at
                set_clauses.append("updated_at = ?")
                params.append(datetime.now(timezone.utc))
                
                # Add chunk_id to params
                params.append(chunk_id)
                
                # Execute update
                cursor = self.conn.cursor()
                cursor.execute(f"""
                    UPDATE chunk_metadata 
                    SET {', '.join(set_clauses)}
                    WHERE chunk_id = ?
                """, params)
                
                self.conn.commit()
                
                # Invalidate cache
                if self.cache:
                    cache_key = f"chunk:{chunk_id}"
                    if cache_key in self.cache.cache:
                        del self.cache.cache[cache_key]
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update chunk metadata {chunk_id}: {e}")
            return False
    
    def delete_chunk_metadata(self, chunk_id: str) -> bool:
        """Delete chunk metadata"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                # Delete from all tables
                cursor.execute("DELETE FROM chunk_tags WHERE chunk_id = ?", (chunk_id,))
                cursor.execute("DELETE FROM chunk_relationships WHERE source_chunk_id = ? OR target_chunk_id = ?", 
                             (chunk_id, chunk_id))
                cursor.execute("DELETE FROM processing_history WHERE chunk_id = ?", (chunk_id,))
                cursor.execute("DELETE FROM chunk_search WHERE chunk_id = ?", (chunk_id,))
                cursor.execute("DELETE FROM chunk_metadata WHERE chunk_id = ?", (chunk_id,))
                
                self.conn.commit()
                
                # Remove from cache
                if self.cache:
                    cache_key = f"chunk:{chunk_id}"
                    if cache_key in self.cache.cache:
                        del self.cache.cache[cache_key]
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete chunk metadata {chunk_id}: {e}")
            return False
    
    def search_chunks(self, **criteria) -> List[ChunkMetadata]:
        """Search chunks by criteria"""
        try:
            with self.lock:
                query = "SELECT * FROM chunk_metadata WHERE 1=1"
                params = []
                
                # Build WHERE clause
                if 'file_path' in criteria:
                    query += " AND file_path = ?"
                    params.append(criteria['file_path'])
                
                if 'language' in criteria:
                    query += " AND language = ?"
                    params.append(criteria['language'])
                
                if 'chunk_type' in criteria:
                    query += " AND chunk_type = ?"
                    params.append(criteria['chunk_type'])
                
                if 'repository' in criteria:
                    query += " AND repository = ?"
                    params.append(criteria['repository'])
                
                if 'project' in criteria:
                    query += " AND project = ?"
                    params.append(criteria['project'])
                
                if 'min_quality' in criteria:
                    query += " AND quality_score >= ?"
                    params.append(criteria['min_quality'])
                
                if 'max_complexity' in criteria:
                    query += " AND complexity_score <= ?"
                    params.append(criteria['max_complexity'])
                
                if 'tags' in criteria:
                    # Search for chunks with specific tags
                    tag_list = criteria['tags'] if isinstance(criteria['tags'], list) else [criteria['tags']]
                    placeholders = ','.join(['?'] * len(tag_list))
                    query += f"""
                        AND chunk_id IN (
                            SELECT ct.chunk_id 
                            FROM chunk_tags ct
                            JOIN tags t ON ct.tag_id = t.id
                            WHERE t.tag_name IN ({placeholders})
                        )
                    """
                    params.extend(tag_list)
                
                if 'full_text' in criteria:
                    # Full-text search
                    query = """
                        SELECT cm.* FROM chunk_metadata cm
                        JOIN chunk_search cs ON cm.chunk_id = cs.chunk_id
                        WHERE cs.chunk_search MATCH ?
                    """
                    params = [criteria['full_text']]
                
                # Add ordering
                order_by = criteria.get('order_by', 'created_at')
                order_dir = criteria.get('order_dir', 'DESC')
                query += f" ORDER BY {order_by} {order_dir}"
                
                # Add limit
                limit = min(criteria.get('limit', 100), self.config.max_query_results)
                query += f" LIMIT {limit}"
                
                if 'offset' in criteria:
                    query += f" OFFSET {criteria['offset']}"
                
                # Execute query
                cursor = self.conn.cursor()
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    data = dict(row)
                    
                    # Parse JSON fields
                    for field in ['child_chunk_ids', 'dependency_ids', 'imports', 'exports',
                                'tags', 'categories', 'patterns', 'custom_attributes']:
                        if data.get(field):
                            data[field] = json.loads(data[field])
                    
                    results.append(ChunkMetadata.from_dict(data))
                
                self.stats['searches'] += 1
                return results
                
        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            self.stats['search_errors'] += 1
            return []
    
    def store_file_metadata(self, metadata: FileMetadata) -> bool:
        """Store file metadata"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                # Convert lists to JSON
                data = asdict(metadata)
                for field in ['imports', 'exports', 'dependencies', 'metadata']:
                    if field in data and data[field]:
                        data[field] = json.dumps(data[field])
                
                cursor.execute("""
                    INSERT OR REPLACE INTO file_metadata (
                        file_path, file_hash, size_bytes, line_count, language,
                        encoding, mime_type, repository, project, branch,
                        chunk_count, total_complexity, average_quality,
                        imports, exports, dependencies, created_at, modified_at,
                        last_processed, file_type, is_test, is_generated, metadata
                    ) VALUES (
                        :file_path, :file_hash, :size_bytes, :line_count, :language,
                        :encoding, :mime_type, :repository, :project, :branch,
                        :chunk_count, :total_complexity, :average_quality,
                        :imports, :exports, :dependencies, :created_at, :modified_at,
                        :last_processed, :file_type, :is_test, :is_generated, :metadata
                    )
                """, data)
                
                self.conn.commit()
                
                # Update cache
                if self.cache:
                    self.cache.put(f"file:{metadata.file_path}", metadata)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store file metadata {metadata.file_path}: {e}")
            return False
    
    def retrieve_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Retrieve file metadata"""
        try:
            # Check cache
            if self.cache:
                cached = self.cache.get(f"file:{file_path}")
                if cached:
                    return cached
            
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute("""
                    SELECT * FROM file_metadata WHERE file_path = ?
                """, (file_path,))
                
                row = cursor.fetchone()
                
                if row:
                    data = dict(row)
                    
                    # Parse JSON fields
                    for field in ['imports', 'exports', 'dependencies', 'metadata']:
                        if data.get(field):
                            data[field] = json.loads(data[field])
                    
                    # Convert timestamps
                    for field in ['created_at', 'modified_at', 'last_processed']:
                        if data.get(field):
                            data[field] = datetime.fromisoformat(data[field])
                    
                    metadata = FileMetadata(**data)
                    
                    # Update cache
                    if self.cache:
                        self.cache.put(f"file:{file_path}", metadata)
                    
                    return metadata
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve file metadata {file_path}: {e}")
            return None
    
    def add_chunk_relationship(self, source_id: str, target_id: str, 
                              relationship_type: str, strength: float = 1.0,
                              metadata: Optional[Dict] = None) -> bool:
        """Add relationship between chunks"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO chunk_relationships 
                    (source_chunk_id, target_chunk_id, relationship_type, strength, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    source_id, target_id, relationship_type, strength,
                    json.dumps(metadata) if metadata else None
                ))
                
                self.conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to add chunk relationship: {e}")
            return False
    
    def get_chunk_relationships(self, chunk_id: str, 
                               relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get relationships for a chunk"""
        try:
            with self.lock:
                query = """
                    SELECT * FROM chunk_relationships 
                    WHERE source_chunk_id = ? OR target_chunk_id = ?
                """
                params = [chunk_id, chunk_id]
                
                if relationship_type:
                    query += " AND relationship_type = ?"
                    params.append(relationship_type)
                
                cursor = self.conn.cursor()
                cursor.execute(query, params)
                
                relationships = []
                for row in cursor.fetchall():
                    rel = dict(row)
                    if rel['metadata']:
                        rel['metadata'] = json.loads(rel['metadata'])
                    relationships.append(rel)
                
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to get chunk relationships: {e}")
            return []
    
    def add_processing_history(self, chunk_id: str, operation: str,
                              status: str, duration_ms: Optional[float] = None,
                              error_message: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> bool:
        """Add processing history entry"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_history 
                    (chunk_id, operation, status, started_at, completed_at, 
                     duration_ms, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id, operation, status,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc) if duration_ms else None,
                    duration_ms, error_message,
                    json.dumps(metadata) if metadata else None
                ))
                
                self.conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to add processing history: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metadata store statistics"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                stats = {
                    'total_chunks': cursor.execute("SELECT COUNT(*) FROM chunk_metadata").fetchone()[0],
                    'total_files': cursor.execute("SELECT COUNT(*) FROM file_metadata").fetchone()[0],
                    'total_relationships': cursor.execute("SELECT COUNT(*) FROM chunk_relationships").fetchone()[0],
                    'total_tags': cursor.execute("SELECT COUNT(*) FROM tags").fetchone()[0],
                    'languages': cursor.execute("SELECT DISTINCT language FROM chunk_metadata WHERE language IS NOT NULL").fetchall(),
                    'repositories': cursor.execute("SELECT DISTINCT repository FROM chunk_metadata WHERE repository IS NOT NULL").fetchall(),
                    'average_quality': cursor.execute("SELECT AVG(quality_score) FROM chunk_metadata").fetchone()[0],
                    'average_complexity': cursor.execute("SELECT AVG(complexity_score) FROM chunk_metadata").fetchone()[0],
                }
                
                # Add cache stats
                if self.cache:
                    stats['cache'] = {
                        'hits': self.cache.hits,
                        'misses': self.cache.misses,
                        'hit_rate': self.cache.hits / (self.cache.hits + self.cache.misses) if (self.cache.hits + self.cache.misses) > 0 else 0
                    }
                
                # Add operation stats
                stats['operations'] = dict(self.stats)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def vacuum(self):
        """Optimize database"""
        try:
            with self.lock:
                self.conn.execute("VACUUM")
                self.conn.execute("ANALYZE")
                logger.info("Database optimized")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()

class MetadataStoreFactory:
    """Factory for creating metadata store instances"""
    
    @staticmethod
    def create(config: MetadataConfig) -> BaseMetadataStore:
        """Create metadata store based on backend"""
        if config.backend == MetadataBackend.SQLITE:
            return SQLiteMetadataStore(config)
        # Add other backends as needed
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

# Convenience functions
def create_metadata_store(backend: str = "sqlite",
                         path: str = "storage/metadata",
                         **kwargs) -> BaseMetadataStore:
    """Create metadata store with common settings"""
    config = MetadataConfig(
        backend=MetadataBackend(backend),
        storage_path=Path(path),
        **kwargs
    )
    return MetadataStoreFactory.create(config)