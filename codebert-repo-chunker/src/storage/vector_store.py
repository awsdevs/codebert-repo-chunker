"""
Vector storage system for managing code embeddings at scale
Provides multiple backend options with efficient similarity search
"""

import numpy as np
import json
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import logging
import threading
from contextlib import contextmanager
import hashlib
import tempfile
import shutil
import mmap
from collections import defaultdict

# Vector database backends
import faiss
import hnswlib
import annoy
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)
import chromadb
from chromadb.config import Settings
import weaviate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pinecone
from elasticsearch import Elasticsearch
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import torch

logger = logging.getLogger(__name__)

class VectorBackend(Enum):
    """Available vector storage backends"""
    FAISS = "faiss"
    HNSWLIB = "hnswlib"
    ANNOY = "annoy"
    MILVUS = "milvus"
    CHROMADB = "chromadb"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"
    NUMPY = "numpy"

class DistanceMetric(Enum):
    """Distance metrics for similarity search"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    HAMMING = "hamming"

class IndexType(Enum):
    """Types of vector indices"""
    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"
    LSH = "lsh"
    ANNOY = "annoy"
    SCANN = "scann"

@dataclass
class VectorConfig:
    """Configuration for vector storage"""
    backend: VectorBackend = VectorBackend.FAISS
    storage_path: Path = Path("storage/vectors")
    
    # Connection settings
    host: str = "localhost"
    port: Optional[int] = None
    api_key: Optional[str] = None
    collection_name: str = "code_embeddings"
    
    # Vector settings
    dimension: int = 768
    metric: DistanceMetric = DistanceMetric.COSINE
    dtype: str = "float32"
    
    # Index settings
    index_type: IndexType = IndexType.HNSW
    index_params: Dict[str, Any] = field(default_factory=lambda: {
        "M": 16,  # HNSW connections
        "ef_construction": 200,  # HNSW construction
        "nlist": 1024,  # IVF clusters
        "nprobe": 10,  # IVF search clusters
        "n_trees": 10,  # Annoy trees
    })
    
    # Performance settings
    batch_size: int = 1000
    cache_enabled: bool = True
    cache_size: int = 10000
    num_threads: int = 4
    use_gpu: bool = False
    
    # Storage settings
    persist_enabled: bool = True
    compress_vectors: bool = False
    quantization: Optional[str] = None  # "scalar", "product"
    
    # Sharding
    enable_sharding: bool = False
    shard_count: int = 4
    replicas: int = 1
    
    # Limits
    max_vectors: Optional[int] = None
    max_memory_gb: Optional[float] = 8.0
    
    def __post_init__(self):
        """Initialize storage path"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

@dataclass
class VectorMetadata:
    """Metadata associated with a vector"""
    vector_id: str
    chunk_id: str
    file_path: str
    dimension: int
    norm: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    language: Optional[str] = None
    chunk_type: Optional[str] = None
    quality_score: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    """Result from vector search"""
    vector_id: str
    score: float
    distance: float
    metadata: VectorMetadata
    vector: Optional[np.ndarray] = None

class VectorCache:
    """LRU cache for vectors"""
    
    def __init__(self, max_size: int = 10000, dimension: int = 768):
        """Initialize vector cache"""
        self.max_size = max_size
        self.dimension = dimension
        self.cache = {}
        self.access_order = []
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, vector_id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """Get vector from cache"""
        with self.lock:
            if vector_id in self.cache:
                # Update access order
                self.access_order.remove(vector_id)
                self.access_order.append(vector_id)
                self.hits += 1
                return self.cache[vector_id]
            self.misses += 1
            return None
    
    def put(self, vector_id: str, vector: np.ndarray, metadata: VectorMetadata):
        """Add vector to cache"""
        with self.lock:
            # Remove oldest if cache is full
            if len(self.cache) >= self.max_size and vector_id not in self.cache:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            # Add to cache
            self.cache[vector_id] = (vector.copy(), metadata)
            
            if vector_id in self.access_order:
                self.access_order.remove(vector_id)
            self.access_order.append(vector_id)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

class BaseVectorStore:
    """Base class for vector storage implementations"""
    
    def __init__(self, config: VectorConfig):
        """Initialize vector store"""
        self.config = config
        self.lock = threading.RLock()
        
        # Initialize cache
        if config.cache_enabled:
            self.cache = VectorCache(config.cache_size, config.dimension)
        else:
            self.cache = None
        
        # Statistics
        self.stats = defaultdict(int)
    
    def add(self, vector_id: str, vector: np.ndarray, 
           metadata: Optional[VectorMetadata] = None) -> bool:
        """Add a vector to the store"""
        raise NotImplementedError
    
    def get(self, vector_id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """Retrieve a vector by ID"""
        raise NotImplementedError
    
    def delete(self, vector_id: str) -> bool:
        """Delete a vector"""
        raise NotImplementedError
    
    def search(self, query_vector: np.ndarray, k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors"""
        raise NotImplementedError
    
    def batch_add(self, vectors: List[Tuple[str, np.ndarray, VectorMetadata]]) -> int:
        """Add multiple vectors"""
        raise NotImplementedError
    
    def exists(self, vector_id: str) -> bool:
        """Check if vector exists"""
        raise NotImplementedError
    
    def count(self) -> int:
        """Get total number of vectors"""
        raise NotImplementedError
    
    def clear(self) -> bool:
        """Clear all vectors"""
        raise NotImplementedError

class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector storage"""
    
    def __init__(self, config: VectorConfig):
        """Initialize FAISS vector store"""
        super().__init__(config)
        
        # Initialize index
        self.index = self._create_index()
        
        # ID mapping
        self.id_to_index = {}
        self.index_to_id = {}
        self.metadata_store = {}
        
        # Index file path
        self.index_path = config.storage_path / "faiss.index"
        self.metadata_path = config.storage_path / "faiss_metadata.pkl"
        
        # Load existing index if available
        self._load_index()
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        dimension = self.config.dimension
        
        # Create base index based on metric
        if self.config.metric == DistanceMetric.COSINE:
            # Use inner product with normalized vectors for cosine similarity
            index = faiss.IndexFlatIP(dimension)
        elif self.config.metric == DistanceMetric.EUCLIDEAN:
            index = faiss.IndexFlatL2(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)
        
        # Wrap with specialized index type
        if self.config.index_type == IndexType.IVF_FLAT:
            nlist = self.config.index_params.get("nlist", 1024)
            quantizer = index
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        elif self.config.index_type == IndexType.IVF_PQ:
            nlist = self.config.index_params.get("nlist", 1024)
            m = self.config.index_params.get("m", 8)  # number of subquantizers
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
        
        elif self.config.index_type == IndexType.HNSW:
            M = self.config.index_params.get("M", 16)
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = self.config.index_params.get("ef_construction", 200)
        
        elif self.config.index_type == IndexType.LSH:
            nbits = self.config.index_params.get("nbits", dimension * 8)
            index = faiss.IndexLSH(dimension, nbits)
        
        # Enable GPU if configured
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def add(self, vector_id: str, vector: np.ndarray,
           metadata: Optional[VectorMetadata] = None) -> bool:
        """Add vector to FAISS index"""
        try:
            with self.lock:
                # Check if vector already exists
                if vector_id in self.id_to_index:
                    logger.warning(f"Vector {vector_id} already exists")
                    return False
                
                # Normalize vector for cosine similarity
                if self.config.metric == DistanceMetric.COSINE:
                    vector = vector / np.linalg.norm(vector)
                
                # Get next index
                idx = self.index.ntotal
                
                # Add to index
                self.index.add(vector.reshape(1, -1).astype(np.float32))
                
                # Update mappings
                self.id_to_index[vector_id] = idx
                self.index_to_id[idx] = vector_id
                
                # Store metadata
                if metadata is None:
                    metadata = VectorMetadata(
                        vector_id=vector_id,
                        chunk_id=vector_id,
                        file_path="",
                        dimension=self.config.dimension,
                        norm=float(np.linalg.norm(vector))
                    )
                self.metadata_store[vector_id] = metadata
                
                # Update cache
                if self.cache:
                    self.cache.put(vector_id, vector, metadata)
                
                self.stats['vectors_added'] += 1
                
                # Auto-save if configured
                if self.config.persist_enabled and self.stats['vectors_added'] % 1000 == 0:
                    self._save_index()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to add vector {vector_id}: {e}")
            self.stats['add_errors'] += 1
            return False
    
    def get(self, vector_id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """Retrieve vector from FAISS index"""
        try:
            # Check cache first
            if self.cache:
                cached = self.cache.get(vector_id)
                if cached:
                    return cached
            
            with self.lock:
                if vector_id not in self.id_to_index:
                    return None
                
                # Get index
                idx = self.id_to_index[vector_id]
                
                # Reconstruct vector
                vector = np.zeros((1, self.config.dimension), dtype=np.float32)
                self.index.reconstruct(int(idx), vector[0])
                
                # Get metadata
                metadata = self.metadata_store.get(vector_id)
                
                # Update cache
                if self.cache and metadata:
                    self.cache.put(vector_id, vector[0], metadata)
                
                self.stats['vectors_retrieved'] += 1
                
                return (vector[0], metadata) if metadata else None
                
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            self.stats['get_errors'] += 1
            return None
    
    def delete(self, vector_id: str) -> bool:
        """Delete vector from FAISS index"""
        try:
            with self.lock:
                if vector_id not in self.id_to_index:
                    return False
                
                # FAISS doesn't support deletion directly
                # We'll mark it as deleted and rebuild index periodically
                idx = self.id_to_index[vector_id]
                
                # Remove from mappings
                del self.id_to_index[vector_id]
                del self.index_to_id[idx]
                del self.metadata_store[vector_id]
                
                # Remove from cache
                if self.cache and vector_id in self.cache.cache:
                    del self.cache.cache[vector_id]
                
                self.stats['vectors_deleted'] += 1
                
                # Rebuild index if too many deletions
                if self.stats['vectors_deleted'] > 100:
                    self._rebuild_index()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors in FAISS index"""
        try:
            with self.lock:
                if self.index.ntotal == 0:
                    return []
                
                # Normalize query vector for cosine similarity
                if self.config.metric == DistanceMetric.COSINE:
                    query_vector = query_vector / np.linalg.norm(query_vector)
                
                # Set search parameters for IVF indices
                if hasattr(self.index, 'nprobe'):
                    self.index.nprobe = self.config.index_params.get("nprobe", 10)
                
                # Search
                k_search = min(k * 3 if filters else k, self.index.ntotal)
                distances, indices = self.index.search(
                    query_vector.reshape(1, -1).astype(np.float32),
                    k_search
                )
                
                # Convert to results
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0:  # Invalid index
                        continue
                    
                    vector_id = self.index_to_id.get(int(idx))
                    if not vector_id:
                        continue
                    
                    metadata = self.metadata_store.get(vector_id)
                    if not metadata:
                        continue
                    
                    # Apply filters
                    if filters:
                        skip = False
                        for key, value in filters.items():
                            if hasattr(metadata, key):
                                if getattr(metadata, key) != value:
                                    skip = True
                                    break
                            elif key in metadata.custom_metadata:
                                if metadata.custom_metadata[key] != value:
                                    skip = True
                                    break
                        
                        if skip:
                            continue
                    
                    # Calculate score (similarity)
                    if self.config.metric == DistanceMetric.COSINE:
                        score = float(dist)  # Inner product for normalized vectors
                    else:
                        score = 1.0 / (1.0 + float(dist))  # Convert distance to similarity
                    
                    results.append(SearchResult(
                        vector_id=vector_id,
                        score=score,
                        distance=float(dist),
                        metadata=metadata
                    ))
                    
                    if len(results) >= k:
                        break
                
                self.stats['searches'] += 1
                return results
                
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            self.stats['search_errors'] += 1
            return []
    
    def batch_add(self, vectors: List[Tuple[str, np.ndarray, VectorMetadata]]) -> int:
        """Add multiple vectors to FAISS index"""
        try:
            with self.lock:
                added = 0
                
                # Prepare batch
                batch_vectors = []
                batch_ids = []
                batch_metadata = []
                
                for vector_id, vector, metadata in vectors:
                    if vector_id in self.id_to_index:
                        continue
                    
                    # Normalize if needed
                    if self.config.metric == DistanceMetric.COSINE:
                        vector = vector / np.linalg.norm(vector)
                    
                    batch_vectors.append(vector)
                    batch_ids.append(vector_id)
                    batch_metadata.append(metadata)
                
                if batch_vectors:
                    # Convert to numpy array
                    batch_array = np.vstack(batch_vectors).astype(np.float32)
                    
                    # Get starting index
                    start_idx = self.index.ntotal
                    
                    # Add to index
                    self.index.add(batch_array)
                    
                    # Update mappings
                    for i, (vector_id, metadata) in enumerate(zip(batch_ids, batch_metadata)):
                        idx = start_idx + i
                        self.id_to_index[vector_id] = idx
                        self.index_to_id[idx] = vector_id
                        self.metadata_store[vector_id] = metadata
                        added += 1
                    
                    self.stats['vectors_added'] += added
                
                # Save if configured
                if self.config.persist_enabled:
                    self._save_index()
                
                return added
                
        except Exception as e:
            logger.error(f"Failed to batch add vectors: {e}")
            return 0
    
    def exists(self, vector_id: str) -> bool:
        """Check if vector exists"""
        return vector_id in self.id_to_index
    
    def count(self) -> int:
        """Get total number of vectors"""
        return len(self.id_to_index)
    
    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            with self.lock:
                self.index = self._create_index()
                self.id_to_index.clear()
                self.index_to_id.clear()
                self.metadata_store.clear()
                
                if self.cache:
                    self.cache.clear()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False
    
    def _rebuild_index(self):
        """Rebuild index after deletions"""
        logger.info("Rebuilding FAISS index")
        
        # Get all valid vectors
        valid_vectors = []
        valid_ids = []
        valid_metadata = []
        
        for vector_id in self.id_to_index:
            result = self.get(vector_id)
            if result:
                vector, metadata = result
                valid_vectors.append(vector)
                valid_ids.append(vector_id)
                valid_metadata.append(metadata)
        
        # Create new index
        self.index = self._create_index()
        self.id_to_index.clear()
        self.index_to_id.clear()
        
        # Re-add vectors
        if valid_vectors:
            batch_array = np.vstack(valid_vectors).astype(np.float32)
            self.index.add(batch_array)
            
            for i, (vector_id, metadata) in enumerate(zip(valid_ids, valid_metadata)):
                self.id_to_index[vector_id] = i
                self.index_to_id[i] = vector_id
                self.metadata_store[vector_id] = metadata
        
        self.stats['vectors_deleted'] = 0
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            # Save index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'metadata_store': self.metadata_store
                }, f)
            
            logger.debug("FAISS index saved")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _load_index(self):
        """Load FAISS index from disk"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_index = data['id_to_index']
                    self.index_to_id = data['index_to_id']
                    self.metadata_store = data['metadata_store']
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")

class HNSWLibVectorStore(BaseVectorStore):
    """HNSWLib-based vector storage"""
    
    def __init__(self, config: VectorConfig):
        """Initialize HNSWLib vector store"""
        super().__init__(config)
        
        # Map distance metrics
        space_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip"
        }
        
        # Initialize index
        self.index = hnswlib.Index(
            space=space_map.get(config.metric, "l2"),
            dim=config.dimension
        )
        
        # Initialize with initial capacity
        max_elements = config.max_vectors or 1000000
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=config.index_params.get("ef_construction", 200),
            M=config.index_params.get("M", 16)
        )
        
        # Set number of threads
        self.index.set_num_threads(config.num_threads)
        
        # ID mapping and metadata
        self.id_to_index = {}
        self.index_to_id = {}
        self.metadata_store = {}
        
        # Index file path
        self.index_path = config.storage_path / "hnswlib.index"
        self.metadata_path = config.storage_path / "hnswlib_metadata.pkl"
        
        # Load existing index if available
        self._load_index()
    
    def add(self, vector_id: str, vector: np.ndarray,
           metadata: Optional[VectorMetadata] = None) -> bool:
        """Add vector to HNSWLib index"""
        try:
            with self.lock:
                if vector_id in self.id_to_index:
                    return False
                
                # Get next index
                idx = len(self.id_to_index)
                
                # Add to index
                self.index.add_items(vector.reshape(1, -1), np.array([idx]))
                
                # Update mappings
                self.id_to_index[vector_id] = idx
                self.index_to_id[idx] = vector_id
                
                # Store metadata
                if metadata is None:
                    metadata = VectorMetadata(
                        vector_id=vector_id,
                        chunk_id=vector_id,
                        file_path="",
                        dimension=self.config.dimension,
                        norm=float(np.linalg.norm(vector))
                    )
                self.metadata_store[vector_id] = metadata
                
                # Update cache
                if self.cache:
                    self.cache.put(vector_id, vector, metadata)
                
                self.stats['vectors_added'] += 1
                
                # Auto-save
                if self.config.persist_enabled and self.stats['vectors_added'] % 1000 == 0:
                    self._save_index()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to add vector {vector_id}: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors in HNSWLib index"""
        try:
            with self.lock:
                if len(self.id_to_index) == 0:
                    return []
                
                # Set ef parameter for search
                self.index.set_ef(max(k * 2, 50))
                
                # Search
                k_search = min(k * 3 if filters else k, len(self.id_to_index))
                indices, distances = self.index.knn_query(query_vector, k=k_search)
                
                # Convert to results
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    vector_id = self.index_to_id.get(idx)
                    if not vector_id:
                        continue
                    
                    metadata = self.metadata_store.get(vector_id)
                    if not metadata:
                        continue
                    
                    # Apply filters
                    if filters:
                        skip = False
                        for key, value in filters.items():
                            if hasattr(metadata, key):
                                if getattr(metadata, key) != value:
                                    skip = True
                                    break
                        
                        if skip:
                            continue
                    
                    # Calculate score
                    if self.config.metric == DistanceMetric.COSINE:
                        score = 1.0 - float(dist)
                    else:
                        score = 1.0 / (1.0 + float(dist))
                    
                    results.append(SearchResult(
                        vector_id=vector_id,
                        score=score,
                        distance=float(dist),
                        metadata=metadata
                    ))
                    
                    if len(results) >= k:
                        break
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def _save_index(self):
        """Save HNSWLib index to disk"""
        try:
            self.index.save_index(str(self.index_path))
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'metadata_store': self.metadata_store
                }, f)
            
            logger.debug("HNSWLib index saved")
            
        except Exception as e:
            logger.error(f"Failed to save HNSWLib index: {e}")
    
    def _load_index(self):
        """Load HNSWLib index from disk"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index.load_index(str(self.index_path))
                
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_index = data['id_to_index']
                    self.index_to_id = data['index_to_id']
                    self.metadata_store = data['metadata_store']
                
                logger.info(f"Loaded HNSWLib index with {len(self.id_to_index)} vectors")
                
        except Exception as e:
            logger.error(f"Failed to load HNSWLib index: {e}")
    
    def get(self, vector_id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """Retrieve vector from HNSWLib index"""
        try:
            # Check cache first
            if self.cache:
                cached = self.cache.get(vector_id)
                if cached:
                    return cached
            
            with self.lock:
                if vector_id not in self.id_to_index:
                    return None
                
                idx = self.id_to_index[vector_id]
                
                # Get vector
                vector = self.index.get_items([idx])[0]
                
                # Get metadata
                metadata = self.metadata_store.get(vector_id)
                
                # Update cache
                if self.cache and metadata:
                    self.cache.put(vector_id, vector, metadata)
                
                return (vector, metadata) if metadata else None
                
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            return None
    
    def delete(self, vector_id: str) -> bool:
        """Delete vector (mark as deleted)"""
        try:
            with self.lock:
                if vector_id not in self.id_to_index:
                    return False
                
                # Mark as deleted
                idx = self.id_to_index[vector_id]
                self.index.mark_deleted(idx)
                
                # Remove from mappings
                del self.id_to_index[vector_id]
                del self.index_to_id[idx]
                del self.metadata_store[vector_id]
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False
    
    def exists(self, vector_id: str) -> bool:
        """Check if vector exists"""
        return vector_id in self.id_to_index
    
    def count(self) -> int:
        """Get total number of vectors"""
        return len(self.id_to_index)
    
    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            with self.lock:
                # Reinitialize index
                max_elements = self.config.max_vectors or 1000000
                self.index.init_index(
                    max_elements=max_elements,
                    ef_construction=self.config.index_params.get("ef_construction", 200),
                    M=self.config.index_params.get("M", 16)
                )
                
                # Clear mappings
                self.id_to_index.clear()
                self.index_to_id.clear()
                self.metadata_store.clear()
                
                if self.cache:
                    self.cache.clear()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False
    
    def batch_add(self, vectors: List[Tuple[str, np.ndarray, VectorMetadata]]) -> int:
        """Add multiple vectors"""
        added = 0
        for vector_id, vector, metadata in vectors:
            if self.add(vector_id, vector, metadata):
                added += 1
        return added

class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB-based vector storage"""
    
    def __init__(self, config: VectorConfig):
        """Initialize ChromaDB vector store"""
        super().__init__(config)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(config.storage_path / "chromadb"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(config.collection_name)
            logger.info(f"Using existing ChromaDB collection: {config.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": config.metric.value}
            )
            logger.info(f"Created new ChromaDB collection: {config.collection_name}")
    
    def add(self, vector_id: str, vector: np.ndarray,
           metadata: Optional[VectorMetadata] = None) -> bool:
        """Add vector to ChromaDB"""
        try:
            # Convert metadata to dict
            meta_dict = {}
            if metadata:
                meta_dict = {
                    'chunk_id': metadata.chunk_id,
                    'file_path': metadata.file_path,
                    'language': metadata.language or '',
                    'chunk_type': metadata.chunk_type or '',
                }
                
                # Add custom metadata
                meta_dict.update(metadata.custom_metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=[vector.tolist()],
                ids=[vector_id],
                metadatas=[meta_dict] if meta_dict else None
            )
            
            self.stats['vectors_added'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vector {vector_id}: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors in ChromaDB"""
        try:
            # Build where clause for filters
            where = filters if filters else None
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=k,
                where=where
            )
            
            # Convert to SearchResult
            search_results = []
            
            for i, vector_id in enumerate(results['ids'][0]):
                # Create metadata
                meta_dict = results['metadatas'][0][i] if results['metadatas'] else {}
                metadata = VectorMetadata(
                    vector_id=vector_id,
                    chunk_id=meta_dict.get('chunk_id', vector_id),
                    file_path=meta_dict.get('file_path', ''),
                    dimension=self.config.dimension,
                    norm=0.0,
                    language=meta_dict.get('language'),
                    chunk_type=meta_dict.get('chunk_type'),
                    custom_metadata=meta_dict
                )
                
                # Calculate score from distance
                distance = results['distances'][0][i] if results['distances'] else 0.0
                if self.config.metric == DistanceMetric.COSINE:
                    score = 1.0 - distance
                else:
                    score = 1.0 / (1.0 + distance)
                
                search_results.append(SearchResult(
                    vector_id=vector_id,
                    score=score,
                    distance=distance,
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def get(self, vector_id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """Retrieve vector from ChromaDB"""
        try:
            result = self.collection.get(ids=[vector_id])
            
            if result['ids']:
                # Get vector
                vector = np.array(result['embeddings'][0]) if result['embeddings'] else None
                
                # Create metadata
                meta_dict = result['metadatas'][0] if result['metadatas'] else {}
                metadata = VectorMetadata(
                    vector_id=vector_id,
                    chunk_id=meta_dict.get('chunk_id', vector_id),
                    file_path=meta_dict.get('file_path', ''),
                    dimension=self.config.dimension,
                    norm=0.0,
                    custom_metadata=meta_dict
                )
                
                return (vector, metadata) if vector is not None else None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            return None
    
    def delete(self, vector_id: str) -> bool:
        """Delete vector from ChromaDB"""
        try:
            self.collection.delete(ids=[vector_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False
    
    def exists(self, vector_id: str) -> bool:
        """Check if vector exists"""
        try:
            result = self.collection.get(ids=[vector_id])
            return len(result['ids']) > 0
        except:
            return False
    
    def count(self) -> int:
        """Get total number of vectors"""
        return self.collection.count()
    
    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.metric.value}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False
    
    def batch_add(self, vectors: List[Tuple[str, np.ndarray, VectorMetadata]]) -> int:
        """Add multiple vectors"""
        try:
            ids = []
            embeddings = []
            metadatas = []
            
            for vector_id, vector, metadata in vectors:
                ids.append(vector_id)
                embeddings.append(vector.tolist())
                
                # Convert metadata
                meta_dict = {}
                if metadata:
                    meta_dict = {
                        'chunk_id': metadata.chunk_id,
                        'file_path': metadata.file_path,
                        'language': metadata.language or '',
                        'chunk_type': metadata.chunk_type or '',
                    }
                    meta_dict.update(metadata.custom_metadata)
                
                metadatas.append(meta_dict)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas if metadatas else None
            )
            
            return len(ids)
            
        except Exception as e:
            logger.error(f"Failed to batch add vectors: {e}")
            return 0

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    @staticmethod
    def create(config: VectorConfig) -> BaseVectorStore:
        """Create vector store based on backend"""
        if config.backend == VectorBackend.FAISS:
            return FAISSVectorStore(config)
        elif config.backend == VectorBackend.HNSWLIB:
            return HNSWLibVectorStore(config)
        elif config.backend == VectorBackend.CHROMADB:
            return ChromaDBVectorStore(config)
        # Add other backends as needed
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

# Convenience functions
def create_vector_store(backend: str = "faiss",
                       dimension: int = 768,
                       metric: str = "cosine",
                       **kwargs) -> BaseVectorStore:
    """Create vector store with common settings"""
    config = VectorConfig(
        backend=VectorBackend(backend),
        dimension=dimension,
        metric=DistanceMetric(metric),
        **kwargs
    )
    return VectorStoreFactory.create(config)

def store_vector(vector_id: str, vector: np.ndarray,
                metadata: Optional[Dict[str, Any]] = None,
                backend: str = "faiss") -> bool:
    """Quick function to store a vector"""
    store = create_vector_store(backend=backend, dimension=len(vector))
    
    # Create metadata
    meta = VectorMetadata(
        vector_id=vector_id,
        chunk_id=vector_id,
        file_path="",
        dimension=len(vector),
        norm=float(np.linalg.norm(vector)),
        custom_metadata=metadata or {}
    )
    
    return store.add(vector_id, vector, meta)

def search_vectors(query_vector: np.ndarray, k: int = 10,
                  backend: str = "faiss") -> List[SearchResult]:
    """Quick function to search vectors"""
    store = create_vector_store(backend=backend, dimension=len(query_vector))
    return store.search(query_vector, k)