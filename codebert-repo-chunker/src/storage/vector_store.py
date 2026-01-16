
"""
src/storage/vector_store.py
Production Vector Store using FAISS.
Handles both Exact (Flat) and Approximate (IVF) indexing logic.
"""
from src.utils.logger import get_logger
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

logger = get_logger(__name__)

# Import FAISS/Numpy locally to avoid crashing if not installed, 
import faiss
import numpy as np

@dataclass
class VectorConfig:
    storage_path: Path = Path("storage/vectors")
    dimension: int = 768
    # Options: "Flat" (Exact, default) or "IVF" (Approximate, requires training)
    index_type: str = "Flat" 
    ivf_nlist: int = 100 # Clusters for IVF. Rule of thumb: 4 * sqrt(N)

    def __post_init__(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)

class VectorStore:
    def __init__(self, config: VectorConfig):
        self.config = config
        self.index_path = config.storage_path / "faiss_index.bin"
        self.map_path = config.storage_path / "id_map.pkl"
        
        self.index = None
        self.id_map: Dict[int, str] = {} # FAISS int ID -> Chunk String ID
        self.reverse_map: Dict[str, int] = {} # Chunk String ID -> FAISS int ID
        self.next_id = 0 # Explicit ID counter
        
        if faiss and np:
            self._load_or_create()
        else:
            logger.warning("FAISS or Numpy not found. Vector store disabled.")

    def _load_or_create(self):
        if not faiss: return
        
        if self.index_path.exists() and self.map_path.exists():
            logger.info("Loading existing vector index...")
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.map_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        # Backward compatibility
                        self.id_map = data
                        self.next_id = max(self.id_map.keys()) + 1 if self.id_map else 0
                    elif isinstance(data, tuple) and len(data) == 2:
                        # New format: (id_map, next_id)
                        self.id_map, self.next_id = data
                    else:
                        self.id_map = {}
                        self.next_id = 0
                
                # Rebuild reverse map for O(1) lookups
                self.reverse_map = {v: k for k, v in self.id_map.items()}
                        
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        logger.info(f"Creating new {self.config.index_type} index (Dim: {self.config.dimension})")
        
        if self.config.index_type == "Flat":
            # Start: CodeBERT Fix - Use Inner Product for Cosine Similarity
            # Wrap with IndexIDMap to support deletion/explicit IDs
            base_index = faiss.IndexFlatIP(self.config.dimension)
            self.index = faiss.IndexIDMap(base_index)
            # End: CodeBERT Fix
        
        elif self.config.index_type == "IVF":
            # IVF requires a quantizer (Flat) and training
            quantizer = faiss.IndexFlatIP(self.config.dimension)
            # IVF automatically supports removal if nlist is small, but standard practice ensures IDMap
            ivf_index = faiss.IndexIVFFlat(quantizer, self.config.dimension, self.config.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.IndexIDMap(ivf_index)
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
            
        self.id_map = {}
        self.reverse_map = {}
        self.next_id = 0

    def _get_underlying_index(self):
        """
        Get the underlying index for operations like training.
        IndexIDMap wraps the real index, so we need to access .index for training.
        """
        if hasattr(self.index, 'index'):
            return self.index.index
        return self.index

    def _ensure_trained(self, embeddings: Any):
        """
        Train IVF index if it hasn't been trained yet.
        Must access underlying index since IndexIDMap wrapper doesn't expose training.
        """
        if not faiss or not np: return
        
        underlying = self._get_underlying_index()
        
        if self.config.index_type == "IVF" and hasattr(underlying, 'is_trained') and not underlying.is_trained:
            logger.info(f"Training IVF Index with {len(embeddings)} vectors...")
            # Ideally we need ~30 * nlist vectors to train effectively
            min_train = self.config.ivf_nlist * 30
            if len(embeddings) < min_train:
                logger.warning(f"Training set {len(embeddings)} is small for nlist={self.config.ivf_nlist}. Accuracy may suffer.")
            
            # Normalize for Cosine Similarity training
            faiss.normalize_L2(embeddings)
            underlying.train(embeddings.astype('float32'))
            logger.info("Index training complete.")

    def add(self, chunk_ids: List[str], embeddings: Any):
        """Add vectors to index, handling training automatically."""
        if not faiss or not np or not self.index: return

        embeddings = embeddings.astype('float32')
        # Normalize vectors for Cosine Similarity (Dot Product of normalized vectors)
        faiss.normalize_L2(embeddings)
        
        # 1. Check Training requirement
        underlying = self._get_underlying_index()
        if hasattr(underlying, 'is_trained') and not underlying.is_trained:
            self._ensure_trained(embeddings)
        
        # 2. Add to Index with explicit IDs
        n = len(chunk_ids)
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        
        try:
            self.index.add_with_ids(embeddings, ids)
        except Exception as e:
            # Fallback for indices that might not support add_with_ids (e.g. some dense-only wrappers)
            # But IndexIDMap and IndexIVF SHOULD support it.
            logger.error(f"Failed to add with IDs: {e}. Trying standard add (Warning: IDs might desync in deletion).")
            self.index.add(embeddings)
        
        # 3. Update ID Map
        for i, chunk_id in enumerate(chunk_ids):
            fid = int(ids[i])
            self.id_map[fid] = chunk_id
            self.reverse_map[chunk_id] = fid
            
        self.next_id += n
            
        logger.info(f"Indexed {len(chunk_ids)} vectors. Total: {self.index.ntotal}")

    def remove(self, chunk_ids: List[str]):
        """Remove vectors by chunk ID"""
        if not faiss or not self.index: return

        # O(1) Reverse lookup using map
        faiss_ids_to_remove = []
        for cid in chunk_ids:
            if cid in self.reverse_map:
                faiss_ids_to_remove.append(self.reverse_map[cid])
        
        if not faiss_ids_to_remove:
            return

        try:
            ids_array = np.array(faiss_ids_to_remove, dtype=np.int64)
            n_removed = self.index.remove_ids(ids_array)
            if n_removed > 0:
                # Clean up maps
                for fid in faiss_ids_to_remove:
                    if fid in self.id_map:
                        cid = self.id_map[fid]
                        del self.id_map[fid]
                        if cid in self.reverse_map:
                            del self.reverse_map[cid]
                            
                logger.info(f"Removed {n_removed} vectors from index.")
            else:
                logger.warning("remove_ids called but FAISS reported 0 removals.")
        except Exception as e:
            logger.error(f"Failed to remove vectors: {e}")

    def search(self, query: Any, k: int = 10) -> List[Tuple[str, float]]:
        if not faiss or not np or not self.index: return []
        if self.index.ntotal == 0:
            return []
            
        query_vec = query.reshape(1, -1).astype('float32')
        # Normalize query vector for Cosine Similarity
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))
        
        return results

    def save(self):
        if not faiss or not self.index: return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.map_path, 'wb') as f:
            # Save tuple of (id_map, next_id)
            pickle.dump((self.id_map, self.next_id), f, protocol=4)

    def close(self):
        self.save()