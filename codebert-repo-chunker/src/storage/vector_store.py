
"""
src/storage/vector_store.py
Production Vector Store using FAISS.
Handles both Exact (Flat) and Approximate (IVF) indexing logic.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
                    self.id_map = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        logger.info(f"Creating new {self.config.index_type} index (Dim: {self.config.dimension})")
        
        if self.config.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.config.dimension)
        
        elif self.config.index_type == "IVF":
            # IVF requires a quantizer (Flat) and training
            quantizer = faiss.IndexFlatL2(self.config.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.config.dimension, self.config.ivf_nlist)
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
            
        self.id_map = {}

    def _ensure_trained(self, embeddings: Any):
        """
        GAP FIX: Logic to train IVF index if it hasn't been trained yet.
        """
        if not faiss or not np: return
        
        if self.config.index_type == "IVF" and not self.index.is_trained:
            logger.info(f"Training IVF Index with {len(embeddings)} vectors...")
            # Ideally we need ~30 * nlist vectors to train effectively
            min_train = self.config.ivf_nlist * 30
            if len(embeddings) < min_train:
                logger.warning(f"Training set {len(embeddings)} is small for nlist={self.config.ivf_nlist}. Accuracy may suffer.")
            
            self.index.train(embeddings.astype('float32'))
            logger.info("Index training complete.")

    def add(self, chunk_ids: List[str], embeddings: Any):
        """Add vectors to index, handling training automatically."""
        if not faiss or not np or not self.index: return

        embeddings = embeddings.astype('float32')
        
        # 1. Check Training requirement
        if not self.index.is_trained:
            self._ensure_trained(embeddings)
        
        # 2. Add to Index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # 3. Update ID Map
        for i, chunk_id in enumerate(chunk_ids):
            self.id_map[start_id + i] = chunk_id
            
        logger.info(f"Indexed {len(chunk_ids)} vectors. Total: {self.index.ntotal}")

    def search(self, query: Any, k: int = 10) -> List[Tuple[str, float]]:
        if not faiss or not np or not self.index: return []
        if self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(query.reshape(1, -1).astype('float32'), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))
        
        return results

    def save(self):
        if not faiss or not self.index: return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.map_path, 'wb') as f:
            pickle.dump(self.id_map, f)

    def close(self):
        self.save()