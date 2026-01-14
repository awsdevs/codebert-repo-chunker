"""
src/storage/storage_manager.py
Unified Facade for Pipeline.
"""
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import logging

from src.storage.chunk_storage import ChunkStorage, StorageConfig
from src.storage.metadata_store import MetadataStore, MetadataConfig
from src.storage.vector_store import VectorStore, VectorConfig

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, workspace: Path):
        self.chunk_store = ChunkStorage(StorageConfig(workspace / "chunks"))
        self.meta_store = MetadataStore(MetadataConfig(workspace / "metadata"))
        # Default to Flat for reliability. Change to "IVF" and set ivf_nlist for scale.
        self.vector_store = VectorStore(VectorConfig(workspace / "vectors", index_type="Flat"))

    def batch_store(self, chunk_ids: List[str], contents: List[str], 
                   metadatas: List[Dict], embeddings: List[np.ndarray]):
        """Atomic-like batch storage of all components"""
        
        if not (len(chunk_ids) == len(contents) == len(metadatas) == len(embeddings)):
            logger.error("Length mismatch in batch store")
            return

        # 1. Content
        for cid, content, meta in zip(chunk_ids, contents, metadatas):
            self.chunk_store.store(
                chunk_id=cid, 
                content=content, 
                file_path=meta.get('file_path', 'unknown'),
                language=meta.get('language', 'unknown')
            )

        # 2. Metadata
        for cid, meta in zip(chunk_ids, metadatas):
            self.meta_store.store(cid, meta)

        # 3. Vectors (Stacked)
        self.vector_store.add(chunk_ids, np.vstack(embeddings))
        
    def close(self):
        self.chunk_store.close()
        self.meta_store.close()
        self.vector_store.close()