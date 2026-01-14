"""
src/storage/chunk_storage.py
Robust Content Storage for Code Chunks.
"""
import sqlite3
import logging
import zlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    storage_path: Path = Path("storage/chunks")
    compression_level: int = 6

    def __post_init__(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)

class ChunkStorage:
    def __init__(self, config: StorageConfig):
        self.config = config
        self.db_path = config.storage_path / "content.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._setup_db()

    def _setup_db(self):
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                content BLOB NOT NULL,
                file_path TEXT, 
                language TEXT,
                size_bytes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Index for faster recovery/debugging lookups
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_filepath ON chunks(file_path)")
        self.conn.commit()

    def store(self, chunk_id: str, content: str, file_path: str, language: str) -> bool:
        try:
            compressed = zlib.compress(content.encode('utf-8'), self.config.compression_level)
            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, content, file_path, language, size_bytes)
                    VALUES (?, ?, ?, ?, ?)
                """, (chunk_id, compressed, file_path, language, len(content)))
            return True
        except Exception as e:
            logger.error(f"Chunk store error for {chunk_id}: {e}")
            return False

    def retrieve(self, chunk_id: str) -> Optional[str]:
        cursor = self.conn.execute("SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            try:
                return zlib.decompress(row[0]).decode('utf-8')
            except Exception as e:
                logger.error(f"Decompression corrupted for {chunk_id}: {e}")
        return None

    def close(self):
        self.conn.close()