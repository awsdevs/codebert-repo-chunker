"""
src/storage/chunk_storage.py
Robust Content Storage for Code Chunks.
"""
import sqlite3
from src.utils.logger import get_logger
import zlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

logger = get_logger(__name__)

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
        
        schema_sql = """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                content BLOB NOT NULL,
                file_path TEXT, 
                language TEXT,
                size_bytes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.conn.execute(schema_sql)
        logger.info(f"Initialized Chunk Schema: {schema_sql.strip()}")
        
        # Index for faster recovery/debugging lookups
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_filepath ON chunks(file_path)")
        self.conn.commit()

    def store(self, chunk_id: str, content: str, file_path: str, language: str) -> bool:
        try:
            file_path = self._normalize_path(file_path)
            compressed = zlib.compress(content.encode('utf-8'), self.config.compression_level)
            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, content, file_path, language, size_bytes)
                    VALUES (?, ?, ?, ?, ?)
                """, (chunk_id, compressed, file_path, language, len(content)))
            logger.debug(f"Stored content for {chunk_id} ({len(content)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Chunk store error for {chunk_id}: {e}")
            return False

    def store_batch(self, chunks: List[Tuple[str, str, str, str]]) -> int:
        """
        Batch store chunks.
        Args: chunks: List of (chunk_id, content, file_path, language)
        Returns: count stored
        """
        if not chunks: return 0
        try:
            batch = [
                (cid, zlib.compress(c.encode('utf-8'), self.config.compression_level), 
                 self._normalize_path(fp), lang, len(c))
                for cid, c, fp, lang in chunks
            ]
            with self.conn:
                self.conn.executemany("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, content, file_path, language, size_bytes)
                    VALUES (?, ?, ?, ?, ?)
                """, batch)
            return len(batch)
        except Exception as e:
            logger.error(f"Batch store error: {e}")
            return 0

    def retrieve(self, chunk_id: str) -> Optional[str]:
        cursor = self.conn.execute("SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            try:
                return zlib.decompress(row[0]).decode('utf-8')
            except Exception as e:
                logger.error(f"Decompression corrupted for {chunk_id}: {e}")
        return None

    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk by ID"""
        try:
            with self.conn:
                self.conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for a file. Returns count."""
        file_path = self._normalize_path(file_path)
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_path = ?", (file_path,)
        )
        count = cursor.fetchone()[0]
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        return count

    def _normalize_path(self, path: str) -> str:
        """Normalize path separators for cross-platform consistency"""
        return str(path).replace('\\', '/')

    def close(self):
        self.conn.close()