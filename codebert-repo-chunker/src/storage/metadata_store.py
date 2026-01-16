"""
src/storage/metadata_store.py
Metadata Storage with Full Text Search (FTS).
"""
import sqlite3
import json
from src.utils.logger import get_logger
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

logger = get_logger(__name__)

@dataclass
class MetadataConfig:
    storage_path: Path = Path("storage/metadata")

    def __post_init__(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)

class MetadataStore:
    def __init__(self, config: MetadataConfig):
        self.config = config
        self.db_path = config.storage_path / "metadata.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._setup_db()

    def _setup_db(self):
        self.conn.execute("PRAGMA journal_mode=WAL")
        
        schema_sql = """
            CREATE TABLE IF NOT EXISTS metadata (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT,
                repository TEXT,
                json_data TEXT,
                updated_at TIMESTAMP
            )
        """
        self.conn.execute(schema_sql)
        logger.info(f"Initialized Metadata Schema: {schema_sql.strip()}")
        
        # FTS5 for rich text searching over descriptions/code logic
        fts_sql = """
            CREATE VIRTUAL TABLE IF NOT EXISTS search_index 
            USING fts5(chunk_id UNINDEXED, rich_text)
        """
        self.conn.execute(fts_sql)
        logger.info(f"Initialized FTS Schema: {fts_sql.strip()}")
        
        self.conn.commit()

    def store(self, chunk_id: str, metadata: Dict[str, Any]):
        """Store metadata for a chunk"""
        try:
            # Enforce removal of heavy fields to prevent redundancy
            # This catches cases where callers bypass StorageManager
            if isinstance(metadata, dict):
                # Don't modify original dict in case caller needs it
                metadata = metadata.copy()
                metadata.pop('content', None)
                metadata.pop('embedding', None)

            # Store as JSON
            json_str = json.dumps(metadata, default=str)
            file_path = metadata.get('file_path', '')
            repo = metadata.get('repository', '')
            
            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO metadata 
                    (chunk_id, file_path, repository, json_data, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (chunk_id, file_path, repo, json_str))

                # Update FTS: Concatenate searchable fields
                # Delete existing to prevent duplicates in FTS
                self.conn.execute("DELETE FROM search_index WHERE chunk_id = ?", (chunk_id,))
                
                rich_text = f"{file_path} {repo} {metadata.get('docstring', '')} {metadata.get('function_name', '')}"
                self.conn.execute("INSERT INTO search_index (chunk_id, rich_text) VALUES (?, ?)", 
                                (chunk_id, rich_text))
            
            logger.debug(f"Stored metadata for {chunk_id}")
        except Exception as e:
            logger.error(f"Metadata store error {chunk_id}: {e}")

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT json_data FROM metadata WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def list_by_file(self, file_path: str) -> List[str]:
        """Get all chunk_ids for a specific file"""
        cursor = self.conn.execute("SELECT chunk_id FROM metadata WHERE file_path = ?", (file_path,))
        return [row[0] for row in cursor.fetchall()]

    def list_by_repository(self, repository: str) -> List[str]:
        """Get all chunk_ids for a repository"""
        cursor = self.conn.execute("SELECT chunk_id FROM metadata WHERE repository = ?", (repository,))
        return [row[0] for row in cursor.fetchall()]

    def get_file_checksums(self, repository: str) -> Dict[str, str]:
        """Get {file_path: checksum} for all files in a repository"""
        # Note: checksum needs to be extracted from json_data or redundant column.
        # Ideally we should elevate checksum to a column, but for now scan JSON.
        # Optimization: Add file_checksum column in future schema migration.
        # Current workaround: Parse JSON (slow but works without schema change yet)
        cursor = self.conn.execute("SELECT file_path, json_data FROM metadata WHERE repository = ?", (repository,))
        checksums = {}
        for row in cursor.fetchall():
            path = row[0]
            try:
                data = json.loads(row[1])
                # Checksum might be in location.file_checksum or flattened
                loc = data.get('location', {})
                csum = loc.get('file_checksum') or data.get('file_checksum')
                if csum:
                    checksums[path] = csum
            except:
                pass
        return checksums

    def search_text(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """
        Full-text search returning (chunk_id, rank) tuples.
        Uses FTS5 BM25 ranking.
        """
        try:
            cursor = self.conn.execute("""
                SELECT chunk_id, bm25(search_index) as rank
                FROM search_index 
                WHERE rich_text MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"FTS search error: {e}")
            return []

    def delete(self, chunk_id: str) -> bool:
        """Delete metadata and FTS entry"""
        try:
            with self.conn:
                self.conn.execute("DELETE FROM metadata WHERE chunk_id = ?", (chunk_id,))
                self.conn.execute("DELETE FROM search_index WHERE chunk_id = ?", (chunk_id,))
            return True
        except Exception as e:
            logger.error(f"Delete error for {chunk_id}: {e}")
            return False

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for a file path. Returns count deleted."""
        chunk_ids = self.list_by_file(file_path)
        count = 0
        for cid in chunk_ids:
            if self.delete(cid):
                count += 1
        return count

    def close(self):
        self.conn.close()