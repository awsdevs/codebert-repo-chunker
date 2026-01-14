"""
src/storage/metadata_store.py
Metadata Storage with Full Text Search (FTS).
"""
import sqlite3
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

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
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT,
                repository TEXT,
                json_data TEXT,
                updated_at TIMESTAMP
            )
        """)
        # FTS5 for rich text searching over descriptions/code logic
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS search_index 
            USING fts5(chunk_id UNINDEXED, rich_text)
        """)
        self.conn.commit()

    def store(self, chunk_id: str, metadata: Dict[str, Any]):
        try:
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
        except Exception as e:
            logger.error(f"Metadata store error {chunk_id}: {e}")

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT json_data FROM metadata WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.conn.close()