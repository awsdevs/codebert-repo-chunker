
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
import hashlib
import uuid
import json
from enum import Enum

import numpy as np

@dataclass
class ChunkLocation:
    """Location of a code chunk in a file"""
    file_path: str
    start_line: int
    end_line: int
    file_checksum: str = ""

class ChunkType(str, Enum):
    """Type of code chunk"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    
    # Documentation
    DOCUMENTATION = "documentation"
    DOCSTRING = "docstring"
    COMMENT = "comment"
    README = "readme"
    
    # Config/Data
    CONFIG = "config"
    SCHEMA = "schema"
    METADATA = "metadata"
    DATA = "data"
    QUERY = "query"
    
    # Tests
    TEST_CLASS = "test_class"
    TEST_FUNCTION = "test_function"
    
    # Generic
    BLOCK = "block"
    SECTION = "section"
    FRAGMENT = "fragment"
    OTHER = "other"
    UNKNOWN = "unknown"

@dataclass
class Chunk:
    """Represents a discrete unit of code for embedding"""
    id: str  # Unique ID (hash)
    content: str
    location: ChunkLocation
    chunk_type: ChunkType
    
    # Metadata
    language: str = "unknown"
    parent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list) # IDs of chunks this depends on
    metadata: Dict[str, Any] = field(default_factory=dict) # Key-value metadata
    
    # Embeddings (optional until processed)
    # Using Any for embedding to support both numpy arrays and lists (for JSON serialization)
    embedding: Optional[Any] = None 
    
    # Text statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "id": self.id,
            "content": self.content,
            "location": {
                "file_path": self.location.file_path,
                "start_line": self.location.start_line,
                "end_line": self.location.end_line,
                "file_checksum": self.location.file_checksum
            },
            "chunk_type": self.chunk_type.value,
            "language": self.language,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "stats": self.stats,
            "created_at": self.created_at.isoformat()
        }
        
        if self.embedding is not None:
            if np is not None and isinstance(self.embedding, np.ndarray):
                data["embedding"] = self.embedding.tolist()
            else:
                data["embedding"] = self.embedding
                
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create Chunk from dictionary"""
        loc_data = data["location"]
        location = ChunkLocation(
            file_path=loc_data["file_path"],
            start_line=loc_data["start_line"],
            end_line=loc_data["end_line"],
            file_checksum=loc_data.get("file_checksum", "")
        )
        
        chunk = cls(
            id=data["id"],
            content=data["content"],
            location=location,
            chunk_type=ChunkType(data["chunk_type"]),
            language=data.get("language", "unknown"),
            parent_id=data.get("parent_id"),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
            stats=data.get("stats", {})
        )
        
        if "created_at" in data:
            chunk.created_at = datetime.fromisoformat(data["created_at"])
            
        if "embedding" in data:
            chunk.embedding = data["embedding"]
            # Convert list back to numpy if available
            if np is not None and isinstance(chunk.embedding, list):
                chunk.embedding = np.array(chunk.embedding, dtype=np.float32)
                
        return chunk