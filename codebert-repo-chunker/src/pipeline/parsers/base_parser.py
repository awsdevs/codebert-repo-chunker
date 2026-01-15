from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Dependency:
    name: str              # e.g., "requests"
    version: str           # e.g., ">=2.25.0"
    type: str              # "runtime", "dev", "peer", "base"
    source_file: str       # "requirements.txt"
    line_number: int  = 0  # 1-indexed line number
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "type": self.type,
            "source_file": self.source_file,
            "line_number": self.line_number
        }

class BaseManifestParser(ABC):
    """Abstract base class for all dependency manifest parsers"""
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Return True if this parser can handle the given file"""
        pass
    
    @abstractmethod
    def parse(self, file_path: Path, content: str) -> List[Dependency]:
        """Parse the content and return a list of dependencies"""
        pass
