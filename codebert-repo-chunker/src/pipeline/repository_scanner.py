
from pathlib import Path
from typing import List, Set, Iterator, Tuple, Optional, Union
from dataclasses import dataclass, field
from src.utils.logger import get_logger
import os

try:
    import pathspec
except ImportError:
    pathspec = None

from src.classifiers.file_classifier import FileClassifier, FileClassification, FileCategory

logger = get_logger(__name__)

@dataclass
class ScannerConfig:
    """Configuration for repository scanning"""
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        ".git", ".svn", ".hg", ".idea", ".vscode", 
        "__pycache__", "node_modules", "venv", ".env",
        "dist", "build", "target", "out", "bin",
        ".DS_Store", "Thumbs.db"
    ])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    follow_symlinks: bool = False
    respect_gitignore: bool = True
    
@dataclass
class ScannedFile:
    """Represents a discovered file with basic metadata"""
    path: Path
    relative_path: Path
    size: int
    classification: Optional[FileClassification] = None

class RepositoryScanner:
    """
    Scans repository for files to process, respecting ignore rules
    and handling file classification.
    """
    
    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
        self.classifier = FileClassifier()
        self._ignore_spec = None
        
    def scan(self, root_path: Union[str, Path]) -> Iterator[ScannedFile]:
        """
        Scan repository and yield discovered files
        
        Args:
            root_path: Root path of repository
            
        Yields:
            ScannedFile objects
        """
        root = Path(root_path).resolve()
        if not root.exists():
            logger.error(f"Root path does not exist: {root}")
            return
            
        # Initialize ignore patterns (gitignore + config)
        self._init_ignore_spec(root)
        
        logger.info(f"Scanning repository at {root}")
        count = 0
        
        for root_dir, dirs, files in os.walk(root, followlinks=self.config.follow_symlinks):
            root_dir_path = Path(root_dir)
            try:
                relative_root = root_dir_path.relative_to(root)
            except ValueError:
                relative_root = Path(".")
            
            # Filter directories in-place to prevent traversal
            valid_dirs = []
            for d in dirs:
                dir_rel_path = relative_root / d
                if not self._is_ignored(dir_rel_path, is_dir=True):
                    valid_dirs.append(d)
            
            dirs[:] = valid_dirs
            
            # Process files
            for f in files:
                file_path = root_dir_path / f
                relative_path = relative_root / f
                
                # Check ignores
                if self._is_ignored(relative_path):
                    continue
                    
                # Check file size
                try:
                    if not file_path.exists():
                        continue
                        
                    size = file_path.stat().st_size
                    if size > self.config.max_file_size:
                        logger.debug(f"Skipping large file: {relative_path} ({size} bytes)")
                        continue
                        
                    # Classify file
                    try:
                        classification = self.classifier.classify(file_path)
                    except Exception as cls_err:
                        logger.warning(f"Classification failed for {file_path}: {cls_err}")
                        classification = None
                    
                    yield ScannedFile(
                        path=file_path,
                        relative_path=relative_path,
                        size=size,
                        classification=classification
                    )
                    count += 1
                    
                except Exception as e:
                    logger.warning(f"Error scanning file {file_path}: {e}")
                    continue
                    
        logger.info(f"Scan complete. Found {count} files.")

    def _init_ignore_spec(self, root: Path):
        """Initialize pathspec for ignore patterns"""
        patterns = set(self.config.exclude_patterns)
        
        if self.config.respect_gitignore:
            gitignore_path = root / ".gitignore"
            if gitignore_path.exists():
                try:
                    with open(gitignore_path, "r") as f:
                        git_patterns = [
                            line.strip() for line in f 
                            if line.strip() and not line.startswith('#')
                        ]
                        patterns.update(git_patterns)
                    logger.debug(f"Loaded {len(git_patterns)} patterns from .gitignore")
                except Exception as e:
                    logger.warning(f"Failed to load .gitignore: {e}")
        
        if pathspec:
            try:
                # Use gitwildmatch for .gitignore compatibility
                self._ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            except Exception:
                self._ignore_spec = None
                self._fallback_patterns = list(patterns)
        else:
            logger.warning("pathspec library not found. Using glob matching.")
            self._ignore_spec = None
            self._fallback_patterns = list(patterns)

    def _is_ignored(self, relative_path: Path, is_dir: bool = False) -> bool:
        """Check if path matches ignore patterns"""
        path_str = str(relative_path)
        if is_dir:
            path_str += "/"
            
        if self._ignore_spec:
            return self._ignore_spec.match_file(path_str)
        
        # Fallback implementation
        import fnmatch
        name = relative_path.name
        
        for pattern in self._fallback_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(path_str, pattern):
                return True
            # Check directory parts for recursive exclusion
            for part in relative_path.parts:
                if fnmatch.fnmatch(part, pattern) and part != ".":
                     if "/" not in pattern: # Simple name pattern matches any part
                         return True
        return False
