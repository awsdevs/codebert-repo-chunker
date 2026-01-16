"""
Base chunker class providing core functionality for all specialized chunkers
Defines the interface and common utilities for semantic code chunking
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from enum import Enum
from src.utils.logger import get_logger
import re
from collections import defaultdict
import time

from src.core.chunk_model import Chunk, ChunkLocation, ChunkType

logger = get_logger(__name__)

# Removed Duplicate ChunkType, ChunkMetadata, ChunkRelation, Chunk classes
# We use src.core.chunk_model.Chunk
# Since ChunkType in chunk_model might determine types, we need to ensure compatibility.
# For now, we reuse strings if needed, or rely on ChunkType from chunk_model.

# If we need ChunkRelationType, we should import or define it. 
# BaseChunker used ChunkRelationType(Enum).
class ChunkRelationType(Enum):
    """Types of relationships between chunks"""
    PARENT = "parent"
    CHILD = "child"
    DEPENDS_ON = "depends_on"
    CALLED_BY = "called_by"
    CALLS = "calls"
    IMPLEMENTS = "implements"

@dataclass
class ChunkerConfig:
    """Configuration for chunkers"""
    max_tokens: int = 450
    min_tokens: int = 50
    overlap_tokens: int = 50
    max_chunk_size: int = 4000  # characters
    min_chunk_size: int = 100
    preserve_structure: bool = True
    include_comments: bool = True
    include_docstrings: bool = True
    include_imports: bool = True
    track_dependencies: bool = True
    track_relationships: bool = True
    extract_metadata: bool = True
    calculate_complexity: bool = True
    detect_patterns: bool = False
    language_specific: bool = True
    adaptive_sizing: bool = True
    quality_threshold: float = 0.7
    timeout_seconds: int = 30
    custom_options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FileContext:
    """Context information about the file being chunked"""
    path: Path
    content: str
    language: Optional[str] = None
    encoding: str = 'utf-8'
    size: Optional[int] = None
    lines: Optional[int] = None
    mime_type: Optional[str] = None
    is_test: bool = False
    is_generated: bool = False
    framework: Optional[str] = None
    module_name: Optional[str] = None
    package_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseChunker(ABC):
    """
    Abstract base class for all chunkers
    Provides common functionality and defines the interface
    """
    
    def __init__(self, tokenizer=None, config: Optional[ChunkerConfig] = None):
        """
        Initialize base chunker
        
        Args:
            tokenizer: Optional tokenizer for counting tokens
            config: Chunker configuration
        """
        self.tokenizer = tokenizer
        self.config = config or ChunkerConfig()
        self._chunk_counter = 0
        self._chunks_created = []
        self._relationships = []
        self._stats = defaultdict(int)
        self._start_time = None
        
        # Token counting function
        self.count_tokens = self._create_token_counter()
        
        # Initialize language-specific settings
        self._init_language_settings()
    
    @property
    def max_tokens(self) -> int:
        """Get max tokens from config"""
        return self.config.max_tokens
    
    def _create_token_counter(self) -> Callable[[str], int]:
        """Create token counting function"""
        if self.tokenizer:
            def count_with_tokenizer(text: str) -> int:
                try:
                    return len(self.tokenizer.encode(text))
                except Exception as e:
                    logger.warning(f"Tokenizer failed, falling back to estimation: {e}")
                    return self._estimate_tokens(text)
        else:
            def count_with_estimation(text: str) -> int:
                return self._estimate_tokens(text)
            return count_with_estimation
        
        return count_with_tokenizer
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count when tokenizer is not available
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Common estimation: ~1 token per 4 characters or ~0.75 tokens per word
        word_count = len(text.split())
        char_count = len(text)
        
        # Use average of both methods
        word_estimate = word_count * 0.75
        char_estimate = char_count / 4
        
        return int((word_estimate + char_estimate) / 2)
    
    def _init_language_settings(self):
        """Initialize language-specific settings"""
        self.comment_patterns = {
            'python': (r'#.*$', r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"),
            'javascript': (r'//.*$', r'/\*[\s\S]*?\*/'),
            'java': (r'//.*$', r'/\*[\s\S]*?\*/'),
            'cpp': (r'//.*$', r'/\*[\s\S]*?\*/'),
            'go': (r'//.*$', r'/\*[\s\S]*?\*/'),
            'ruby': (r'#.*$', r'=begin[\s\S]*?=end'),
            'sql': (r'--.*$', r'/\*[\s\S]*?\*/'),
        }
        
        self.docstring_patterns = {
            'python': (r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"),
            'javascript': (r'/\*\*[\s\S]*?\*/',),
            'java': (r'/\*\*[\s\S]*?\*/',),
        }
    
    @abstractmethod
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from content
        
        Args:
            content: Content to chunk
            file_context: Context about the file
            
        Returns:
            List of chunks
        """
        pass
    
    def create_chunk(self, 
                    content: str,
                    chunk_type: Union[ChunkType, str],
                    start_line: int = 0,
                    end_line: int = 0,
                    metadata: Optional[Dict[str, Any]] = None,
                    parent_id: Optional[str] = None,
                    file_path: Optional[str] = None) -> Chunk:
        """
        Create a chunk with automatic ID generation and metadata
        
        Args:
            content: Chunk content
            chunk_type: Type of chunk
            start_line: Starting line number
            end_line: Ending line number
            metadata: Optional metadata
            parent_id: Optional parent chunk ID
            file_path: Optional file path
            
        Returns:
            Created chunk
        """
        # Convert string chunk_type to enum if needed
        if isinstance(chunk_type, str):
            try:
                chunk_type = ChunkType[chunk_type.upper()]
            except KeyError:
                chunk_type = ChunkType.UNKNOWN
        
        # Generate unique chunk ID
        # Generate unique chunk ID
        self._chunk_counter += 1
        chunk_id = f"{file_path or 'unknown'}_{chunk_type.value}_{self._chunk_counter}"
        
        # Create metadata dictionary
        # Removed ChunkMetadata class usage, using simple dict as per new ChunkModel
        chunk_metadata = {}
        if metadata:
            chunk_metadata.update(metadata)
        
        # Count tokens
        token_count = self.count_tokens(content)
        
        # Create location
        location = ChunkLocation(
            file_path=file_path or '',
            start_line=start_line,
            end_line=end_line if end_line > 0 else start_line + content.count('\n'),
            file_checksum=""
        )

        # Create chunk
        chunk = Chunk(
            id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            location=location,
            parent_id=parent_id,
            metadata=chunk_metadata,
            stats={'token_count': token_count}
        )
        
        # Set warnings in metadata if needed
        chunk.metadata.setdefault('warnings', [])
        
        # Track chunk
        self._chunks_created.append(chunk)
        self._stats['chunks_created'] += 1
        # self._stats[f'chunks_{chunk_type.value}'] += 1 # Stats dict keys might need init
        self._stats['total_tokens'] += token_count
        
        # Check size constraints
        if token_count > self.config.max_tokens:
            chunk.metadata['warnings'].append(f"Chunk exceeds max tokens: {token_count} > {self.config.max_tokens}")
        elif token_count < self.config.min_tokens:
            chunk.metadata['warnings'].append(f"Chunk below min tokens: {token_count} < {self.config.min_tokens}")
        
        return chunk
    
    def add_relationship(self,
                        source_chunk: Chunk,
                        target_chunk: Chunk,
                        relation_type: ChunkRelationType,
                        strength: float = 1.0,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Add a relationship between chunks
        
        Args:
            source_chunk: Source chunk
            target_chunk: Target chunk
            relation_type: Type of relationship
            strength: Relationship strength (0-1)
            metadata: Optional relationship metadata
        """
        relation = ChunkRelation(
            source_chunk_id=source_chunk.id,
            target_chunk_id=target_chunk.id,
            relation_type=relation_type,
            strength=strength,
            metadata=metadata or {}
        )
        
        source_chunk.relations.append(relation)
        self._relationships.append(relation)
        self._stats['relationships_created'] += 1
        
        # Update chunk relationships
        if relation_type in [ChunkRelationType.PARENT, ChunkRelationType.CHILD]:
            if relation_type == ChunkRelationType.PARENT:
                target_chunk.parent_id = source_chunk.id
                source_chunk.children_ids.append(target_chunk.id)
            else:
                source_chunk.parent_id = target_chunk.id
                target_chunk.children_ids.append(source_chunk.id)
    
    def split_by_tokens(self, 
                       content: str,
                       max_tokens: Optional[int] = None,
                       overlap: Optional[int] = None,
                       preserve_lines: bool = True) -> List[str]:
        """
        Split content into chunks based on token count
        
        Args:
            content: Content to split
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks
            preserve_lines: Whether to preserve line boundaries
            
        Returns:
            List of content chunks
        """
        max_tokens = max_tokens or self.config.max_tokens
        overlap = overlap or self.config.overlap_tokens
        
        if self.count_tokens(content) <= max_tokens:
            return [content]
        
        chunks = []
        
        if preserve_lines:
            lines = content.split('\n')
            current_chunk = []
            current_tokens = 0
            
            for line in lines:
                line_tokens = self.count_tokens(line + '\n')
                
                if current_tokens + line_tokens > max_tokens and current_chunk:
                    # Save current chunk
                    chunks.append('\n'.join(current_chunk))
                    
                    # Start new chunk with overlap
                    if overlap > 0 and chunks:
                        # Take last few lines for overlap
                        overlap_lines = []
                        overlap_tokens = 0
                        
                        for i in range(len(current_chunk) - 1, -1, -1):
                            line_tok = self.count_tokens(current_chunk[i] + '\n')
                            if overlap_tokens + line_tok <= overlap:
                                overlap_lines.insert(0, current_chunk[i])
                                overlap_tokens += line_tok
                            else:
                                break
                        
                        current_chunk = overlap_lines + [line]
                        current_tokens = overlap_tokens + line_tokens
                    else:
                        current_chunk = [line]
                        current_tokens = line_tokens
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens
            
            # Add remaining chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
        
        else:
            # Character-based splitting with token awareness
            words = content.split()
            current_chunk = []
            current_tokens = 0
            
            for word in words:
                word_tokens = self.count_tokens(word + ' ')
                
                if current_tokens + word_tokens > max_tokens and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Handle overlap
                    if overlap > 0 and chunks:
                        overlap_words = []
                        overlap_tokens = 0
                        
                        for i in range(len(current_chunk) - 1, -1, -1):
                            w_tok = self.count_tokens(current_chunk[i] + ' ')
                            if overlap_tokens + w_tok <= overlap:
                                overlap_words.insert(0, current_chunk[i])
                                overlap_tokens += w_tok
                            else:
                                break
                        
                        current_chunk = overlap_words + [word]
                        current_tokens = overlap_tokens + word_tokens
                    else:
                        current_chunk = [word]
                        current_tokens = word_tokens
                else:
                    current_chunk.append(word)
                    current_tokens += word_tokens
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        self._stats['splits_performed'] += len(chunks) - 1
        
        return chunks
    
    def extract_structure(self, content: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract structural information from content
        
        Args:
            content: Content to analyze
            language: Programming language
            
        Returns:
            Structural information
        """
        structure = {
            'classes': [],
            'functions': [],
            'imports': [],
            'exports': [],
            'comments': [],
            'docstrings': [],
            'complexity': 0,
            'dependencies': []
        }
        
        # Language-specific extraction
        if language == 'python':
            structure['classes'] = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            structure['functions'] = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            structure['imports'] = re.findall(r'^(?:from|import)\s+([^\s]+)', content, re.MULTILINE)
        elif language in ['javascript', 'typescript']:
            structure['classes'] = re.findall(r'class\s+(\w+)', content)
            structure['functions'] = re.findall(r'function\s+(\w+)', content)
            structure['imports'] = re.findall(r'import.*from\s+[\'"]([^\'"]]+)', content)
            structure['exports'] = re.findall(r'export\s+(?:default\s+)?(\w+)', content)
        elif language == 'java':
            structure['classes'] = re.findall(r'(?:public\s+)?class\s+(\w+)', content)
            structure['functions'] = re.findall(r'(?:public|private|protected).*\s+(\w+)\s*\(', content)
            structure['imports'] = re.findall(r'import\s+([^;]+);', content)
        
        return structure
    
    def calculate_complexity(self, content: str, language: Optional[str] = None) -> float:
        """
        Calculate complexity score for content
        
        Args:
            content: Content to analyze
            language: Programming language
            
        Returns:
            Complexity score
        """
        complexity = 0.0
        
        # Count control flow statements
        control_flow = ['if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'try', 'catch', 'except']
        for keyword in control_flow:
            complexity += content.count(f' {keyword} ') + content.count(f' {keyword}(')
        
        # Count nesting depth
        max_indent = 0
        for line in content.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        complexity += max_indent / 4  # Assuming 4-space indents
        
        # Count function/method definitions
        if language == 'python':
            complexity += len(re.findall(r'^def\s+', content, re.MULTILINE)) * 2
        elif language in ['javascript', 'java', 'cpp']:
            complexity += len(re.findall(r'function\s+\w+|public\s+\w+\s*\(', content)) * 2
        
        self._stats['complexity_calculated'] += 1
        
        return complexity
    
    def remove_comments(self, content: str, language: Optional[str] = None) -> str:
        """
        Remove comments from content
        
        Args:
            content: Content to process
            language: Programming language
            
        Returns:
            Content without comments
        """
        if not language or not self.config.include_comments:
            return content
        
        if language in self.comment_patterns:
            for pattern in self.comment_patterns[language]:
                content = re.sub(pattern, '', content, flags=re.MULTILINE)
        
        return content
    
    def extract_comments(self, content: str, language: Optional[str] = None) -> List[str]:
        """
        Extract comments from content
        
        Args:
            content: Content to process
            language: Programming language
            
        Returns:
            List of comments
        """
        comments = []
        
        if language and language in self.comment_patterns:
            for pattern in self.comment_patterns[language]:
                matches = re.findall(pattern, content, flags=re.MULTILINE)
                comments.extend(matches)
        
        return comments
    
    def validate_chunk(self, chunk: Chunk) -> Tuple[bool, List[str]]:
        """
        Validate a chunk against configured constraints
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check token count
        if chunk.token_count:
            if chunk.token_count > self.config.max_tokens:
                issues.append(f"Token count {chunk.token_count} exceeds max {self.config.max_tokens}")
            elif chunk.token_count < self.config.min_tokens:
                issues.append(f"Token count {chunk.token_count} below min {self.config.min_tokens}")
        
        # Check character count
        if chunk.char_count:
            if chunk.char_count > self.config.max_chunk_size:
                issues.append(f"Character count {chunk.char_count} exceeds max {self.config.max_chunk_size}")
            elif chunk.char_count < self.config.min_chunk_size:
                issues.append(f"Character count {chunk.char_count} below min {self.config.min_chunk_size}")
        
        # Check for empty content
        if not chunk.content.strip():
            issues.append("Chunk content is empty")
        
        # Check for required metadata
        if self.config.extract_metadata and not chunk.metadata:
            issues.append("Metadata extraction enabled but no metadata present")
        
        is_valid = len(issues) == 0
        
        self._stats['validations_performed'] += 1
        if not is_valid:
            self._stats['validation_failures'] += 1
        
        return is_valid, issues
    
    def merge_chunks(self, chunks: List[Chunk]) -> Chunk:
        """
        Merge multiple chunks into one
        
        Args:
            chunks: Chunks to merge
            
        Returns:
            Merged chunk
        """
        if not chunks:
            raise ValueError("Cannot merge empty chunk list")
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Merge content
        merged_content = '\n\n'.join(chunk.content for chunk in chunks)
        
        # Merge metadata
        merged_metadata = ChunkMetadata()
        for chunk in chunks:
            if chunk.metadata.tags:
                merged_metadata.tags.extend(chunk.metadata.tags)
            if chunk.metadata.dependencies:
                merged_metadata.dependencies.extend(chunk.metadata.dependencies)
        
        merged_metadata.tags = list(set(merged_metadata.tags))
        merged_metadata.dependencies = list(set(merged_metadata.dependencies))
        
        # Create merged chunk
        merged = self.create_chunk(
            content=merged_content,
            chunk_type=chunks[0].chunk_type,
            start_line=chunks[0].start_line,
            end_line=chunks[-1].end_line,
            metadata=asdict(merged_metadata)
        )
        
        self._stats['chunks_merged'] += len(chunks)
        
        return merged
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get chunker statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = dict(self._stats)
        
        if self._start_time:
            stats['processing_time'] = time.time() - self._start_time
        
        if self._chunks_created:
            stats['average_chunk_size'] = sum(c.char_count or 0 for c in self._chunks_created) / len(self._chunks_created)
            stats['average_token_count'] = sum(c.token_count or 0 for c in self._chunks_created) / len(self._chunks_created)
        
        return stats
    
    def reset_statistics(self):
        """Reset internal statistics"""
        self._stats = defaultdict(int)
        self._chunk_counter = 0
        self._chunks_created = []
        self._relationships = []
        self._start_time = None
    
    def __enter__(self):
        """Context manager entry"""
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._start_time:
            self._stats['total_time'] = time.time() - self._start_time