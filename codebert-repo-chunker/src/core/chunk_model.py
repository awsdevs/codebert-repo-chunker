"""
Enhanced chunk model with advanced features for semantic chunking
Provides rich chunk representation, serialization, and manipulation capabilities
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    List, Optional, Dict, Any, Tuple, Set, Union, 
    TypeVar, Generic, ClassVar, Protocol, runtime_checkable
)
import re
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar('T')

class ChunkType(Enum):
    """Comprehensive chunk type enumeration"""
    # Code structure chunks
    MODULE = "module"
    PACKAGE = "package"
    CLASS = "class"
    INTERFACE = "interface"
    TRAIT = "trait"
    STRUCT = "struct"
    ENUM = "enum"
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor"
    PROPERTY = "property"
    FIELD = "field"
    CONSTANT = "constant"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    
    # Code block chunks
    BLOCK = "block"
    SCOPE = "scope"
    NAMESPACE = "namespace"
    CLOSURE = "closure"
    LAMBDA = "lambda"
    ANONYMOUS_FUNCTION = "anonymous_function"
    
    # Control flow chunks
    CONDITIONAL = "conditional"
    LOOP = "loop"
    SWITCH = "switch"
    CASE = "case"
    TRY_CATCH = "try_catch"
    FINALLY = "finally"
    
    # Documentation chunks
    COMMENT = "comment"
    DOCSTRING = "docstring"
    README = "readme"
    LICENSE = "license"
    CHANGELOG = "changelog"
    TODO = "todo"
    NOTE = "note"
    WARNING = "warning"
    
    # Test chunks
    TEST_SUITE = "test_suite"
    TEST_CLASS = "test_class"
    TEST_METHOD = "test_method"
    TEST_FIXTURE = "test_fixture"
    TEST_MOCK = "test_mock"
    BENCHMARK = "benchmark"
    
    # Configuration chunks
    CONFIG = "config"
    SETTINGS = "settings"
    ENVIRONMENT = "environment"
    MANIFEST = "manifest"
    SCHEMA = "schema"
    TEMPLATE = "template"
    
    # Data chunks
    DATA = "data"
    DATASET = "dataset"
    MODEL_DATA = "model_data"
    FIXTURE_DATA = "fixture_data"
    SEED_DATA = "seed_data"
    MIGRATION = "migration"
    
    # Database chunks
    TABLE = "table"
    VIEW = "view"
    INDEX = "index"
    TRIGGER = "trigger"
    PROCEDURE = "procedure"
    QUERY = "query"
    TRANSACTION = "transaction"
    
    # Infrastructure chunks
    DEPLOYMENT = "deployment"
    CONTAINER = "container"
    SERVICE = "service"
    PIPELINE = "pipeline"
    WORKFLOW = "workflow"
    
    # Web/API chunks
    ENDPOINT = "endpoint"
    ROUTE = "route"
    MIDDLEWARE = "middleware"
    CONTROLLER = "controller"
    MODEL = "model"
    VIEW_COMPONENT = "view_component"
    
    # Generic chunks
    SECTION = "section"
    FRAGMENT = "fragment"
    SNIPPET = "snippet"
    PARTIAL = "partial"
    COMPLETE = "complete"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"

class ChunkPriority(IntEnum):
    """Priority levels for chunk processing"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1

class ChunkStatus(Enum):
    """Status of chunk processing"""
    CREATED = "created"
    VALIDATED = "validated"
    INDEXED = "indexed"
    EMBEDDED = "embedded"
    CACHED = "cached"
    PROCESSED = "processed"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class ChunkVisibility(Enum):
    """Visibility/access level of chunk"""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"
    INTERNAL = "internal"

class RelationType(Enum):
    """Types of relationships between chunks"""
    # Structural relationships
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    ANCESTOR = "ancestor"
    DESCENDANT = "descendant"
    
    # Code relationships
    IMPORTS = "imports"
    IMPORTED_BY = "imported_by"
    EXPORTS = "exports"
    EXPORTED_BY = "exported_by"
    CALLS = "calls"
    CALLED_BY = "called_by"
    INSTANTIATES = "instantiates"
    INSTANTIATED_BY = "instantiated_by"
    INHERITS = "inherits"
    INHERITED_BY = "inherited_by"
    IMPLEMENTS = "implements"
    IMPLEMENTED_BY = "implemented_by"
    EXTENDS = "extends"
    EXTENDED_BY = "extended_by"
    USES = "uses"
    USED_BY = "used_by"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    
    # Test relationships
    TESTS = "tests"
    TESTED_BY = "tested_by"
    MOCKS = "mocks"
    MOCKED_BY = "mocked_by"
    
    # Documentation relationships
    DOCUMENTS = "documents"
    DOCUMENTED_BY = "documented_by"
    ANNOTATES = "annotates"
    ANNOTATED_BY = "annotated_by"
    
    # Dependency relationships
    DEPENDS_ON = "depends_on"
    DEPENDENCY_OF = "dependency_of"
    REQUIRES = "requires"
    REQUIRED_BY = "required_by"
    
    # Versioning relationships
    VERSION_OF = "version_of"
    HAS_VERSION = "has_version"
    REPLACES = "replaces"
    REPLACED_BY = "replaced_by"
    
    # Semantic relationships
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    ALTERNATIVE_TO = "alternative_to"
    CONTRADICTS = "contradicts"

@dataclass
class ChunkLocation:
    """Location information for a chunk"""
    file_path: str
    start_line: int
    end_line: int
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    start_byte: Optional[int] = None
    end_byte: Optional[int] = None
    
    def contains(self, line: int, column: Optional[int] = None) -> bool:
        """Check if location contains given line/column"""
        if not (self.start_line <= line <= self.end_line):
            return False
        
        if column is not None and self.start_column and self.end_column:
            if line == self.start_line and column < self.start_column:
                return False
            if line == self.end_line and column > self.end_column:
                return False
        
        return True
    
    def overlaps(self, other: 'ChunkLocation') -> bool:
        """Check if this location overlaps with another"""
        if self.file_path != other.file_path:
            return False
        
        return not (
            self.end_line < other.start_line or 
            self.start_line > other.end_line
        )
    
    def merge(self, other: 'ChunkLocation') -> 'ChunkLocation':
        """Merge with another location"""
        if self.file_path != other.file_path:
            raise ValueError("Cannot merge locations from different files")
        
        return ChunkLocation(
            file_path=self.file_path,
            start_line=min(self.start_line, other.start_line),
            end_line=max(self.end_line, other.end_line),
            start_column=min(self.start_column or 0, other.start_column or 0) or None,
            end_column=max(self.end_column or 0, other.end_column or 0) or None,
            start_byte=min(self.start_byte or 0, other.start_byte or 0) or None,
            end_byte=max(self.end_byte or 0, other.end_byte or 0) or None
        )

@dataclass
class ChunkSignature:
    """Signature information for code chunks"""
    name: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    type_parameters: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)  # public, static, async, etc.
    annotations: List[str] = field(default_factory=list)  # decorators, attributes
    throws: List[str] = field(default_factory=list)  # exceptions
    
    def to_string(self) -> str:
        """Convert signature to string representation"""
        parts = []
        
        if self.modifiers:
            parts.append(' '.join(self.modifiers))
        
        if self.return_type:
            parts.append(self.return_type)
        
        if self.name:
            parts.append(self.name)
        
        if self.type_parameters:
            parts.append(f"<{', '.join(self.type_parameters)}>")
        
        if self.parameters:
            params = []
            for param in self.parameters:
                param_str = f"{param.get('type', 'Any')} {param.get('name', '_')}"
                if param.get('default'):
                    param_str += f" = {param['default']}"
                params.append(param_str)
            parts.append(f"({', '.join(params)})")
        
        if self.throws:
            parts.append(f"throws {', '.join(self.throws)}")
        
        return ' '.join(parts)

@dataclass
class ChunkMetrics:
    """Metrics and measurements for chunks"""
    lines_of_code: int = 0
    lines_of_comments: int = 0
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    maintainability_index: float = 0.0
    halstead_metrics: Dict[str, float] = field(default_factory=dict)
    token_count: int = 0
    character_count: int = 0
    word_count: int = 0
    unique_tokens: int = 0
    vocabulary_size: int = 0
    readability_score: float = 0.0
    test_coverage: Optional[float] = None
    duplication_ratio: float = 0.0
    
    def quality_score(self) -> float:
        """Calculate overall quality score"""
        score = 100.0
        
        # Penalize high complexity
        if self.cyclomatic_complexity > 10:
            score -= (self.cyclomatic_complexity - 10) * 2
        
        if self.cognitive_complexity > 15:
            score -= (self.cognitive_complexity - 15) * 1.5
        
        # Reward good maintainability
        if self.maintainability_index > 0:
            score = (score + self.maintainability_index) / 2
        
        # Penalize duplication
        score -= self.duplication_ratio * 20
        
        # Reward test coverage
        if self.test_coverage is not None:
            score += self.test_coverage * 0.2
        
        return max(0, min(100, score))

@dataclass
class ChunkEmbedding:
    """Embedding representation of a chunk"""
    vector: Optional[np.ndarray] = None
    model: Optional[str] = None
    dimension: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def similarity(self, other: 'ChunkEmbedding') -> float:
        """Calculate cosine similarity with another embedding"""
        if self.vector is None or other.vector is None:
            return 0.0
        
        dot_product = np.dot(self.vector, other.vector)
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def to_list(self) -> Optional[List[float]]:
        """Convert vector to list"""
        if self.vector is not None:
            return self.vector.tolist()
        return None
    
    @classmethod
    def from_list(cls, vector_list: List[float], model: Optional[str] = None) -> 'ChunkEmbedding':
        """Create from list"""
        return cls(
            vector=np.array(vector_list),
            model=model,
            dimension=len(vector_list),
            timestamp=datetime.now()
        )

@dataclass
class ChunkAnnotation:
    """Annotation/tag for a chunk"""
    key: str
    value: Any
    confidence: float = 1.0
    source: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkRelation:
    """Relationship between chunks"""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def reverse(self) -> 'ChunkRelation':
        """Create reverse relationship"""
        # Map relation types to their reverse
        reverse_map = {
            RelationType.PARENT: RelationType.CHILD,
            RelationType.CHILD: RelationType.PARENT,
            RelationType.IMPORTS: RelationType.IMPORTED_BY,
            RelationType.IMPORTED_BY: RelationType.IMPORTS,
            RelationType.CALLS: RelationType.CALLED_BY,
            RelationType.CALLED_BY: RelationType.CALLS,
            RelationType.INHERITS: RelationType.INHERITED_BY,
            RelationType.INHERITED_BY: RelationType.INHERITS,
            RelationType.TESTS: RelationType.TESTED_BY,
            RelationType.TESTED_BY: RelationType.TESTS,
            RelationType.DEPENDS_ON: RelationType.DEPENDENCY_OF,
            RelationType.DEPENDENCY_OF: RelationType.DEPENDS_ON,
        }
        
        reversed_type = reverse_map.get(self.relation_type, self.relation_type)
        
        return ChunkRelation(
            source_id=self.target_id,
            target_id=self.source_id,
            relation_type=reversed_type,
            strength=self.strength,
            bidirectional=self.bidirectional,
            metadata=self.metadata,
            timestamp=self.timestamp
        )

@dataclass
class ChunkDependency:
    """Dependency information for a chunk"""
    name: str
    version: Optional[str] = None
    dependency_type: str = "runtime"  # runtime, compile, test, dev
    scope: str = "required"  # required, optional, provided
    resolved: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkHistory:
    """Version history for a chunk"""
    version: str
    timestamp: datetime
    author: Optional[str] = None
    message: Optional[str] = None
    diff: Optional[str] = None
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    """
    Enhanced chunk model with comprehensive features
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    chunk_type: ChunkType = ChunkType.UNKNOWN
    
    # Content
    content: str = ""
    original_content: Optional[str] = None
    normalized_content: Optional[str] = None
    
    # Location
    location: Optional[ChunkLocation] = None
    
    # Structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    
    # Signature (for code chunks)
    signature: Optional[ChunkSignature] = None
    
    # Metadata
    language: Optional[str] = None
    framework: Optional[str] = None
    visibility: ChunkVisibility = ChunkVisibility.PUBLIC
    priority: ChunkPriority = ChunkPriority.MEDIUM
    status: ChunkStatus = ChunkStatus.CREATED
    
    # Metrics
    metrics: ChunkMetrics = field(default_factory=ChunkMetrics)
    
    # Embedding
    embedding: Optional[ChunkEmbedding] = None
    
    # Relationships
    relations: List[ChunkRelation] = field(default_factory=list)
    
    # Dependencies
    dependencies: List[ChunkDependency] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    
    # Annotations
    annotations: List[ChunkAnnotation] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    # History
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    history: List[ChunkHistory] = field(default_factory=list)
    
    # Caching
    hash: Optional[str] = None
    cache_key: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Class-level configuration
    _registry: ClassVar[Dict[str, 'Chunk']] = {}
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Calculate hash if not provided
        if not self.hash:
            self.hash = self.calculate_hash()
        
        # Generate cache key
        if not self.cache_key:
            self.cache_key = f"{self.chunk_type.value}_{self.hash[:8]}"
        
        # Calculate metrics
        if not self.metrics.character_count:
            self.metrics.character_count = len(self.content)
        
        if not self.metrics.lines_of_code:
            self.metrics.lines_of_code = self.content.count('\n') + 1
        
        # Register chunk
        self._registry[self.id] = self
    
    def calculate_hash(self) -> str:
        """Calculate content hash"""
        content = self.content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def add_relation(self, target_id: str, relation_type: RelationType, 
                    strength: float = 1.0, bidirectional: bool = False):
        """Add a relation to another chunk"""
        relation = ChunkRelation(
            source_id=self.id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
            timestamp=datetime.now()
        )
        self.relations.append(relation)
        
        # Add reverse relation if bidirectional
        if bidirectional and target_id in self._registry:
            target = self._registry[target_id]
            target.relations.append(relation.reverse())
    
    def add_annotation(self, key: str, value: Any, confidence: float = 1.0, 
                      source: Optional[str] = None):
        """Add an annotation"""
        annotation = ChunkAnnotation(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            timestamp=datetime.now()
        )
        self.annotations.append(annotation)
    
    def get_annotation(self, key: str) -> Optional[Any]:
        """Get annotation value by key"""
        for annotation in self.annotations:
            if annotation.key == key:
                return annotation.value
        return None
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[ChunkRelation]:
        """Get all relations of a specific type"""
        return [r for r in self.relations if r.relation_type == relation_type]
    
    def get_related_chunks(self, relation_type: Optional[RelationType] = None) -> List['Chunk']:
        """Get related chunks"""
        related = []
        relations = self.relations if relation_type is None else self.get_relations_by_type(relation_type)
        
        for relation in relations:
            target_id = relation.target_id if relation.source_id == self.id else relation.source_id
            if target_id in self._registry:
                related.append(self._registry[target_id])
        
        return related
    
    def get_parent(self) -> Optional['Chunk']:
        """Get parent chunk"""
        if self.parent_id and self.parent_id in self._registry:
            return self._registry[self.parent_id]
        return None
    
    def get_children(self) -> List['Chunk']:
        """Get child chunks"""
        children = []
        for child_id in self.children_ids:
            if child_id in self._registry:
                children.append(self._registry[child_id])
        return children
    
    def get_siblings(self) -> List['Chunk']:
        """Get sibling chunks"""
        siblings = []
        
        # Get siblings through parent
        parent = self.get_parent()
        if parent:
            for child_id in parent.children_ids:
                if child_id != self.id and child_id in self._registry:
                    siblings.append(self._registry[child_id])
        
        # Also check explicit sibling IDs
        for sibling_id in self.sibling_ids:
            if sibling_id in self._registry:
                sibling = self._registry[sibling_id]
                if sibling not in siblings:
                    siblings.append(sibling)
        
        return siblings
    
    def is_test(self) -> bool:
        """Check if this is a test chunk"""
        return self.chunk_type in [
            ChunkType.TEST_SUITE, ChunkType.TEST_CLASS, 
            ChunkType.TEST_METHOD, ChunkType.TEST_FIXTURE
        ]
    
    def is_documentation(self) -> bool:
        """Check if this is documentation"""
        return self.chunk_type in [
            ChunkType.COMMENT, ChunkType.DOCSTRING, 
            ChunkType.README, ChunkType.CHANGELOG
        ]
    
    def is_config(self) -> bool:
        """Check if this is configuration"""
        return self.chunk_type in [
            ChunkType.CONFIG, ChunkType.SETTINGS, 
            ChunkType.ENVIRONMENT, ChunkType.SCHEMA
        ]
    
    def update_content(self, new_content: str, author: Optional[str] = None, 
                      message: Optional[str] = None):
        """Update content with history tracking"""
        # Save current version to history
        history_entry = ChunkHistory(
            version=self.version,
            timestamp=self.updated_at,
            author=author,
            message=message,
            diff=self._calculate_diff(self.content, new_content),
            parent_version=self.history[-1].version if self.history else None
        )
        self.history.append(history_entry)
        
        # Update content
        self.original_content = self.content
        self.content = new_content
        self.hash = self.calculate_hash()
        self.updated_at = datetime.now()
        
        # Increment version
        version_parts = self.version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        self.version = '.'.join(version_parts)
        
        # Update metrics
        self.metrics.character_count = len(new_content)
        self.metrics.lines_of_code = new_content.count('\n') + 1
    
    def _calculate_diff(self, old_content: str, new_content: str) -> str:
        """Calculate diff between old and new content"""
        # Simple line-based diff (can be enhanced with difflib)
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        diff_lines = []
        for i, (old, new) in enumerate(zip(old_lines, new_lines)):
            if old != new:
                diff_lines.append(f"Line {i+1}: {old} -> {new}")
        
        return '\n'.join(diff_lines[:10])  # Limit to first 10 changes
    
    def merge_with(self, other: 'Chunk', strategy: str = "append") -> 'Chunk':
        """Merge with another chunk"""
        if strategy == "append":
            merged_content = self.content + "\n\n" + other.content
        elif strategy == "prepend":
            merged_content = other.content + "\n\n" + self.content
        elif strategy == "replace":
            merged_content = other.content
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        # Create new merged chunk
        merged = Chunk(
            chunk_type=self.chunk_type,
            content=merged_content,
            language=self.language,
            framework=self.framework,
            priority=max(self.priority, other.priority),
            visibility=self.visibility
        )
        
        # Merge locations
        if self.location and other.location:
            merged.location = self.location.merge(other.location)
        
        # Merge relations
        merged.relations = self.relations + other.relations
        
        # Merge dependencies
        merged.dependencies = list(set(self.dependencies + other.dependencies))
        
        # Merge tags
        merged.tags = self.tags.union(other.tags)
        
        return merged
    
    def split(self, max_size: int = 1000) -> List['Chunk']:
        """Split chunk into smaller chunks"""
        if len(self.content) <= max_size:
            return [self]
        
        chunks = []
        lines = self.content.split('\n')
        current_chunk_lines = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_size and current_chunk_lines:
                # Create chunk
                chunk_content = '\n'.join(current_chunk_lines)
                chunk = Chunk(
                    chunk_type=ChunkType.PARTIAL,
                    content=chunk_content,
                    language=self.language,
                    parent_id=self.id
                )
                chunks.append(chunk)
                
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Add remaining lines
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk = Chunk(
                chunk_type=ChunkType.PARTIAL,
                content=chunk_content,
                language=self.language,
                parent_id=self.id
            )
            chunks.append(chunk)
        
        return chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'chunk_type': self.chunk_type.value,
            'content': self.content,
            'location': asdict(self.location) if self.location else None,
            'signature': asdict(self.signature) if self.signature else None,
            'language': self.language,
            'framework': self.framework,
            'visibility': self.visibility.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'metrics': asdict(self.metrics),
            'embedding': self.embedding.to_list() if self.embedding else None,
            'relations': [asdict(r) for r in self.relations],
            'dependencies': [asdict(d) for d in self.dependencies],
            'imports': self.imports,
            'exports': self.exports,
            'annotations': [asdict(a) for a in self.annotations],
            'tags': list(self.tags),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'hash': self.hash,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create from dictionary"""
        # Convert enums
        if 'chunk_type' in data:
            data['chunk_type'] = ChunkType(data['chunk_type'])
        if 'visibility' in data:
            data['visibility'] = ChunkVisibility(data['visibility'])
        if 'priority' in data:
            data['priority'] = ChunkPriority(data['priority'])
        if 'status' in data:
            data['status'] = ChunkStatus(data['status'])
        
        # Convert complex types
        if 'location' in data and data['location']:
            data['location'] = ChunkLocation(**data['location'])
        if 'signature' in data and data['signature']:
            data['signature'] = ChunkSignature(**data['signature'])
        if 'metrics' in data:
            data['metrics'] = ChunkMetrics(**data['metrics'])
        if 'embedding' in data and data['embedding']:
            data['embedding'] = ChunkEmbedding.from_list(data['embedding'])
        
        # Convert datetime strings
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert sets
        if 'tags' in data:
            data['tags'] = set(data['tags'])
        
        # Convert relations
        if 'relations' in data:
            relations = []
            for rel in data['relations']:
                if 'relation_type' in rel:
                    rel['relation_type'] = RelationType(rel['relation_type'])
                relations.append(ChunkRelation(**rel))
            data['relations'] = relations
        
        # Convert annotations
        if 'annotations' in data:
            data['annotations'] = [ChunkAnnotation(**ann) for ann in data['annotations']]
        
        # Convert dependencies
        if 'dependencies' in data:
            data['dependencies'] = [ChunkDependency(**dep) for dep in data['dependencies']]
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Chunk':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Chunk(id={self.id}, type={self.chunk_type.value}, size={len(self.content)})"
    
    def __str__(self) -> str:
        """Human-readable string"""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"{self.chunk_type.value}: {preview}"
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, Chunk):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts"""
        return hash(self.id)

class ChunkCollection:
    """Collection of chunks with search and filter capabilities"""
    
    def __init__(self, chunks: Optional[List[Chunk]] = None):
        """Initialize collection"""
        self.chunks: Dict[str, Chunk] = {}
        if chunks:
            for chunk in chunks:
                self.add(chunk)
    
    def add(self, chunk: Chunk):
        """Add chunk to collection"""
        self.chunks[chunk.id] = chunk
    
    def remove(self, chunk_id: str):
        """Remove chunk from collection"""
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]
    
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def filter_by_type(self, chunk_type: ChunkType) -> List[Chunk]:
        """Filter chunks by type"""
        return [c for c in self.chunks.values() if c.chunk_type == chunk_type]
    
    def filter_by_language(self, language: str) -> List[Chunk]:
        """Filter chunks by language"""
        return [c for c in self.chunks.values() if c.language == language]
    
    def filter_by_tag(self, tag: str) -> List[Chunk]:
        """Filter chunks by tag"""
        return [c for c in self.chunks.values() if tag in c.tags]
    
    def search_content(self, pattern: str, regex: bool = False) -> List[Chunk]:
        """Search chunks by content"""
        chunks = []
        for chunk in self.chunks.values():
            if regex:
                if re.search(pattern, chunk.content):
                    chunks.append(chunk)
            else:
                if pattern in chunk.content:
                    chunks.append(chunk)
        return chunks
    
    def get_related(self, chunk_id: str, relation_type: Optional[RelationType] = None) -> List[Chunk]:
        """Get chunks related to given chunk"""
        chunk = self.get(chunk_id)
        if not chunk:
            return []
        
        related = []
        for relation in chunk.relations:
            if relation_type and relation.relation_type != relation_type:
                continue
            
            target = self.get(relation.target_id)
            if target:
                related.append(target)
        
        return related
    
    def build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build dependency graph"""
        graph = defaultdict(set)
        
        for chunk in self.chunks.values():
            for relation in chunk.relations:
                if relation.relation_type in [RelationType.DEPENDS_ON, RelationType.IMPORTS]:
                    graph[chunk.id].add(relation.target_id)
        
        return dict(graph)
    
    def topological_sort(self) -> List[Chunk]:
        """Sort chunks in dependency order"""
        graph = self.build_dependency_graph()
        visited = set()
        stack = []
        
        def visit(chunk_id: str):
            if chunk_id in visited:
                return
            
            visited.add(chunk_id)
            for dep in graph.get(chunk_id, []):
                visit(dep)
            
            if chunk_id in self.chunks:
                stack.append(self.chunks[chunk_id])
        
        for chunk_id in self.chunks:
            visit(chunk_id)
        
        return list(reversed(stack))
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert collection to JSON"""
        data = {
            'chunks': [chunk.to_dict() for chunk in self.chunks.values()],
            'total': len(self.chunks),
            'types': dict(Counter(c.chunk_type.value for c in self.chunks.values()))
        }
        return json.dumps(data, default=str, indent=indent)
    
    def __len__(self) -> int:
        """Number of chunks"""
        return len(self.chunks)
    
    def __iter__(self):
        """Iterate over chunks"""
        return iter(self.chunks.values())
    
    def __contains__(self, chunk_id: str) -> bool:
        """Check if chunk ID exists"""
        return chunk_id in self.chunks