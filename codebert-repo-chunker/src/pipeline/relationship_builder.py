"""
Relationship builder for establishing connections between code chunks
Builds comprehensive relationship graphs for better code understanding
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json
import pickle
import re
import ast
import logging
from collections import defaultdict, Counter
import hashlib
from datetime import datetime, timezone
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from tqdm import tqdm

# Internal imports
from src.core.chunk_model import Chunk, ChunkRelation, RelationType
from src.core.file_context import FileContext
from src.embeddings.embedding_storage import EmbeddingStorage
from src.classifiers.content_analyzer import ContentAnalyzer
from src.utils.similarity import SimilarityCalculator
from src.utils.graph_utils import GraphAnalyzer, CommunityDetector

logger = logging.getLogger(__name__)

class RelationshipStrength(Enum):
    """Strength levels for relationships"""
    VERY_STRONG = 1.0
    STRONG = 0.8
    MODERATE = 0.6
    WEAK = 0.4
    VERY_WEAK = 0.2

class RelationshipConfidence(Enum):
    """Confidence levels for detected relationships"""
    CERTAIN = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    UNCERTAIN = 0.2

@dataclass
class RelationshipConfig:
    """Configuration for relationship building"""
    # Analysis settings
    enable_import_analysis: bool = True
    enable_call_analysis: bool = True
    enable_inheritance_analysis: bool = True
    enable_similarity_analysis: bool = True
    enable_semantic_analysis: bool = True
    
    # Similarity thresholds
    similarity_threshold: float = 0.7
    semantic_threshold: float = 0.75
    structural_threshold: float = 0.6
    
    # Graph settings
    max_relationships_per_chunk: int = 50
    min_relationship_strength: float = 0.3
    enable_transitive_reduction: bool = True
    
    # Performance settings
    max_workers: int = 8
    batch_size: int = 100
    use_cache: bool = True
    cache_path: Path = Path(".cache/relationships")
    
    # Analysis depth
    max_depth: int = 5
    include_indirect: bool = True
    analyze_cross_file: bool = True
    
    # Weights for relationship scoring
    weights: Dict[str, float] = field(default_factory=lambda: {
        'import': 1.0,
        'call': 0.9,
        'inheritance': 0.95,
        'similarity': 0.7,
        'semantic': 0.8,
        'structural': 0.6,
        'proximity': 0.5
    })

@dataclass
class RelationshipEdge:
    """Represents an edge in the relationship graph"""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float
    confidence: float
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class RelationshipGraph:
    """Complete relationship graph"""
    nodes: Dict[str, Dict[str, Any]]  # chunk_id -> node attributes
    edges: List[RelationshipEdge]
    clusters: Dict[int, List[str]]  # cluster_id -> chunk_ids
    hierarchy: Dict[str, List[str]]  # parent_id -> child_ids
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]

class RelationshipBuilder:
    """
    Builds comprehensive relationships between code chunks
    """
    
    def __init__(self, config: Optional[RelationshipConfig] = None):
        """
        Initialize relationship builder
        
        Args:
            config: Configuration for relationship building
        """
        self.config = config or RelationshipConfig()
        
        # Initialize components
        self._init_components()
        
        # Caching
        self.cache = {}
        self.cache_path = self.config.cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _init_components(self):
        """Initialize analysis components"""
        # Graph for relationship storage
        self.graph = nx.DiGraph()
        
        # Similarity calculator
        self.similarity_calculator = SimilarityCalculator()
        
        # Content analyzer
        self.content_analyzer = ContentAnalyzer()
        
        # Graph analyzer
        self.graph_analyzer = GraphAnalyzer()
        
        # Community detector
        self.community_detector = CommunityDetector()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def build_relationships(self, 
                          chunks: List[Chunk],
                          storage: Optional[EmbeddingStorage] = None,
                          show_progress: bool = True) -> RelationshipGraph:
        """
        Build relationships between chunks
        
        Args:
            chunks: List of chunks to analyze
            storage: Optional embedding storage for similarity
            show_progress: Show progress bar
            
        Returns:
            Complete relationship graph
        """
        logger.info(f"Building relationships for {len(chunks)} chunks")
        
        # Create chunk index
        self.chunk_index = {chunk.id: chunk for chunk in chunks}
        
        # Initialize graph nodes
        self._init_graph_nodes(chunks)
        
        # Build different types of relationships
        relationships = []
        
        # 1. Import relationships
        if self.config.enable_import_analysis:
            import_rels = self._analyze_imports(chunks, show_progress)
            relationships.extend(import_rels)
        
        # 2. Call relationships
        if self.config.enable_call_analysis:
            call_rels = self._analyze_calls(chunks, show_progress)
            relationships.extend(call_rels)
        
        # 3. Inheritance relationships
        if self.config.enable_inheritance_analysis:
            inheritance_rels = self._analyze_inheritance(chunks, show_progress)
            relationships.extend(inheritance_rels)
        
        # 4. Similarity relationships
        if self.config.enable_similarity_analysis and storage:
            similarity_rels = self._analyze_similarity(chunks, storage, show_progress)
            relationships.extend(similarity_rels)
        
        # 5. Semantic relationships
        if self.config.enable_semantic_analysis:
            semantic_rels = self._analyze_semantic(chunks, show_progress)
            relationships.extend(semantic_rels)
        
        # Add relationships to graph
        self._add_relationships_to_graph(relationships)
        
        # Post-process graph
        self._post_process_graph()
        
        # Detect communities
        clusters = self._detect_communities()
        
        # Build hierarchy
        hierarchy = self._build_hierarchy()
        
        # Calculate statistics
        statistics = self._calculate_statistics()
        
        # Create relationship graph
        relationship_graph = RelationshipGraph(
            nodes=dict(self.graph.nodes(data=True)),
            edges=relationships,
            clusters=clusters,
            hierarchy=hierarchy,
            statistics=statistics,
            metadata={
                'total_chunks': len(chunks),
                'total_relationships': len(relationships),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
        
        logger.info(f"Built {len(relationships)} relationships")
        
        return relationship_graph
    
    def _init_graph_nodes(self, chunks: List[Chunk]):
        """Initialize graph nodes from chunks"""
        for chunk in chunks:
            self.graph.add_node(
                chunk.id,
                chunk_type=chunk.chunk_type.value,
                file_path=chunk.file_path,
                language=chunk.metadata.language,
                size=len(chunk.content),
                lines=chunk.line_count,
                complexity=chunk.metadata.metrics.get('complexity', 0)
            )
    
    def _analyze_imports(self, chunks: List[Chunk], show_progress: bool) -> List[RelationshipEdge]:
        """Analyze import relationships"""
        logger.info("Analyzing import relationships")
        relationships = []
        
        if show_progress:
            chunks = tqdm(chunks, desc="Analyzing imports")
        
        for chunk in chunks:
            if not chunk.metadata.language:
                continue
            
            # Extract imports from chunk
            imports = self._extract_imports(chunk)
            
            # Find chunks that match imports
            for import_name in imports:
                matching_chunks = self._find_matching_chunks(import_name, chunks)
                
                for target_chunk in matching_chunks:
                    if target_chunk.id != chunk.id:
                        rel = RelationshipEdge(
                            source_id=chunk.id,
                            target_id=target_chunk.id,
                            relation_type=RelationType.IMPORTS,
                            strength=RelationshipStrength.STRONG.value,
                            confidence=RelationshipConfidence.HIGH.value,
                            metadata={'import': import_name},
                            evidence=[f"Imports {import_name}"]
                        )
                        relationships.append(rel)
                        self.stats['import_relationships'] += 1
        
        return relationships
    
    def _extract_imports(self, chunk: Chunk) -> List[str]:
        """Extract import statements from chunk"""
        imports = []
        
        if chunk.metadata.language == 'python':
            # Python imports
            import_pattern = re.compile(r'^\s*(?:from\s+([^\s]+)\s+)?import\s+([^\s]+)', re.MULTILINE)
            for match in import_pattern.finditer(chunk.content):
                module = match.group(1) or match.group(2)
                imports.append(module)
            
            # Also parse with AST for accuracy
            try:
                tree = ast.parse(chunk.content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except:
                pass
        
        elif chunk.metadata.language in ['javascript', 'typescript']:
            # JavaScript/TypeScript imports
            import_patterns = [
                re.compile(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'),
                re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]'),
                re.compile(r'import\s*\(\s*[\'"]([^\'"]+)[\'"]')  # Dynamic imports
            ]
            for pattern in import_patterns:
                for match in pattern.finditer(chunk.content):
                    imports.append(match.group(1))
        
        elif chunk.metadata.language == 'java':
            # Java imports
            import_pattern = re.compile(r'^\s*import\s+([^;]+);', re.MULTILINE)
            for match in import_pattern.finditer(chunk.content):
                imports.append(match.group(1))
        
        elif chunk.metadata.language == 'go':
            # Go imports
            import_pattern = re.compile(r'import\s+(?:\(([^)]+)\)|"([^"]+)")')
            for match in import_pattern.finditer(chunk.content):
                if match.group(1):  # Multiple imports
                    for line in match.group(1).split('\n'):
                        line = line.strip().strip('"')
                        if line:
                            imports.append(line)
                elif match.group(2):  # Single import
                    imports.append(match.group(2))
        
        return imports
    
    def _find_matching_chunks(self, import_name: str, chunks: List[Chunk]) -> List[Chunk]:
        """Find chunks that match an import name"""
        matching = []
        
        # Normalize import name
        import_parts = import_name.replace('.', '/').split('/')
        import_base = import_parts[-1]
        
        for chunk in chunks:
            # Check file name
            if chunk.file_path:
                file_name = Path(chunk.file_path).stem
                if file_name == import_base or import_base in file_name:
                    matching.append(chunk)
                    continue
            
            # Check chunk name/signature
            if hasattr(chunk, 'signature') and chunk.signature:
                if chunk.signature.name == import_base:
                    matching.append(chunk)
                    continue
            
            # Check exports
            if chunk.metadata.exports:
                if import_base in chunk.metadata.exports:
                    matching.append(chunk)
        
        return matching
    
    def _analyze_calls(self, chunks: List[Chunk], show_progress: bool) -> List[RelationshipEdge]:
        """Analyze function/method call relationships"""
        logger.info("Analyzing call relationships")
        relationships = []
        
        if show_progress:
            chunks = tqdm(chunks, desc="Analyzing calls")
        
        for chunk in chunks:
            if not chunk.metadata.language:
                continue
            
            # Extract function calls
            calls = self._extract_function_calls(chunk)
            
            # Find target chunks
            for call_name in calls:
                target_chunks = self._find_function_chunks(call_name, chunks)
                
                for target_chunk in target_chunks:
                    if target_chunk.id != chunk.id:
                        rel = RelationshipEdge(
                            source_id=chunk.id,
                            target_id=target_chunk.id,
                            relation_type=RelationType.CALLS,
                            strength=RelationshipStrength.MODERATE.value,
                            confidence=RelationshipConfidence.MEDIUM.value,
                            metadata={'function': call_name},
                            evidence=[f"Calls {call_name}"]
                        )
                        relationships.append(rel)
                        self.stats['call_relationships'] += 1
        
        return relationships
    
    def _extract_function_calls(self, chunk: Chunk) -> List[str]:
        """Extract function call names from chunk"""
        calls = []
        
        if chunk.metadata.language == 'python':
            # Python function calls
            try:
                tree = ast.parse(chunk.content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            calls.append(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            calls.append(node.func.attr)
            except:
                # Fallback to regex
                call_pattern = re.compile(r'(\w+)\s*\(')
                for match in call_pattern.finditer(chunk.content):
                    calls.append(match.group(1))
        
        elif chunk.metadata.language in ['javascript', 'typescript']:
            # JavaScript function calls
            call_patterns = [
                re.compile(r'(\w+)\s*\('),  # Regular calls
                re.compile(r'\.(\w+)\s*\('),  # Method calls
                re.compile(r'await\s+(\w+)\s*\('),  # Async calls
            ]
            for pattern in call_patterns:
                for match in pattern.finditer(chunk.content):
                    calls.append(match.group(1))
        
        elif chunk.metadata.language == 'java':
            # Java method calls
            call_patterns = [
                re.compile(r'\.(\w+)\s*\('),  # Method calls
                re.compile(r'new\s+(\w+)\s*\('),  # Constructor calls
            ]
            for pattern in call_patterns:
                for match in pattern.finditer(chunk.content):
                    calls.append(match.group(1))
        
        return list(set(calls))  # Remove duplicates
    
    def _find_function_chunks(self, function_name: str, chunks: List[Chunk]) -> List[Chunk]:
        """Find chunks that define a function"""
        matching = []
        
        for chunk in chunks:
            if chunk.chunk_type.value in ['function', 'method']:
                # Check signature
                if hasattr(chunk, 'signature') and chunk.signature:
                    if chunk.signature.name == function_name:
                        matching.append(chunk)
                        continue
            
            # Check content for function definition
            if self._contains_function_definition(chunk.content, function_name, chunk.metadata.language):
                matching.append(chunk)
        
        return matching
    
    def _contains_function_definition(self, content: str, function_name: str, language: Optional[str]) -> bool:
        """Check if content contains function definition"""
        if not language:
            return False
        
        patterns = {
            'python': rf'^\s*def\s+{function_name}\s*\(',
            'javascript': rf'function\s+{function_name}\s*\(|const\s+{function_name}\s*=',
            'java': rf'(?:public|private|protected)?\s+\w+\s+{function_name}\s*\(',
            'go': rf'func\s+(?:\(\w+\s+\*?\w+\)\s+)?{function_name}\s*\(',
            'rust': rf'fn\s+{function_name}\s*\(',
            'cpp': rf'\w+\s+{function_name}\s*\(',
        }
        
        pattern = patterns.get(language)
        if pattern:
            return bool(re.search(pattern, content, re.MULTILINE))
        
        return False
    
    def _analyze_inheritance(self, chunks: List[Chunk], show_progress: bool) -> List[RelationshipEdge]:
        """Analyze class inheritance relationships"""
        logger.info("Analyzing inheritance relationships")
        relationships = []
        
        if show_progress:
            chunks = tqdm(chunks, desc="Analyzing inheritance")
        
        for chunk in chunks:
            if chunk.chunk_type.value not in ['class', 'interface', 'trait']:
                continue
            
            # Extract parent classes
            parents = self._extract_parent_classes(chunk)
            
            # Find parent chunks
            for parent_name in parents:
                parent_chunks = self._find_class_chunks(parent_name, chunks)
                
                for parent_chunk in parent_chunks:
                    if parent_chunk.id != chunk.id:
                        rel = RelationshipEdge(
                            source_id=chunk.id,
                            target_id=parent_chunk.id,
                            relation_type=RelationType.INHERITS,
                            strength=RelationshipStrength.VERY_STRONG.value,
                            confidence=RelationshipConfidence.CERTAIN.value,
                            metadata={'parent_class': parent_name},
                            evidence=[f"Inherits from {parent_name}"]
                        )
                        relationships.append(rel)
                        self.stats['inheritance_relationships'] += 1
        
        return relationships
    
    def _extract_parent_classes(self, chunk: Chunk) -> List[str]:
        """Extract parent class names from chunk"""
        parents = []
        
        if chunk.metadata.language == 'python':
            # Python inheritance
            class_pattern = re.compile(r'class\s+\w+\s*\(([^)]+)\)')
            for match in class_pattern.finditer(chunk.content):
                parent_list = match.group(1)
                for parent in parent_list.split(','):
                    parent = parent.strip()
                    if parent and parent != 'object':
                        parents.append(parent)
        
        elif chunk.metadata.language == 'java':
            # Java inheritance
            extends_pattern = re.compile(r'extends\s+(\w+)')
            implements_pattern = re.compile(r'implements\s+([\w\s,]+)')
            
            for match in extends_pattern.finditer(chunk.content):
                parents.append(match.group(1))
            
            for match in implements_pattern.finditer(chunk.content):
                interfaces = match.group(1)
                for interface in interfaces.split(','):
                    parents.append(interface.strip())
        
        elif chunk.metadata.language in ['javascript', 'typescript']:
            # JavaScript/TypeScript inheritance
            extends_pattern = re.compile(r'class\s+\w+\s+extends\s+(\w+)')
            for match in extends_pattern.finditer(chunk.content):
                parents.append(match.group(1))
        
        elif chunk.metadata.language == 'cpp':
            # C++ inheritance
            inherit_pattern = re.compile(r'class\s+\w+\s*:\s*(?:public|private|protected)?\s*(\w+)')
            for match in inherit_pattern.finditer(chunk.content):
                parents.append(match.group(1))
        
        return parents
    
    def _find_class_chunks(self, class_name: str, chunks: List[Chunk]) -> List[Chunk]:
        """Find chunks that define a class"""
        matching = []
        
        for chunk in chunks:
            if chunk.chunk_type.value in ['class', 'interface', 'trait']:
                # Check signature
                if hasattr(chunk, 'signature') and chunk.signature:
                    if chunk.signature.name == class_name:
                        matching.append(chunk)
                        continue
            
            # Check content for class definition
            if self._contains_class_definition(chunk.content, class_name, chunk.metadata.language):
                matching.append(chunk)
        
        return matching
    
    def _contains_class_definition(self, content: str, class_name: str, language: Optional[str]) -> bool:
        """Check if content contains class definition"""
        if not language:
            return False
        
        patterns = {
            'python': rf'^\s*class\s+{class_name}\s*[\(:]',
            'java': rf'(?:public\s+)?class\s+{class_name}\s*[{{<]',
            'javascript': rf'class\s+{class_name}\s*[{{]',
            'typescript': rf'(?:export\s+)?class\s+{class_name}\s*[{{<]',
            'cpp': rf'class\s+{class_name}\s*[{{:]',
            'csharp': rf'(?:public\s+)?class\s+{class_name}\s*[{{:]',
        }
        
        pattern = patterns.get(language)
        if pattern:
            return bool(re.search(pattern, content, re.MULTILINE))
        
        return False
    
    def _analyze_similarity(self, chunks: List[Chunk], 
                          storage: EmbeddingStorage,
                          show_progress: bool) -> List[RelationshipEdge]:
        """Analyze similarity-based relationships"""
        logger.info("Analyzing similarity relationships")
        relationships = []
        
        # Get embeddings for all chunks
        embeddings = {}
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings[chunk.id] = chunk.embedding
            else:
                # Try to retrieve from storage
                embedding, _ = storage.retrieve(chunk.id)
                if embedding is not None:
                    embeddings[chunk.id] = embedding
        
        # Calculate pairwise similarities
        chunk_ids = list(embeddings.keys())
        
        if show_progress:
            total_pairs = len(chunk_ids) * (len(chunk_ids) - 1) // 2
            pbar = tqdm(total=total_pairs, desc="Calculating similarities")
        
        for i, chunk_id1 in enumerate(chunk_ids):
            for chunk_id2 in chunk_ids[i+1:]:
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(
                    embeddings[chunk_id1],
                    embeddings[chunk_id2]
                )
                
                if similarity >= self.config.similarity_threshold:
                    rel = RelationshipEdge(
                        source_id=chunk_id1,
                        target_id=chunk_id2,
                        relation_type=RelationType.SIMILAR_TO,
                        strength=similarity,
                        confidence=RelationshipConfidence.HIGH.value,
                        bidirectional=True,
                        metadata={'similarity_score': similarity},
                        evidence=[f"Similarity score: {similarity:.3f}"]
                    )
                    relationships.append(rel)
                    self.stats['similarity_relationships'] += 1
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return relationships
    
    def _calculate_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _analyze_semantic(self, chunks: List[Chunk], show_progress: bool) -> List[RelationshipEdge]:
        """Analyze semantic relationships"""
        logger.info("Analyzing semantic relationships")
        relationships = []
        
        if show_progress:
            chunks = tqdm(chunks, desc="Analyzing semantics")
        
        # Group chunks by semantic categories
        semantic_groups = defaultdict(list)
        
        for chunk in chunks:
            # Extract semantic features
            features = self._extract_semantic_features(chunk)
            
            # Categorize chunk
            for category in features['categories']:
                semantic_groups[category].append(chunk)
        
        # Create relationships within semantic groups
        for category, group_chunks in semantic_groups.items():
            if len(group_chunks) < 2:
                continue
            
            for i, chunk1 in enumerate(group_chunks):
                for chunk2 in group_chunks[i+1:]:
                    if chunk1.id != chunk2.id:
                        rel = RelationshipEdge(
                            source_id=chunk1.id,
                            target_id=chunk2.id,
                            relation_type=RelationType.RELATED_TO,
                            strength=RelationshipStrength.MODERATE.value,
                            confidence=RelationshipConfidence.MEDIUM.value,
                            bidirectional=True,
                            metadata={'semantic_category': category},
                            evidence=[f"Same semantic category: {category}"]
                        )
                        relationships.append(rel)
                        self.stats['semantic_relationships'] += 1
        
        return relationships
    
    def _extract_semantic_features(self, chunk: Chunk) -> Dict[str, Any]:
        """Extract semantic features from chunk"""
        features = {
            'categories': [],
            'concepts': [],
            'patterns': []
        }
        
        # Categorize by chunk type and content
        if chunk.chunk_type.value == 'class':
            # Determine class category
            if 'Controller' in chunk.content:
                features['categories'].append('controller')
            elif 'Service' in chunk.content:
                features['categories'].append('service')
            elif 'Model' in chunk.content or 'Entity' in chunk.content:
                features['categories'].append('model')
            elif 'Repository' in chunk.content or 'DAO' in chunk.content:
                features['categories'].append('repository')
            elif 'Test' in chunk.content:
                features['categories'].append('test')
        
        elif chunk.chunk_type.value == 'function':
            # Categorize functions
            if re.search(r'test_|_test|Test', chunk.content):
                features['categories'].append('test')
            elif re.search(r'get_|get[A-Z]', chunk.content):
                features['categories'].append('getter')
            elif re.search(r'set_|set[A-Z]', chunk.content):
                features['categories'].append('setter')
            elif re.search(r'validate|check|verify', chunk.content, re.IGNORECASE):
                features['categories'].append('validation')
            elif re.search(r'save|store|persist', chunk.content, re.IGNORECASE):
                features['categories'].append('persistence')
            elif re.search(r'load|fetch|retrieve|query', chunk.content, re.IGNORECASE):
                features['categories'].append('data_access')
        
        # Extract concepts from comments and docstrings
        comment_pattern = re.compile(r'(?:#|//|/\*|\*)\s*(.+)')
        for match in comment_pattern.finditer(chunk.content):
            comment = match.group(1).lower()
            if 'todo' in comment:
                features['concepts'].append('todo')
            if 'fixme' in comment:
                features['concepts'].append('fixme')
            if 'deprecated' in comment:
                features['concepts'].append('deprecated')
        
        # Detect design patterns
        if re.search(r'singleton|getInstance', chunk.content, re.IGNORECASE):
            features['patterns'].append('singleton')
        if re.search(r'factory|create\w+', chunk.content):
            features['patterns'].append('factory')
        if re.search(r'observer|listener|subscribe', chunk.content, re.IGNORECASE):
            features['patterns'].append('observer')
        
        return features
    
    def _add_relationships_to_graph(self, relationships: List[RelationshipEdge]):
        """Add relationships to the graph"""
        for rel in relationships:
            # Add edge to graph
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                relation_type=rel.relation_type.value,
                weight=rel.strength,
                confidence=rel.confidence,
                metadata=rel.metadata
            )
            
            # Add reverse edge if bidirectional
            if rel.bidirectional:
                self.graph.add_edge(
                    rel.target_id,
                    rel.source_id,
                    relation_type=rel.relation_type.value,
                    weight=rel.strength,
                    confidence=rel.confidence,
                    metadata=rel.metadata
                )
    
    def _post_process_graph(self):
        """Post-process the relationship graph"""
        # Remove weak relationships
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('weight', 0) < self.config.min_relationship_strength:
                edges_to_remove.append((u, v))
        
        for edge in edges_to_remove:
            self.graph.remove_edge(*edge)
        
        # Limit relationships per chunk
        for node in self.graph.nodes():
            edges = list(self.graph.edges(node, data=True))
            if len(edges) > self.config.max_relationships_per_chunk:
                # Keep only strongest relationships
                edges.sort(key=lambda x: x[2].get('weight', 0), reverse=True)
                edges_to_remove = edges[self.config.max_relationships_per_chunk:]
                
                for u, v, _ in edges_to_remove:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)
        
        # Transitive reduction if enabled
        if self.config.enable_transitive_reduction:
            self._perform_transitive_reduction()
    
    def _perform_transitive_reduction(self):
        """Perform transitive reduction on the graph"""
        try:
            # Only for DAGs
            if nx.is_directed_acyclic_graph(self.graph):
                reduced = nx.transitive_reduction(self.graph)
                # Preserve edge attributes
                for u, v in reduced.edges():
                    if self.graph.has_edge(u, v):
                        reduced[u][v].update(self.graph[u][v])
                self.graph = reduced
        except:
            # Graph has cycles, skip reduction
            pass
    
    def _detect_communities(self) -> Dict[int, List[str]]:
        """Detect communities in the graph"""
        logger.info("Detecting communities")
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Use Louvain community detection
        communities = self.community_detector.detect_louvain(undirected)
        
        # Convert to dict format
        community_dict = defaultdict(list)
        for node, community_id in communities.items():
            community_dict[community_id].append(node)
        
        logger.info(f"Detected {len(community_dict)} communities")
        
        return dict(community_dict)
    
    def _build_hierarchy(self) -> Dict[str, List[str]]:
        """Build hierarchy from relationships"""
        hierarchy = defaultdict(list)
        
        # Use inheritance and parent-child relationships
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relation_type')
            if rel_type in ['inherits', 'parent', 'extends']:
                hierarchy[v].append(u)  # v is parent of u
        
        return dict(hierarchy)
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate graph statistics"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'relationship_counts': Counter()
        }
        
        # Count relationship types
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relation_type', 'unknown')
            stats['relationship_counts'][rel_type] += 1
        
        # Page rank for importance
        try:
            page_rank = nx.pagerank(self.graph, max_iter=100)
            top_nodes = sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['most_important_chunks'] = top_nodes
        except:
            stats['most_important_chunks'] = []
        
        # Centrality measures
        try:
            stats['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
            stats['closeness_centrality'] = nx.closeness_centrality(self.graph)
        except:
            pass
        
        return stats
    
    def save_graph(self, path: Path, format: str = "graphml"):
        """Save the relationship graph to file"""
        if format == "graphml":
            nx.write_graphml(self.graph, path)
        elif format == "json":
            data = nx.node_link_data(self.graph)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Graph saved to {path}")
    
    def load_graph(self, path: Path, format: str = "graphml"):
        """Load relationship graph from file"""
        if format == "graphml":
            self.graph = nx.read_graphml(path)
        elif format == "json":
            with open(path, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
        elif format == "pickle":
            with open(path, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Graph loaded from {path}")
    
    def find_related_chunks(self, chunk_id: str, 
                          max_hops: int = 2,
                          relation_types: Optional[List[RelationType]] = None) -> List[Tuple[str, int, float]]:
        """
        Find related chunks within specified hops
        
        Args:
            chunk_id: Starting chunk ID
            max_hops: Maximum distance in graph
            relation_types: Filter by relationship types
            
        Returns:
            List of (chunk_id, distance, strength) tuples
        """
        if chunk_id not in self.graph:
            return []
        
        related = []
        visited = {chunk_id}
        current_level = [(chunk_id, 0, 1.0)]
        
        while current_level and current_level[0][1] < max_hops:
            next_level = []
            
            for current_id, distance, strength in current_level:
                # Get neighbors
                for neighbor in self.graph.neighbors(current_id):
                    if neighbor not in visited:
                        edge_data = self.graph[current_id][neighbor]
                        
                        # Filter by relation type if specified
                        if relation_types:
                            edge_type = edge_data.get('relation_type')
                            if edge_type not in [rt.value for rt in relation_types]:
                                continue
                        
                        # Calculate cumulative strength
                        edge_strength = edge_data.get('weight', 1.0)
                        cumulative_strength = strength * edge_strength
                        
                        visited.add(neighbor)
                        next_level.append((neighbor, distance + 1, cumulative_strength))
                        related.append((neighbor, distance + 1, cumulative_strength))
            
            current_level = next_level
        
        # Sort by distance and strength
        related.sort(key=lambda x: (x[1], -x[2]))
        
        return related
    
    def get_chunk_importance(self, chunk_id: str) -> float:
        """Get importance score for a chunk"""
        if chunk_id not in self.graph:
            return 0.0
        
        # Use PageRank as importance measure
        try:
            page_rank = nx.pagerank(self.graph, max_iter=100)
            return page_rank.get(chunk_id, 0.0)
        except:
            # Fallback to degree centrality
            degree = self.graph.degree(chunk_id)
            max_degree = max(dict(self.graph.degree()).values()) if self.graph.number_of_nodes() > 0 else 1
            return degree / max_degree if max_degree > 0 else 0.0