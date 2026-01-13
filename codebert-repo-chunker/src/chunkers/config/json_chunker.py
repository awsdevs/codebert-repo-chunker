"""
JSON-specific chunker for intelligent semantic chunking
Handles complex nested structures, JSON-LD, GeoJSON, JSON Schema, and large JSON files
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from enum import Enum
from decimal import Decimal
import math

from src.core.base_chunker import BaseChunker, Chunk
from src.core.file_context import FileContext
from config.settings import settings

logger = logging.getLogger(__name__)

class JSONType(Enum):
    """Types of JSON structures"""
    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    NULL = "null"
    NESTED_OBJECT = "nested_object"
    NESTED_ARRAY = "nested_array"

class JSONFormat(Enum):
    """Special JSON formats"""
    STANDARD = "standard"
    JSON_LD = "json_ld"
    GEOJSON = "geojson"
    JSON_SCHEMA = "json_schema"
    OPENAPI = "openapi"
    PACKAGE_JSON = "package_json"
    TSCONFIG = "tsconfig"
    COMPOSER_JSON = "composer_json"
    SWAGGER = "swagger"
    POSTMAN = "postman"
    GRAPHQL_SCHEMA = "graphql_schema"
    AWS_CLOUDFORMATION = "cloudformation"
    AZURE_ARM = "arm_template"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"
    ELASTICSEARCH = "elasticsearch"
    MONGODB = "mongodb"

@dataclass
class JSONNode:
    """Represents a node in JSON structure"""
    path: str
    key: Optional[str]
    value: Any
    node_type: JSONType
    depth: int
    size: int  # Size in characters
    token_count: int
    parent: Optional['JSONNode'] = None
    children: List['JSONNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_full_path(self) -> str:
        """Get full JSONPath to this node"""
        if self.parent:
            parent_path = self.parent.get_full_path()
            if self.key is not None:
                if self.parent.node_type == JSONType.ARRAY:
                    return f"{parent_path}[{self.key}]"
                else:
                    return f"{parent_path}.{self.key}" if parent_path != "$" else f"$.{self.key}"
            return parent_path
        return "$"

@dataclass
class JSONStructure:
    """Represents overall JSON structure analysis"""
    format: JSONFormat
    root_type: JSONType
    total_size: int
    total_tokens: int
    max_depth: int
    node_count: int
    array_count: int
    object_count: int
    leaf_count: int
    schema_detected: bool
    has_references: bool
    has_circular_refs: bool
    key_patterns: Dict[str, int]
    value_types: Dict[str, int]
    special_fields: Dict[str, Any]
    statistics: Dict[str, Any]

class JSONAnalyzer:
    """Analyzes JSON structure for intelligent chunking"""
    
    # Special field patterns to recognize
    SPECIAL_FIELDS = {
        # JSON-LD
        '@context', '@id', '@type', '@graph', '@value', '@language',
        
        # JSON Schema
        '$schema', '$ref', '$id', 'definitions', 'properties', 'required',
        'type', 'enum', 'allOf', 'anyOf', 'oneOf',
        
        # OpenAPI/Swagger
        'openapi', 'swagger', 'paths', 'components', 'schemas', 'parameters',
        'responses', 'requestBody', 'servers', 'security',
        
        # GeoJSON
        'type', 'geometry', 'coordinates', 'features', 'properties',
        
        # Package.json
        'name', 'version', 'dependencies', 'devDependencies', 'scripts',
        'main', 'module', 'types', 'repository', 'author', 'license',
        
        # Kubernetes
        'apiVersion', 'kind', 'metadata', 'spec', 'status',
        
        # Docker Compose
        'version', 'services', 'networks', 'volumes', 'configs', 'secrets',
        
        # CloudFormation
        'AWSTemplateFormatVersion', 'Resources', 'Parameters', 'Outputs',
        
        # Common patterns
        'id', 'uuid', 'guid', 'timestamp', 'created_at', 'updated_at',
        'url', 'uri', 'href', 'links', 'meta', 'data', 'items', 'results'
    }
    
    # Format detection patterns
    FORMAT_PATTERNS = {
        JSONFormat.JSON_LD: ['@context', '@graph', '@id', '@type'],
        JSONFormat.GEOJSON: ['type', 'geometry', 'coordinates', 'features'],
        JSONFormat.JSON_SCHEMA: ['$schema', 'definitions', 'properties'],
        JSONFormat.OPENAPI: ['openapi', 'paths', 'components'],
        JSONFormat.SWAGGER: ['swagger', 'paths', 'definitions'],
        JSONFormat.PACKAGE_JSON: ['name', 'version', 'dependencies', 'scripts'],
        JSONFormat.TSCONFIG: ['compilerOptions', 'include', 'exclude'],
        JSONFormat.COMPOSER_JSON: ['name', 'require', 'autoload'],
        JSONFormat.POSTMAN: ['info', 'item', 'request', 'response'],
        JSONFormat.AWS_CLOUDFORMATION: ['AWSTemplateFormatVersion', 'Resources'],
        JSONFormat.KUBERNETES: ['apiVersion', 'kind', 'metadata', 'spec'],
        JSONFormat.DOCKER_COMPOSE: ['version', 'services', 'networks'],
        JSONFormat.ELASTICSEARCH: ['mappings', 'settings', 'index'],
        JSONFormat.MONGODB: ['_id', '$set', '$push', '$pull']
    }
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.nodes = []
        self.references = {}  # Track $ref references
        self.circular_refs = set()
        
    def analyze_json(self, data: Union[str, dict, list], 
                    file_path: Optional[Path] = None) -> JSONStructure:
        """
        Analyze JSON structure
        
        Args:
            data: JSON string or parsed JSON object
            file_path: Optional file path for format detection
            
        Returns:
            JSONStructure analysis
        """
        # Parse if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                return self._create_error_structure(str(e))
        else:
            parsed_data = data
        
        # Detect format
        json_format = self._detect_format(parsed_data, file_path)
        
        # Build node tree
        root_node = self._build_node_tree(parsed_data, "$", None, 0)
        
        # Analyze structure
        structure = JSONStructure(
            format=json_format,
            root_type=root_node.node_type,
            total_size=root_node.size,
            total_tokens=root_node.token_count,
            max_depth=self._calculate_max_depth(root_node),
            node_count=self._count_nodes(root_node),
            array_count=self._count_type(root_node, JSONType.ARRAY),
            object_count=self._count_type(root_node, JSONType.OBJECT),
            leaf_count=self._count_leaves(root_node),
            schema_detected=json_format in [
                JSONFormat.JSON_SCHEMA, JSONFormat.OPENAPI, JSONFormat.SWAGGER
            ],
            has_references=bool(self.references),
            has_circular_refs=bool(self.circular_refs),
            key_patterns=self._analyze_key_patterns(root_node),
            value_types=self._analyze_value_types(root_node),
            special_fields=self._extract_special_fields(parsed_data),
            statistics=self._calculate_statistics(root_node)
        )
        
        # Store root for chunking
        self.root_node = root_node
        
        return structure
    
    def _detect_format(self, data: Any, file_path: Optional[Path]) -> JSONFormat:
        """Detect JSON format/schema type"""
        # Check filename hints
        if file_path:
            name = file_path.name.lower()
            if name == 'package.json':
                return JSONFormat.PACKAGE_JSON
            elif name == 'composer.json':
                return JSONFormat.COMPOSER_JSON
            elif name == 'tsconfig.json':
                return JSONFormat.TSCONFIG
            elif 'openapi' in name or 'swagger' in name:
                return JSONFormat.OPENAPI
            elif 'schema' in name:
                return JSONFormat.JSON_SCHEMA
            elif name.endswith('.geojson'):
                return JSONFormat.GEOJSON
        
        # Check content patterns
        if isinstance(data, dict):
            keys = set(data.keys())
            
            for format_type, patterns in self.FORMAT_PATTERNS.items():
                if any(pattern in keys for pattern in patterns):
                    # Additional validation for specific formats
                    if format_type == JSONFormat.GEOJSON:
                        if data.get('type') in ['Feature', 'FeatureCollection', 'Point', 
                                                'LineString', 'Polygon', 'MultiPoint',
                                                'MultiLineString', 'MultiPolygon']:
                            return JSONFormat.GEOJSON
                    elif format_type == JSONFormat.JSON_SCHEMA:
                        if '$schema' in data or 'definitions' in data:
                            return JSONFormat.JSON_SCHEMA
                    else:
                        return format_type
        
        return JSONFormat.STANDARD
    
    def _build_node_tree(self, data: Any, path: str, parent: Optional[JSONNode], 
                        depth: int) -> JSONNode:
        """Build tree of JSON nodes"""
        # Determine node type
        if isinstance(data, dict):
            node_type = JSONType.NESTED_OBJECT if depth > 0 else JSONType.OBJECT
        elif isinstance(data, list):
            node_type = JSONType.NESTED_ARRAY if depth > 0 else JSONType.ARRAY
        elif isinstance(data, str):
            node_type = JSONType.STRING
        elif isinstance(data, (int, float)):
            node_type = JSONType.NUMBER
        elif isinstance(data, bool):
            node_type = JSONType.BOOLEAN
        elif data is None:
            node_type = JSONType.NULL
        else:
            node_type = JSONType.STRING  # Fallback
        
        # Calculate size and tokens
        data_str = json.dumps(data, separators=(',', ':'))
        size = len(data_str)
        token_count = self._count_tokens(data_str) if self.tokenizer else size // 4
        
        # Create node
        node = JSONNode(
            path=path,
            key=path.split('.')[-1] if '.' in path else None,
            value=data,
            node_type=node_type,
            depth=depth,
            size=size,
            token_count=token_count,
            parent=parent
        )
        
        # Add metadata
        if node_type in [JSONType.OBJECT, JSONType.NESTED_OBJECT]:
            node.metadata['key_count'] = len(data)
            node.metadata['keys'] = list(data.keys())
        elif node_type in [JSONType.ARRAY, JSONType.NESTED_ARRAY]:
            node.metadata['length'] = len(data)
        
        # Process children
        if isinstance(data, dict):
            for key, value in data.items():
                child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
                
                # Check for $ref
                if key == '$ref':
                    self.references[child_path] = value
                    # Check for circular reference
                    if self._is_circular_ref(value, path):
                        self.circular_refs.add((path, value))
                
                child_node = self._build_node_tree(value, child_path, node, depth + 1)
                child_node.key = key
                node.children.append(child_node)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                child_path = f"{path}[{i}]"
                child_node = self._build_node_tree(item, child_path, node, depth + 1)
                child_node.key = str(i)
                node.children.append(child_node)
        
        self.nodes.append(node)
        return node
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback estimation
        return len(text.split()) + text.count(',') + text.count(':')
    
    def _is_circular_ref(self, ref: str, current_path: str) -> bool:
        """Check if reference creates circular dependency"""
        # Simple check - would need more sophisticated analysis for complex schemas
        return ref.startswith('#') and ref in current_path
    
    def _calculate_max_depth(self, node: JSONNode) -> int:
        """Calculate maximum depth of tree"""
        if not node.children:
            return node.depth
        return max(self._calculate_max_depth(child) for child in node.children)
    
    def _count_nodes(self, node: JSONNode) -> int:
        """Count total nodes in tree"""
        return 1 + sum(self._count_nodes(child) for child in node.children)
    
    def _count_type(self, node: JSONNode, node_type: JSONType) -> int:
        """Count nodes of specific type"""
        count = 1 if node.node_type == node_type else 0
        return count + sum(self._count_type(child, node_type) for child in node.children)
    
    def _count_leaves(self, node: JSONNode) -> int:
        """Count leaf nodes"""
        if not node.children:
            return 1
        return sum(self._count_leaves(child) for child in node.children)
    
    def _analyze_key_patterns(self, node: JSONNode) -> Dict[str, int]:
        """Analyze patterns in object keys"""
        patterns = defaultdict(int)
        
        def collect_keys(n: JSONNode):
            if n.node_type in [JSONType.OBJECT, JSONType.NESTED_OBJECT]:
                for key in n.metadata.get('keys', []):
                    patterns[key] += 1
            for child in n.children:
                collect_keys(child)
        
        collect_keys(node)
        return dict(patterns)
    
    def _analyze_value_types(self, node: JSONNode) -> Dict[str, int]:
        """Analyze distribution of value types"""
        type_counts = defaultdict(int)
        
        def count_types(n: JSONNode):
            type_counts[n.node_type.value] += 1
            for child in n.children:
                count_types(child)
        
        count_types(node)
        return dict(type_counts)
    
    def _extract_special_fields(self, data: Any) -> Dict[str, Any]:
        """Extract special fields for format-specific handling"""
        special = {}
        
        if isinstance(data, dict):
            for field in self.SPECIAL_FIELDS:
                if field in data:
                    special[field] = data[field]
        
        return special
    
    def _calculate_statistics(self, node: JSONNode) -> Dict[str, Any]:
        """Calculate statistical information about JSON structure"""
        stats = {
            'avg_object_size': 0,
            'avg_array_length': 0,
            'max_string_length': 0,
            'max_number_value': None,
            'min_number_value': None,
            'unique_keys': set(),
            'null_count': 0,
            'boolean_true_count': 0,
            'boolean_false_count': 0
        }
        
        object_sizes = []
        array_lengths = []
        string_lengths = []
        numbers = []
        
        def analyze(n: JSONNode):
            if n.node_type in [JSONType.OBJECT, JSONType.NESTED_OBJECT]:
                object_sizes.append(n.metadata.get('key_count', 0))
                stats['unique_keys'].update(n.metadata.get('keys', []))
            elif n.node_type in [JSONType.ARRAY, JSONType.NESTED_ARRAY]:
                array_lengths.append(n.metadata.get('length', 0))
            elif n.node_type == JSONType.STRING:
                if isinstance(n.value, str):
                    string_lengths.append(len(n.value))
            elif n.node_type == JSONType.NUMBER:
                if isinstance(n.value, (int, float)):
                    numbers.append(n.value)
            elif n.node_type == JSONType.NULL:
                stats['null_count'] += 1
            elif n.node_type == JSONType.BOOLEAN:
                if n.value is True:
                    stats['boolean_true_count'] += 1
                else:
                    stats['boolean_false_count'] += 1
            
            for child in n.children:
                analyze(child)
        
        analyze(node)
        
        # Calculate averages
        if object_sizes:
            stats['avg_object_size'] = sum(object_sizes) / len(object_sizes)
        if array_lengths:
            stats['avg_array_length'] = sum(array_lengths) / len(array_lengths)
        if string_lengths:
            stats['max_string_length'] = max(string_lengths)
        if numbers:
            stats['max_number_value'] = max(numbers)
            stats['min_number_value'] = min(numbers)
        
        stats['unique_keys'] = list(stats['unique_keys'])
        
        return stats
    
    def _create_error_structure(self, error_msg: str) -> JSONStructure:
        """Create error structure for invalid JSON"""
        return JSONStructure(
            format=JSONFormat.STANDARD,
            root_type=JSONType.NULL,
            total_size=0,
            total_tokens=0,
            max_depth=0,
            node_count=0,
            array_count=0,
            object_count=0,
            leaf_count=0,
            schema_detected=False,
            has_references=False,
            has_circular_refs=False,
            key_patterns={},
            value_types={},
            special_fields={},
            statistics={'error': error_msg}
        )

class ChunkingStrategy:
    """Base class for JSON chunking strategies"""
    
    def chunk(self, node: JSONNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Create chunks from JSON node"""
        raise NotImplementedError

class ObjectChunkingStrategy(ChunkingStrategy):
    """Strategy for chunking JSON objects"""
    
    def chunk(self, node: JSONNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk object by keys"""
        chunks = []
        
        if node.node_type not in [JSONType.OBJECT, JSONType.NESTED_OBJECT]:
            return [{'path': node.path, 'data': node.value}]
        
        # If object fits in one chunk
        if node.token_count <= max_tokens:
            return [{'path': node.path, 'data': node.value}]
        
        # Group keys by size
        key_groups = self._group_keys_by_size(node, max_tokens)
        
        for group in key_groups:
            chunk_data = {key: node.value[key] for key in group['keys']}
            chunks.append({
                'path': node.path,
                'data': chunk_data,
                'partial': True,
                'keys': group['keys']
            })
        
        return chunks
    
    def _group_keys_by_size(self, node: JSONNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Group object keys to fit in token limit"""
        groups = []
        current_group = {'keys': [], 'tokens': 0}
        
        for child in node.children:
            if current_group['tokens'] + child.token_count > max_tokens:
                if current_group['keys']:
                    groups.append(current_group)
                current_group = {'keys': [child.key], 'tokens': child.token_count}
            else:
                current_group['keys'].append(child.key)
                current_group['tokens'] += child.token_count
        
        if current_group['keys']:
            groups.append(current_group)
        
        return groups

class ArrayChunkingStrategy(ChunkingStrategy):
    """Strategy for chunking JSON arrays"""
    
    def chunk(self, node: JSONNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk array by elements"""
        chunks = []
        
        if node.node_type not in [JSONType.ARRAY, JSONType.NESTED_ARRAY]:
            return [{'path': node.path, 'data': node.value}]
        
        # If array fits in one chunk
        if node.token_count <= max_tokens:
            return [{'path': node.path, 'data': node.value}]
        
        # Group elements by size
        element_groups = self._group_elements_by_size(node, max_tokens)
        
        for i, group in enumerate(element_groups):
            chunks.append({
                'path': f"{node.path}[batch_{i}]",
                'data': group['elements'],
                'partial': True,
                'range': group['range']
            })
        
        return chunks
    
    def _group_elements_by_size(self, node: JSONNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Group array elements to fit in token limit"""
        groups = []
        current_group = {'elements': [], 'tokens': 0, 'range': [0, 0]}
        start_index = 0
        
        for i, child in enumerate(node.children):
            if current_group['tokens'] + child.token_count > max_tokens:
                if current_group['elements']:
                    current_group['range'] = [start_index, i - 1]
                    groups.append(current_group)
                current_group = {
                    'elements': [child.value], 
                    'tokens': child.token_count,
                    'range': [i, i]
                }
                start_index = i
            else:
                current_group['elements'].append(child.value)
                current_group['tokens'] += child.token_count
                current_group['range'][1] = i
        
        if current_group['elements']:
            groups.append(current_group)
        
        return groups

class SchemaChunkingStrategy(ChunkingStrategy):
    """Strategy for chunking JSON Schema and OpenAPI specs"""
    
    def chunk(self, node: JSONNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk schema by definitions and paths"""
        chunks = []
        
        # Handle different schema sections
        if isinstance(node.value, dict):
            # Extract main schema info
            main_schema = {
                k: v for k, v in node.value.items()
                if k in ['$schema', 'title', 'description', 'version', 'openapi', 'info']
            }
            
            if main_schema:
                chunks.append({
                    'path': node.path,
                    'data': main_schema,
                    'type': 'schema_metadata'
                })
            
            # Handle definitions/components
            if 'definitions' in node.value:
                def_chunks = self._chunk_definitions(
                    node.value['definitions'], 
                    f"{node.path}.definitions",
                    max_tokens
                )
                chunks.extend(def_chunks)
            
            if 'components' in node.value:
                comp_chunks = self._chunk_components(
                    node.value['components'],
                    f"{node.path}.components",
                    max_tokens
                )
                chunks.extend(comp_chunks)
            
            # Handle paths (OpenAPI)
            if 'paths' in node.value:
                path_chunks = self._chunk_paths(
                    node.value['paths'],
                    f"{node.path}.paths",
                    max_tokens
                )
                chunks.extend(path_chunks)
        
        return chunks if chunks else [{'path': node.path, 'data': node.value}]
    
    def _chunk_definitions(self, definitions: Dict, base_path: str, 
                         max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk schema definitions"""
        chunks = []
        
        for name, schema in definitions.items():
            schema_str = json.dumps(schema)
            if len(schema_str) // 4 <= max_tokens:  # Simple estimation
                chunks.append({
                    'path': f"{base_path}.{name}",
                    'data': {name: schema},
                    'type': 'definition'
                })
            else:
                # Further split large definitions
                if isinstance(schema, dict) and 'properties' in schema:
                    # Split by properties
                    prop_chunks = self._chunk_properties(
                        schema['properties'],
                        f"{base_path}.{name}.properties",
                        max_tokens
                    )
                    chunks.extend(prop_chunks)
                else:
                    chunks.append({
                        'path': f"{base_path}.{name}",
                        'data': {name: schema},
                        'type': 'definition'
                    })
        
        return chunks
    
    def _chunk_components(self, components: Dict, base_path: str,
                        max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk OpenAPI components"""
        chunks = []
        
        for component_type, component_data in components.items():
            if isinstance(component_data, dict):
                for name, spec in component_data.items():
                    chunks.append({
                        'path': f"{base_path}.{component_type}.{name}",
                        'data': {name: spec},
                        'type': f'component_{component_type}'
                    })
        
        return chunks
    
    def _chunk_paths(self, paths: Dict, base_path: str,
                   max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk OpenAPI paths"""
        chunks = []
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                    chunks.append({
                        'path': f"{base_path}.{path}.{method}",
                        'data': {path: {method: spec}},
                        'type': 'api_endpoint'
                    })
        
        return chunks
    
    def _chunk_properties(self, properties: Dict, base_path: str,
                        max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk schema properties"""
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for prop_name, prop_schema in properties.items():
            prop_str = json.dumps(prop_schema)
            prop_size = len(prop_str) // 4
            
            if current_size + prop_size > max_tokens and current_chunk:
                chunks.append({
                    'path': base_path,
                    'data': current_chunk,
                    'type': 'properties',
                    'partial': True
                })
                current_chunk = {prop_name: prop_schema}
                current_size = prop_size
            else:
                current_chunk[prop_name] = prop_schema
                current_size += prop_size
        
        if current_chunk:
            chunks.append({
                'path': base_path,
                'data': current_chunk,
                'type': 'properties',
                'partial': len(chunks) > 0
            })
        
        return chunks

class JSONChunker(BaseChunker):
    """Chunker specialized for JSON files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, max_tokens)
        self.analyzer = JSONAnalyzer(tokenizer)
        
        # Chunking strategies
        self.strategies = {
            JSONType.OBJECT: ObjectChunkingStrategy(),
            JSONType.ARRAY: ArrayChunkingStrategy(),
            JSONType.NESTED_OBJECT: ObjectChunkingStrategy(),
            JSONType.NESTED_ARRAY: ArrayChunkingStrategy(),
        }
        
        self.schema_strategy = SchemaChunkingStrategy()
    
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from JSON file
        
        Args:
            content: JSON content as string
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze JSON structure
            structure = self.analyzer.analyze_json(content, file_context.path)
            
            # Choose chunking strategy based on format
            if structure.format in [JSONFormat.JSON_SCHEMA, JSONFormat.OPENAPI, 
                                   JSONFormat.SWAGGER]:
                return self._chunk_schema(content, structure, file_context)
            elif structure.format == JSONFormat.GEOJSON:
                return self._chunk_geojson(content, structure, file_context)
            elif structure.format == JSONFormat.PACKAGE_JSON:
                return self._chunk_package_json(content, structure, file_context)
            else:
                return self._chunk_standard(content, structure, file_context)
                
        except Exception as e:
            logger.error(f"Error chunking JSON file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _chunk_standard(self, content: str, structure: JSONStructure,
                       file_context: FileContext) -> List[Chunk]:
        """Chunk standard JSON"""
        chunks = []
        
        # Parse JSON
        data = json.loads(content)
        
        # If small enough, single chunk
        if structure.total_tokens <= self.max_tokens:
            return [self.create_chunk(
                content=content,
                chunk_type='json_complete',
                metadata={
                    'format': structure.format.value,
                    'root_type': structure.root_type.value,
                    'total_size': structure.total_size,
                    'node_count': structure.node_count
                },
                file_path=str(file_context.path)
            )]
        
        # Choose strategy based on root type
        strategy = self.strategies.get(structure.root_type)
        
        if strategy:
            chunk_data = strategy.chunk(self.analyzer.root_node, self.max_tokens)
            
            for i, chunk_info in enumerate(chunk_data):
                chunk_content = json.dumps(chunk_info['data'], indent=2)
                
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='json_partial',
                    metadata={
                        'format': structure.format.value,
                        'path': chunk_info['path'],
                        'partial': chunk_info.get('partial', False),
                        'chunk_index': i,
                        'total_chunks': len(chunk_data),
                        'keys': chunk_info.get('keys', []),
                        'range': chunk_info.get('range', None)
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks if chunks else self._fallback_chunking(content, file_context)
    
    def _chunk_schema(self, content: str, structure: JSONStructure,
                     file_context: FileContext) -> List[Chunk]:
        """Chunk JSON Schema or OpenAPI"""
        chunks = []
        data = json.loads(content)
        
        # Use schema-specific strategy
        chunk_data = self.schema_strategy.chunk(self.analyzer.root_node, self.max_tokens)
        
        for chunk_info in chunk_data:
            chunk_content = json.dumps(chunk_info['data'], indent=2)
            
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type=f"json_{chunk_info.get('type', 'schema')}",
                metadata={
                    'format': structure.format.value,
                    'path': chunk_info['path'],
                    'schema_type': chunk_info.get('type', 'unknown'),
                    'has_references': structure.has_references
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_geojson(self, content: str, structure: JSONStructure,
                      file_context: FileContext) -> List[Chunk]:
        """Chunk GeoJSON data"""
        chunks = []
        data = json.loads(content)
        
        # Handle FeatureCollection
        if data.get('type') == 'FeatureCollection':
            # Metadata chunk
            metadata = {
                'type': data.get('type'),
                'crs': data.get('crs'),
                'bbox': data.get('bbox'),
                'feature_count': len(data.get('features', []))
            }
            
            chunks.append(self.create_chunk(
                content=json.dumps(metadata, indent=2),
                chunk_type='geojson_metadata',
                metadata={
                    'format': 'geojson',
                    'feature_count': metadata['feature_count']
                },
                file_path=str(file_context.path)
            ))
            
            # Chunk features
            features = data.get('features', [])
            feature_chunks = self._chunk_array(features, 'features', self.max_tokens)
            
            for i, feature_batch in enumerate(feature_chunks):
                chunks.append(self.create_chunk(
                    content=json.dumps(feature_batch, indent=2),
                    chunk_type='geojson_features',
                    metadata={
                        'format': 'geojson',
                        'batch_index': i,
                        'feature_count': len(feature_batch)
                    },
                    file_path=str(file_context.path)
                ))
        else:
            # Single geometry
            chunks.append(self.create_chunk(
                content=content,
                chunk_type='geojson_geometry',
                metadata={
                    'format': 'geojson',
                    'geometry_type': data.get('type')
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_package_json(self, content: str, structure: JSONStructure,
                          file_context: FileContext) -> List[Chunk]:
        """Chunk package.json file"""
        chunks = []
        data = json.loads(content)
        
        # Main metadata
        main_info = {
            'name': data.get('name'),
            'version': data.get('version'),
            'description': data.get('description'),
            'main': data.get('main'),
            'scripts': data.get('scripts', {}),
            'author': data.get('author'),
            'license': data.get('license')
        }
        
        chunks.append(self.create_chunk(
            content=json.dumps(main_info, indent=2),
            chunk_type='package_json_main',
            metadata={
                'format': 'package_json',
                'package_name': data.get('name'),
                'version': data.get('version')
            },
            file_path=str(file_context.path)
        ))
        
        # Dependencies
        if 'dependencies' in data:
            chunks.append(self.create_chunk(
                content=json.dumps({'dependencies': data['dependencies']}, indent=2),
                chunk_type='package_json_dependencies',
                metadata={
                    'format': 'package_json',
                    'dependency_count': len(data['dependencies'])
                },
                file_path=str(file_context.path)
            ))
        
        # Dev dependencies
        if 'devDependencies' in data:
            chunks.append(self.create_chunk(
                content=json.dumps({'devDependencies': data['devDependencies']}, indent=2),
                chunk_type='package_json_dev_dependencies',
                metadata={
                    'format': 'package_json',
                    'dev_dependency_count': len(data['devDependencies'])
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_array(self, array: List, name: str, max_tokens: int) -> List[List]:
        """Chunk array into batches"""
        batches = []
        current_batch = []
        current_size = 0
        
        for item in array:
            item_str = json.dumps(item)
            item_size = self.count_tokens(item_str)
            
            if current_size + item_size > max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [item]
                current_size = item_size
            else:
                current_batch.append(item)
                current_size += item_size
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for invalid or complex JSON"""
        logger.warning(f"Using fallback chunking for JSON file {file_context.path}")
        
        chunks = []
        
        # Try to parse as JSONL (JSON Lines)
        if '\n' in content:
            lines = content.split('\n')
            jsonl_objects = []
            
            for line in lines:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        jsonl_objects.append(obj)
                    except:
                        pass
            
            if jsonl_objects:
                # It's JSONL format
                batches = self._chunk_array(jsonl_objects, 'jsonl', self.max_tokens)
                
                for i, batch in enumerate(batches):
                    chunks.append(self.create_chunk(
                        content='\n'.join(json.dumps(obj) for obj in batch),
                        chunk_type='jsonl_batch',
                        metadata={
                            'format': 'jsonl',
                            'batch_index': i,
                            'object_count': len(batch)
                        },
                        file_path=str(file_context.path)
                    ))
                
                return chunks
        
        # Fall back to string chunking
        lines = content.split('\n')
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='json_text',
                    metadata={
                        'format': 'unknown',
                        'is_fallback': True
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='json_text',
                metadata={
                    'format': 'unknown',
                    'is_fallback': True
                },
                file_path=str(file_context.path)
            ))
        
        return chunks