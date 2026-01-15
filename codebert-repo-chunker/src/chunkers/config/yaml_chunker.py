"""
YAML-specific chunker for intelligent semantic chunking
Handles complex YAML documents, Kubernetes manifests, Docker Compose, CI/CD configs, and more
"""

import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from enum import Enum
import json
from io import StringIO

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = logging.getLogger(__name__)

class YAMLFormat(Enum):
    """Types of YAML formats"""
    GENERIC = "generic"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    ANSIBLE_PLAYBOOK = "ansible_playbook"
    ANSIBLE_INVENTORY = "ansible_inventory"
    HELM_CHART = "helm_chart"
    HELM_VALUES = "helm_values"
    CLOUDFORMATION = "cloudformation"
    SERVERLESS = "serverless"
    TRAVIS_CI = "travis_ci"
    CIRCLE_CI = "circle_ci"
    AZURE_PIPELINES = "azure_pipelines"
    JENKINS_PIPELINE = "jenkins_pipeline"
    SWAGGER = "swagger"
    OPENAPI = "openapi"
    SALT_STATE = "salt_state"
    PUPPET = "puppet"
    CONDA_ENV = "conda_env"
    MKDOCS = "mkdocs"
    PRE_COMMIT = "pre_commit"
    POETRY = "poetry"
    RAILS_DATABASE = "rails_database"
    SPRING_BOOT = "spring_boot"

@dataclass
class YAMLNode:
    """Represents a YAML node in the document tree"""
    key: Optional[str]
    value: Any
    node_type: str  # 'scalar', 'list', 'dict', 'null'
    path: str
    depth: int
    line_start: int
    line_end: int
    parent: Optional['YAMLNode']
    children: List['YAMLNode']
    size: int  # Size in characters
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_full_path(self) -> str:
        """Get full path to this node"""
        if self.parent:
            parent_path = self.parent.get_full_path()
            if self.key:
                if parent_path == "$":
                    return f"$.{self.key}"
                elif self.parent.node_type == 'list':
                    return f"{parent_path}[{self.key}]"
                else:
                    return f"{parent_path}.{self.key}"
            return parent_path
        return "$"

@dataclass
class YAMLDocument:
    """Represents a single YAML document (for multi-document files)"""
    index: int
    content: Any
    start_line: int
    end_line: int
    metadata: Dict[str, Any]
    root_node: Optional[YAMLNode]

@dataclass
class YAMLStructure:
    """Represents overall YAML structure"""
    format: YAMLFormat
    documents: List[YAMLDocument]
    is_multi_document: bool
    total_nodes: int
    total_scalars: int
    total_lists: int
    total_dicts: int
    max_depth: int
    has_anchors: bool
    has_aliases: bool
    has_tags: bool
    has_binary: bool
    has_multiline: bool
    anchors: Dict[str, Any]
    special_keys: Dict[str, Any]
    validation_errors: List[str]
    statistics: Dict[str, Any]

class YAMLAnalyzer:
    """Analyzes YAML structure for intelligent chunking"""
    
    # Format detection patterns
    FORMAT_PATTERNS = {
        YAMLFormat.KUBERNETES: {
            'required_keys': ['apiVersion', 'kind'],
            'optional_keys': ['metadata', 'spec', 'status'],
            'kinds': ['Pod', 'Service', 'Deployment', 'ConfigMap', 'Secret', 
                     'Ingress', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']
        },
        YAMLFormat.DOCKER_COMPOSE: {
            'required_keys': ['services'],
            'optional_keys': ['version', 'networks', 'volumes', 'configs', 'secrets'],
            'version_pattern': re.compile(r'^[23]\.\d+$')
        },
        YAMLFormat.GITHUB_ACTIONS: {
            'required_keys': ['jobs'],
            'optional_keys': ['name', 'on', 'env', 'defaults'],
            'job_keys': ['runs-on', 'steps', 'needs', 'if']
        },
        YAMLFormat.GITLAB_CI: {
            'required_keys': [],  # Very flexible
            'optional_keys': ['stages', 'variables', 'default', 'include'],
            'job_indicators': ['script', 'stage', 'before_script', 'after_script']
        },
        YAMLFormat.ANSIBLE_PLAYBOOK: {
            'required_keys': [],
            'list_of_plays': True,
            'play_keys': ['hosts', 'tasks', 'vars', 'roles', 'handlers']
        },
        YAMLFormat.HELM_CHART: {
            'required_keys': ['apiVersion', 'name', 'version'],
            'optional_keys': ['description', 'type', 'dependencies', 'maintainers'],
            'file_name': 'Chart.yaml'
        },
        YAMLFormat.HELM_VALUES: {
            'file_name': 'values.yaml',
            'common_keys': ['replicaCount', 'image', 'service', 'ingress', 'resources']
        },
        YAMLFormat.CLOUDFORMATION: {
            'required_keys': ['Resources'],
            'optional_keys': ['AWSTemplateFormatVersion', 'Parameters', 'Outputs', 'Mappings']
        },
        YAMLFormat.SERVERLESS: {
            'required_keys': ['service', 'provider', 'functions'],
            'provider_keys': ['name', 'runtime', 'stage', 'region']
        },
        YAMLFormat.SWAGGER: {
            'required_keys': ['swagger', 'info', 'paths'],
            'version_pattern': re.compile(r'^2\.\d+$')
        },
        YAMLFormat.OPENAPI: {
            'required_keys': ['openapi', 'info', 'paths'],
            'version_pattern': re.compile(r'^3\.\d+\.\d+$')
        }
    }
    
    # Special keys that should be preserved together
    ATOMIC_GROUPS = {
        # Kubernetes
        'metadata': ['name', 'namespace', 'labels', 'annotations'],
        'spec': ['replicas', 'selector', 'template'],
        'container': ['name', 'image', 'ports', 'env', 'volumeMounts'],
        
        # Docker Compose
        'service': ['image', 'build', 'ports', 'volumes', 'environment', 'depends_on'],
        
        # CI/CD
        'job': ['stage', 'script', 'artifacts', 'cache', 'when'],
        'step': ['name', 'uses', 'run', 'with', 'env'],
        
        # Ansible
        'task': ['name', 'module', 'args', 'when', 'register', 'loop'],
        'play': ['hosts', 'vars', 'tasks', 'handlers', 'roles']
    }
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.documents = []
        self.anchors = {}
        self.line_map = {}
        
    def analyze_yaml(self, content: str, file_path: Optional[Path] = None) -> YAMLStructure:
        """
        Analyze YAML structure
        
        Args:
            content: YAML content as string
            file_path: Optional file path for format detection
            
        Returns:
            YAMLStructure analysis
        """
        try:
            # Build line map
            self._build_line_map(content)
            
            # Parse all documents
            documents = list(yaml.safe_load_all(content))
            
            # Analyze each document
            yaml_documents = []
            current_line = 0
            
            for i, doc in enumerate(documents):
                # Find document boundaries in content
                doc_start = self._find_document_start(content, i)
                doc_end = self._find_document_end(content, i, len(documents))
                
                # Build node tree for document
                root_node = self._build_node_tree(doc, None, "$", 0)
                
                yaml_doc = YAMLDocument(
                    index=i,
                    content=doc,
                    start_line=doc_start,
                    end_line=doc_end,
                    metadata=self._extract_document_metadata(doc),
                    root_node=root_node
                )
                
                yaml_documents.append(yaml_doc)
            
            # Detect format
            yaml_format = self._detect_format(yaml_documents, file_path)
            
            # Extract anchors and aliases
            self._extract_anchors(content)
            
            # Create structure
            structure = YAMLStructure(
                format=yaml_format,
                documents=yaml_documents,
                is_multi_document=len(yaml_documents) > 1,
                total_nodes=sum(self._count_nodes(d.root_node) for d in yaml_documents),
                total_scalars=sum(self._count_type(d.root_node, 'scalar') for d in yaml_documents),
                total_lists=sum(self._count_type(d.root_node, 'list') for d in yaml_documents),
                total_dicts=sum(self._count_type(d.root_node, 'dict') for d in yaml_documents),
                max_depth=max((self._calculate_max_depth(d.root_node) for d in yaml_documents), default=0),
                has_anchors=bool(self.anchors),
                has_aliases='*' in content,
                has_tags='!' in content,
                has_binary='!!binary' in content,
                has_multiline='|' in content or '>' in content,
                anchors=self.anchors,
                special_keys=self._extract_special_keys(yaml_documents),
                validation_errors=self._validate_format(yaml_documents, yaml_format),
                statistics=self._calculate_statistics(yaml_documents)
            )
            
            self.structure = structure
            return structure
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            return self._create_error_structure(str(e))
        except Exception as e:
            logger.error(f"Error analyzing YAML: {e}")
            return self._create_error_structure(str(e))
    
    def _build_line_map(self, content: str):
        """Build map of line numbers to positions"""
        self.line_map = {}
        lines = content.split('\n')
        position = 0
        
        for i, line in enumerate(lines):
            self.line_map[i] = {
                'start': position,
                'end': position + len(line),
                'content': line
            }
            position += len(line) + 1
    
    def _build_node_tree(self, data: Any, parent: Optional[YAMLNode], 
                        path: str, depth: int, key: Optional[str] = None) -> YAMLNode:
        """Build tree of YAML nodes"""
        # Determine node type
        if isinstance(data, dict):
            node_type = 'dict'
        elif isinstance(data, list):
            node_type = 'list'
        elif data is None:
            node_type = 'null'
        else:
            node_type = 'scalar'
        
        # Calculate size and tokens
        if node_type == 'scalar':
            data_str = str(data)
        else:
            data_str = yaml.dump(data, default_flow_style=False)
        
        size = len(data_str)
        token_count = self._count_tokens(data_str) if self.tokenizer else size // 4
        
        # Create node
        node = YAMLNode(
            key=key,
            value=data if node_type in ['scalar', 'null'] else None,
            node_type=node_type,
            path=path,
            depth=depth,
            line_start=0,  # Will be updated
            line_end=0,    # Will be updated
            parent=parent,
            children=[],
            size=size,
            token_count=token_count,
            metadata={}
        )
        
        # Process children
        if node_type == 'dict':
            node.metadata['key_count'] = len(data)
            for k, v in data.items():
                child_path = f"{path}.{k}" if path != "$" else f"$.{k}"
                child_node = self._build_node_tree(v, node, child_path, depth + 1, k)
                node.children.append(child_node)
                
        elif node_type == 'list':
            node.metadata['length'] = len(data)
            for i, item in enumerate(data):
                child_path = f"{path}[{i}]"
                child_node = self._build_node_tree(item, node, child_path, depth + 1, str(i))
                node.children.append(child_node)
        
        # Add type-specific metadata
        if node_type == 'scalar':
            node.metadata['is_multiline'] = '\n' in str(data)
            node.metadata['is_number'] = isinstance(data, (int, float))
            node.metadata['is_boolean'] = isinstance(data, bool)
            node.metadata['is_date'] = self._is_date_string(str(data))
        
        return node
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback estimation
        return len(text.split()) + text.count(':') + text.count('-')
    
    def _find_document_start(self, content: str, doc_index: int) -> int:
        """Find start line of document"""
        if doc_index == 0:
            return 0
        
        # Look for document separator
        separator_count = 0
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip() == '---':
                if separator_count == doc_index - 1:
                    return i + 1
                separator_count += 1
        
        return 0
    
    def _find_document_end(self, content: str, doc_index: int, total_docs: int) -> int:
        """Find end line of document"""
        lines = content.split('\n')
        
        if doc_index == total_docs - 1:
            return len(lines) - 1
        
        # Look for next document separator
        separator_count = 0
        
        for i, line in enumerate(lines):
            if line.strip() == '---':
                if separator_count == doc_index:
                    return i - 1
                separator_count += 1
        
        return len(lines) - 1
    
    def _extract_document_metadata(self, doc: Any) -> Dict[str, Any]:
        """Extract metadata from document"""
        metadata = {}
        
        if isinstance(doc, dict):
            # Check for Kubernetes
            if 'apiVersion' in doc and 'kind' in doc:
                metadata['type'] = 'kubernetes'
                metadata['api_version'] = doc['apiVersion']
                metadata['kind'] = doc['kind']
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    metadata['name'] = doc['metadata'].get('name')
                    metadata['namespace'] = doc['metadata'].get('namespace')
            
            # Check for Docker Compose service
            elif 'image' in doc or 'build' in doc:
                metadata['type'] = 'docker_service'
                metadata['image'] = doc.get('image')
            
            # Check for CI/CD job
            elif 'script' in doc or 'steps' in doc:
                metadata['type'] = 'ci_job'
        
        return metadata
    
    def _detect_format(self, documents: List[YAMLDocument], 
                      file_path: Optional[Path]) -> YAMLFormat:
        """Detect YAML format"""
        # Check filename hints
        if file_path:
            name = file_path.name.lower()
            
            if name == 'docker-compose.yml' or name == 'docker-compose.yaml':
                return YAMLFormat.DOCKER_COMPOSE
            elif name == 'chart.yaml':
                return YAMLFormat.HELM_CHART
            elif name == 'values.yaml':
                return YAMLFormat.HELM_VALUES
            elif name == '.gitlab-ci.yml':
                return YAMLFormat.GITLAB_CI
            elif name == '.travis.yml':
                return YAMLFormat.TRAVIS_CI
            elif name == '.circleci/config.yml':
                return YAMLFormat.CIRCLE_CI
            elif 'azure-pipelines' in name:
                return YAMLFormat.AZURE_PIPELINES
            elif name == '.pre-commit-config.yaml':
                return YAMLFormat.PRE_COMMIT
            elif name == 'mkdocs.yml':
                return YAMLFormat.MKDOCS
            elif name == 'pyproject.toml':
                return YAMLFormat.POETRY
            elif name == 'environment.yml' or name == 'environment.yaml':
                return YAMLFormat.CONDA_ENV
            elif name == 'serverless.yml':
                return YAMLFormat.SERVERLESS
            elif 'playbook' in name:
                return YAMLFormat.ANSIBLE_PLAYBOOK
            elif '.github/workflows' in str(file_path):
                return YAMLFormat.GITHUB_ACTIONS
        
        # Check content patterns
        if documents:
            first_doc = documents[0].content
            
            # Check each format pattern
            for format_type, pattern in self.FORMAT_PATTERNS.items():
                if self._matches_format_pattern(first_doc, pattern):
                    return format_type
        
        return YAMLFormat.GENERIC
    
    def _matches_format_pattern(self, doc: Any, pattern: Dict[str, Any]) -> bool:
        """Check if document matches format pattern"""
        if not isinstance(doc, dict) and pattern.get('list_of_plays'):
            # Check for Ansible playbook (list of plays)
            if isinstance(doc, list) and doc:
                first_item = doc[0]
                if isinstance(first_item, dict):
                    play_keys = pattern.get('play_keys', [])
                    return any(key in first_item for key in play_keys)
            return False
        
        if not isinstance(doc, dict):
            return False
        
        # Check required keys
        required = pattern.get('required_keys', [])
        if required and not all(key in doc for key in required):
            return False
        
        # Check optional keys (at least one should be present)
        optional = pattern.get('optional_keys', [])
        if optional and not any(key in doc for key in optional):
            return False
        
        # Check version patterns
        if 'version_pattern' in pattern:
            version = doc.get('version') or doc.get('swagger') or doc.get('openapi')
            if version:
                return bool(pattern['version_pattern'].match(str(version)))
        
        # Check job indicators (GitLab CI)
        if 'job_indicators' in pattern:
            indicators = pattern['job_indicators']
            for key, value in doc.items():
                if isinstance(value, dict) and any(ind in value for ind in indicators):
                    return True
        
        return True
    
    def _extract_anchors(self, content: str):
        """Extract YAML anchors and aliases"""
        self.anchors = {}
        
        # Find anchors
        anchor_pattern = re.compile(r'&(\w+)\s*\n')
        for match in anchor_pattern.finditer(content):
            anchor_name = match.group(1)
            # Find the value after the anchor
            start_pos = match.end()
            # Simplified: just store the position
            self.anchors[anchor_name] = start_pos
    
    def _extract_special_keys(self, documents: List[YAMLDocument]) -> Dict[str, Any]:
        """Extract special keys from documents"""
        special = {}
        
        for doc in documents:
            if isinstance(doc.content, dict):
                # Kubernetes special keys
                if 'apiVersion' in doc.content:
                    special['kubernetes'] = {
                        'api_version': doc.content.get('apiVersion'),
                        'kind': doc.content.get('kind')
                    }
                
                # Docker Compose special keys
                if 'services' in doc.content:
                    special['docker'] = {
                        'version': doc.content.get('version'),
                        'service_count': len(doc.content.get('services', {}))
                    }
                
                # CI/CD special keys
                if 'jobs' in doc.content:
                    special['ci'] = {
                        'job_count': len(doc.content.get('jobs', {}))
                    }
        
        return special
    
    def _validate_format(self, documents: List[YAMLDocument], 
                        format_type: YAMLFormat) -> List[str]:
        """Validate YAML against format requirements"""
        errors = []
        
        if format_type == YAMLFormat.KUBERNETES:
            for doc in documents:
                if isinstance(doc.content, dict):
                    if 'apiVersion' not in doc.content:
                        errors.append(f"Document {doc.index}: Missing apiVersion")
                    if 'kind' not in doc.content:
                        errors.append(f"Document {doc.index}: Missing kind")
                    
                    # Validate kind
                    kind = doc.content.get('kind')
                    if kind and kind not in self.FORMAT_PATTERNS[YAMLFormat.KUBERNETES].get('kinds', []):
                        errors.append(f"Document {doc.index}: Unknown kind '{kind}'")
        
        elif format_type == YAMLFormat.DOCKER_COMPOSE:
            if documents:
                doc = documents[0].content
                if isinstance(doc, dict):
                    if 'services' not in doc:
                        errors.append("Missing 'services' key")
                    
                    # Validate version
                    version = doc.get('version')
                    if version and not self.FORMAT_PATTERNS[YAMLFormat.DOCKER_COMPOSE]['version_pattern'].match(str(version)):
                        errors.append(f"Invalid version: {version}")
        
        return errors
    
    def _count_nodes(self, node: Optional[YAMLNode]) -> int:
        """Count total nodes"""
        if not node:
            return 0
        return 1 + sum(self._count_nodes(child) for child in node.children)
    
    def _count_type(self, node: Optional[YAMLNode], node_type: str) -> int:
        """Count nodes of specific type"""
        if not node:
            return 0
        count = 1 if node.node_type == node_type else 0
        return count + sum(self._count_type(child, node_type) for child in node.children)
    
    def _calculate_max_depth(self, node: Optional[YAMLNode]) -> int:
        """Calculate maximum depth"""
        if not node:
            return 0
        if not node.children:
            return node.depth
        return max(self._calculate_max_depth(child) for child in node.children)
    
    def _is_date_string(self, value: str) -> bool:
        """Check if string looks like a date"""
        date_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'),
            re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$')
        ]
        return any(p.match(value) for p in date_patterns)
    
    def _calculate_statistics(self, documents: List[YAMLDocument]) -> Dict[str, Any]:
        """Calculate statistics about YAML structure"""
        stats = {
            'document_count': len(documents),
            'unique_keys': set(),
            'key_frequency': defaultdict(int),
            'max_list_length': 0,
            'avg_dict_size': 0,
            'multiline_scalars': 0,
            'numeric_values': 0,
            'boolean_values': 0,
            'null_values': 0
        }
        
        dict_sizes = []
        
        def analyze(node: Optional[YAMLNode]):
            if not node:
                return
            
            if node.key:
                stats['unique_keys'].add(node.key)
                stats['key_frequency'][node.key] += 1
            
            if node.node_type == 'dict':
                dict_sizes.append(len(node.children))
            elif node.node_type == 'list':
                stats['max_list_length'] = max(stats['max_list_length'], len(node.children))
            elif node.node_type == 'scalar':
                if node.metadata.get('is_multiline'):
                    stats['multiline_scalars'] += 1
                if node.metadata.get('is_number'):
                    stats['numeric_values'] += 1
                if node.metadata.get('is_boolean'):
                    stats['boolean_values'] += 1
            elif node.node_type == 'null':
                stats['null_values'] += 1
            
            for child in node.children:
                analyze(child)
        
        for doc in documents:
            analyze(doc.root_node)
        
        # Convert sets to lists
        stats['unique_keys'] = list(stats['unique_keys'])
        stats['key_frequency'] = dict(stats['key_frequency'])
        
        # Calculate average dict size
        if dict_sizes:
            stats['avg_dict_size'] = sum(dict_sizes) / len(dict_sizes)
        
        return stats
    
    def _create_error_structure(self, error_msg: str) -> YAMLStructure:
        """Create error structure for invalid YAML"""
        return YAMLStructure(
            format=YAMLFormat.GENERIC,
            documents=[],
            is_multi_document=False,
            total_nodes=0,
            total_scalars=0,
            total_lists=0,
            total_dicts=0,
            max_depth=0,
            has_anchors=False,
            has_aliases=False,
            has_tags=False,
            has_binary=False,
            has_multiline=False,
            anchors={},
            special_keys={},
            validation_errors=[error_msg],
            statistics={}
        )

class YAMLChunkingStrategy:
    """Base class for YAML chunking strategies"""
    
    def chunk(self, documents: List[YAMLDocument], max_tokens: int, 
             format_type: YAMLFormat) -> List[Dict[str, Any]]:
        """Create chunks from YAML documents"""
        raise NotImplementedError

class FormatSpecificStrategy(YAMLChunkingStrategy):
    """Format-specific chunking strategies"""
    
    def chunk(self, documents: List[YAMLDocument], max_tokens: int,
             format_type: YAMLFormat) -> List[Dict[str, Any]]:
        """Chunk based on format"""
        if format_type == YAMLFormat.KUBERNETES:
            return self._chunk_kubernetes(documents, max_tokens)
        elif format_type == YAMLFormat.DOCKER_COMPOSE:
            return self._chunk_docker_compose(documents, max_tokens)
        elif format_type == YAMLFormat.GITHUB_ACTIONS:
            return self._chunk_github_actions(documents, max_tokens)
        elif format_type == YAMLFormat.ANSIBLE_PLAYBOOK:
            return self._chunk_ansible_playbook(documents, max_tokens)
        elif format_type == YAMLFormat.HELM_VALUES:
            return self._chunk_helm_values(documents, max_tokens)
        else:
            return self._chunk_generic(documents, max_tokens)
    
    def _chunk_kubernetes(self, documents: List[YAMLDocument], 
                         max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk Kubernetes manifests"""
        chunks = []
        
        for doc in documents:
            if isinstance(doc.content, dict):
                # Each Kubernetes resource as a chunk
                resource = doc.content
                
                # Metadata chunk
                if 'metadata' in resource:
                    metadata_chunk = {
                        'type': 'k8s_metadata',
                        'kind': resource.get('kind'),
                        'api_version': resource.get('apiVersion'),
                        'content': {
                            'apiVersion': resource.get('apiVersion'),
                            'kind': resource.get('kind'),
                            'metadata': resource.get('metadata')
                        }
                    }
                    chunks.append(metadata_chunk)
                
                # Spec chunk(s)
                if 'spec' in resource:
                    spec = resource['spec']
                    spec_yaml = yaml.dump(spec, default_flow_style=False)
                    
                    if self._count_tokens(spec_yaml) <= max_tokens:
                        chunks.append({
                            'type': 'k8s_spec',
                            'kind': resource.get('kind'),
                            'content': {'spec': spec}
                        })
                    else:
                        # Split large spec
                        spec_chunks = self._split_k8s_spec(spec, resource.get('kind'), max_tokens)
                        chunks.extend(spec_chunks)
                
                # Status chunk (if present)
                if 'status' in resource:
                    chunks.append({
                        'type': 'k8s_status',
                        'kind': resource.get('kind'),
                        'content': {'status': resource['status']}
                    })
        
        return chunks
    
    def _chunk_docker_compose(self, documents: List[YAMLDocument],
                            max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk Docker Compose file"""
        chunks = []
        
        if not documents:
            return chunks
        
        compose = documents[0].content
        if not isinstance(compose, dict):
            return chunks
        
        # Version and top-level config
        top_level = {
            'version': compose.get('version'),
            'x-anchors': {k: v for k, v in compose.items() if k.startswith('x-')}
        }
        
        if top_level['version'] or top_level['x-anchors']:
            chunks.append({
                'type': 'compose_metadata',
                'content': top_level
            })
        
        # Services
        if 'services' in compose:
            services = compose['services']
            
            for service_name, service_config in services.items():
                service_yaml = yaml.dump({service_name: service_config}, default_flow_style=False)
                
                if self._count_tokens(service_yaml) <= max_tokens:
                    chunks.append({
                        'type': 'compose_service',
                        'service': service_name,
                        'content': {service_name: service_config}
                    })
                else:
                    # Split large service
                    service_chunks = self._split_docker_service(service_name, service_config, max_tokens)
                    chunks.extend(service_chunks)
        
        # Networks
        if 'networks' in compose:
            chunks.append({
                'type': 'compose_networks',
                'content': {'networks': compose['networks']}
            })
        
        # Volumes
        if 'volumes' in compose:
            chunks.append({
                'type': 'compose_volumes',
                'content': {'volumes': compose['volumes']}
            })
        
        # Configs and Secrets
        for key in ['configs', 'secrets']:
            if key in compose:
                chunks.append({
                    'type': f'compose_{key}',
                    'content': {key: compose[key]}
                })
        
        return chunks
    
    def _chunk_github_actions(self, documents: List[YAMLDocument],
                            max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk GitHub Actions workflow"""
        chunks = []
        
        if not documents:
            return chunks
        
        workflow = documents[0].content
        if not isinstance(workflow, dict):
            return chunks
        
        # Workflow metadata
        metadata = {
            'name': workflow.get('name'),
            'on': workflow.get('on'),
            'env': workflow.get('env'),
            'defaults': workflow.get('defaults')
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        if metadata:
            chunks.append({
                'type': 'workflow_metadata',
                'content': metadata
            })
        
        # Jobs
        if 'jobs' in workflow:
            jobs = workflow['jobs']
            
            for job_name, job_config in jobs.items():
                job_yaml = yaml.dump({job_name: job_config}, default_flow_style=False)
                
                if self._count_tokens(job_yaml) <= max_tokens:
                    chunks.append({
                        'type': 'workflow_job',
                        'job': job_name,
                        'content': {job_name: job_config}
                    })
                else:
                    # Split large job by steps
                    job_chunks = self._split_github_job(job_name, job_config, max_tokens)
                    chunks.extend(job_chunks)
        
        return chunks
    
    def _chunk_ansible_playbook(self, documents: List[YAMLDocument],
                              max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk Ansible playbook"""
        chunks = []
        
        if not documents:
            return chunks
        
        playbook = documents[0].content
        
        # Handle list of plays
        if isinstance(playbook, list):
            for i, play in enumerate(playbook):
                if isinstance(play, dict):
                    play_yaml = yaml.dump(play, default_flow_style=False)
                    
                    if self._count_tokens(play_yaml) <= max_tokens:
                        chunks.append({
                            'type': 'ansible_play',
                            'play_index': i,
                            'hosts': play.get('hosts'),
                            'content': play
                        })
                    else:
                        # Split play by tasks
                        play_chunks = self._split_ansible_play(play, i, max_tokens)
                        chunks.extend(play_chunks)
        
        return chunks
    
    def _chunk_helm_values(self, documents: List[YAMLDocument],
                         max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk Helm values file"""
        chunks = []
        
        if not documents:
            return chunks
        
        values = documents[0].content
        if not isinstance(values, dict):
            return chunks
        
        # Common Helm value sections
        sections = ['global', 'image', 'service', 'ingress', 'resources',
                   'nodeSelector', 'tolerations', 'affinity']
        
        # Process each section
        for section in sections:
            if section in values:
                section_yaml = yaml.dump({section: values[section]}, default_flow_style=False)
                
                if self._count_tokens(section_yaml) <= max_tokens:
                    chunks.append({
                        'type': 'helm_values_section',
                        'section': section,
                        'content': {section: values[section]}
                    })
                else:
                    # Split large section
                    section_chunks = self._split_values_section(section, values[section], max_tokens)
                    chunks.extend(section_chunks)
        
        # Other values
        other_values = {k: v for k, v in values.items() if k not in sections}
        
        if other_values:
            other_yaml = yaml.dump(other_values, default_flow_style=False)
            
            if self._count_tokens(other_yaml) <= max_tokens:
                chunks.append({
                    'type': 'helm_values_other',
                    'content': other_values
                })
            else:
                # Split by keys
                for key, value in other_values.items():
                    chunks.append({
                        'type': 'helm_values_other',
                        'key': key,
                        'content': {key: value}
                    })
        
        return chunks
    
    def _chunk_generic(self, documents: List[YAMLDocument],
                      max_tokens: int) -> List[Dict[str, Any]]:
        """Generic YAML chunking"""
        chunks = []
        
        for doc_idx, doc in enumerate(documents):
            if doc.root_node:
                doc_chunks = self._chunk_node_tree(doc.root_node, max_tokens)
                
                for chunk in doc_chunks:
                    chunk['document_index'] = doc_idx
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_node_tree(self, node: YAMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk a node tree"""
        chunks = []
        
        if node.token_count <= max_tokens:
            # Entire node fits in one chunk
            chunks.append({
                'type': 'yaml_node',
                'path': node.get_full_path(),
                'content': self._node_to_yaml(node)
            })
        else:
            # Split node
            if node.node_type == 'dict':
                chunks.extend(self._split_dict_node(node, max_tokens))
            elif node.node_type == 'list':
                chunks.extend(self._split_list_node(node, max_tokens))
            else:
                # Scalar too large (shouldn't happen often)
                chunks.append({
                    'type': 'yaml_scalar',
                    'path': node.get_full_path(),
                    'content': node.value
                })
        
        return chunks
    
    def _split_k8s_spec(self, spec: Dict[str, Any], kind: str,
                       max_tokens: int) -> List[Dict[str, Any]]:
        """Split Kubernetes spec"""
        chunks = []
        
        # Common spec sections
        if kind == 'Deployment' or kind == 'StatefulSet':
            # Split by replicas, selector, template
            for key in ['replicas', 'selector']:
                if key in spec:
                    chunks.append({
                        'type': f'k8s_spec_{key}',
                        'kind': kind,
                        'content': {key: spec[key]}
                    })
            
            # Template is usually the largest part
            if 'template' in spec:
                template_chunks = self._split_pod_template(spec['template'], max_tokens)
                chunks.extend(template_chunks)
        else:
            # Generic spec splitting
            for key, value in spec.items():
                chunks.append({
                    'type': 'k8s_spec_part',
                    'kind': kind,
                    'key': key,
                    'content': {key: value}
                })
        
        return chunks
    
    def _split_pod_template(self, template: Dict[str, Any],
                          max_tokens: int) -> List[Dict[str, Any]]:
        """Split pod template"""
        chunks = []
        
        # Metadata
        if 'metadata' in template:
            chunks.append({
                'type': 'k8s_pod_metadata',
                'content': {'metadata': template['metadata']}
            })
        
        # Spec
        if 'spec' in template:
            pod_spec = template['spec']
            
            # Containers
            if 'containers' in pod_spec:
                for container in pod_spec['containers']:
                    chunks.append({
                        'type': 'k8s_container',
                        'name': container.get('name'),
                        'content': container
                    })
            
            # Other spec fields
            other_spec = {k: v for k, v in pod_spec.items() if k != 'containers'}
            if other_spec:
                chunks.append({
                    'type': 'k8s_pod_spec_other',
                    'content': other_spec
                })
        
        return chunks
    
    def _split_docker_service(self, service_name: str, service_config: Dict[str, Any],
                            max_tokens: int) -> List[Dict[str, Any]]:
        """Split Docker service configuration"""
        chunks = []
        
        # Core configuration
        core_keys = ['image', 'build', 'command', 'entrypoint']
        core_config = {k: v for k, v in service_config.items() if k in core_keys}
        
        if core_config:
            chunks.append({
                'type': 'compose_service_core',
                'service': service_name,
                'content': core_config
            })
        
        # Environment
        if 'environment' in service_config:
            chunks.append({
                'type': 'compose_service_env',
                'service': service_name,
                'content': {'environment': service_config['environment']}
            })
        
        # Volumes
        if 'volumes' in service_config:
            chunks.append({
                'type': 'compose_service_volumes',
                'service': service_name,
                'content': {'volumes': service_config['volumes']}
            })
        
        # Networks and ports
        net_port = {k: v for k, v in service_config.items() 
                   if k in ['networks', 'ports', 'expose']}
        
        if net_port:
            chunks.append({
                'type': 'compose_service_network',
                'service': service_name,
                'content': net_port
            })
        
        # Other configuration
        other_keys = set(service_config.keys()) - set(core_keys) - {'environment', 'volumes', 'networks', 'ports', 'expose'}
        other_config = {k: service_config[k] for k in other_keys}
        
        if other_config:
            chunks.append({
                'type': 'compose_service_other',
                'service': service_name,
                'content': other_config
            })
        
        return chunks
    
    def _split_github_job(self, job_name: str, job_config: Dict[str, Any],
                        max_tokens: int) -> List[Dict[str, Any]]:
        """Split GitHub Actions job"""
        chunks = []
        
        # Job metadata
        metadata = {k: v for k, v in job_config.items() 
                   if k in ['runs-on', 'needs', 'if', 'strategy', 'container']}
        
        if metadata:
            chunks.append({
                'type': 'workflow_job_metadata',
                'job': job_name,
                'content': metadata
            })
        
        # Steps
        if 'steps' in job_config:
            steps = job_config['steps']
            
            # Batch steps if they're small
            step_batches = []
            current_batch = []
            current_tokens = 0
            
            for step in steps:
                step_yaml = yaml.dump(step, default_flow_style=False)
                step_tokens = self._count_tokens(step_yaml)
                
                if current_tokens + step_tokens > max_tokens and current_batch:
                    step_batches.append(current_batch)
                    current_batch = [step]
                    current_tokens = step_tokens
                else:
                    current_batch.append(step)
                    current_tokens += step_tokens
            
            if current_batch:
                step_batches.append(current_batch)
            
            # Create chunks for step batches
            for i, batch in enumerate(step_batches):
                chunks.append({
                    'type': 'workflow_job_steps',
                    'job': job_name,
                    'batch': i,
                    'content': {'steps': batch}
                })
        
        return chunks
    
    def _split_ansible_play(self, play: Dict[str, Any], play_index: int,
                          max_tokens: int) -> List[Dict[str, Any]]:
        """Split Ansible play"""
        chunks = []
        
        # Play header
        header = {k: v for k, v in play.items() 
                 if k in ['name', 'hosts', 'vars', 'become', 'gather_facts']}
        
        if header:
            chunks.append({
                'type': 'ansible_play_header',
                'play_index': play_index,
                'content': header
            })
        
        # Tasks
        if 'tasks' in play:
            task_chunks = self._chunk_ansible_tasks(play['tasks'], play_index, max_tokens)
            chunks.extend(task_chunks)
        
        # Handlers
        if 'handlers' in play:
            handler_chunks = self._chunk_ansible_tasks(play['handlers'], play_index, max_tokens, is_handler=True)
            chunks.extend(handler_chunks)
        
        # Roles
        if 'roles' in play:
            chunks.append({
                'type': 'ansible_play_roles',
                'play_index': play_index,
                'content': {'roles': play['roles']}
            })
        
        return chunks
    
    def _chunk_ansible_tasks(self, tasks: List[Dict[str, Any]], play_index: int,
                           max_tokens: int, is_handler: bool = False) -> List[Dict[str, Any]]:
        """Chunk Ansible tasks"""
        chunks = []
        
        # Batch tasks
        task_batches = []
        current_batch = []
        current_tokens = 0
        
        for task in tasks:
            task_yaml = yaml.dump(task, default_flow_style=False)
            task_tokens = self._count_tokens(task_yaml)
            
            if current_tokens + task_tokens > max_tokens and current_batch:
                task_batches.append(current_batch)
                current_batch = [task]
                current_tokens = task_tokens
            else:
                current_batch.append(task)
                current_tokens += task_tokens
        
        if current_batch:
            task_batches.append(current_batch)
        
        # Create chunks
        task_type = 'ansible_handlers' if is_handler else 'ansible_tasks'
        
        for i, batch in enumerate(task_batches):
            chunks.append({
                'type': task_type,
                'play_index': play_index,
                'batch': i,
                'content': batch
            })
        
        return chunks
    
    def _split_values_section(self, section: str, values: Any,
                            max_tokens: int) -> List[Dict[str, Any]]:
        """Split Helm values section"""
        chunks = []
        
        if isinstance(values, dict):
            # Split by keys
            for key, value in values.items():
                chunks.append({
                    'type': 'helm_values_subsection',
                    'section': section,
                    'key': key,
                    'content': {key: value}
                })
        else:
            # Single value
            chunks.append({
                'type': 'helm_values_section',
                'section': section,
                'content': {section: values}
            })
        
        return chunks
    
    def _split_dict_node(self, node: YAMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Split dictionary node"""
        chunks = []
        
        # Group children by size
        child_groups = []
        current_group = []
        current_tokens = 0
        
        for child in node.children:
            if current_tokens + child.token_count > max_tokens and current_group:
                child_groups.append(current_group)
                current_group = [child]
                current_tokens = child.token_count
            else:
                current_group.append(child)
                current_tokens += child.token_count
        
        if current_group:
            child_groups.append(current_group)
        
        # Create chunks
        for i, group in enumerate(child_groups):
            group_dict = {}
            for child in group:
                if child.node_type == 'dict':
                    group_dict[child.key] = self._node_to_dict(child)
                elif child.node_type == 'list':
                    group_dict[child.key] = self._node_to_list(child)
                else:
                    group_dict[child.key] = child.value
            
            chunks.append({
                'type': 'yaml_dict_part',
                'path': node.get_full_path(),
                'part': i,
                'content': group_dict
            })
        
        return chunks
    
    def _split_list_node(self, node: YAMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Split list node"""
        chunks = []
        
        # Batch list items
        item_batches = []
        current_batch = []
        current_tokens = 0
        
        for child in node.children:
            if current_tokens + child.token_count > max_tokens and current_batch:
                item_batches.append(current_batch)
                current_batch = [child]
                current_tokens = child.token_count
            else:
                current_batch.append(child)
                current_tokens += child.token_count
        
        if current_batch:
            item_batches.append(current_batch)
        
        # Create chunks
        for i, batch in enumerate(item_batches):
            batch_list = []
            for child in batch:
                if child.node_type == 'dict':
                    batch_list.append(self._node_to_dict(child))
                elif child.node_type == 'list':
                    batch_list.append(self._node_to_list(child))
                else:
                    batch_list.append(child.value)
            
            chunks.append({
                'type': 'yaml_list_part',
                'path': node.get_full_path(),
                'part': i,
                'content': batch_list
            })
        
        return chunks
    
    def _node_to_yaml(self, node: YAMLNode) -> Any:
        """Convert node to YAML-serializable structure"""
        if node.node_type == 'dict':
            return self._node_to_dict(node)
        elif node.node_type == 'list':
            return self._node_to_list(node)
        else:
            return node.value
    
    def _node_to_dict(self, node: YAMLNode) -> Dict[str, Any]:
        """Convert dict node to dictionary"""
        result = {}
        for child in node.children:
            if child.node_type == 'dict':
                result[child.key] = self._node_to_dict(child)
            elif child.node_type == 'list':
                result[child.key] = self._node_to_list(child)
            else:
                result[child.key] = child.value
        return result
    
    def _node_to_list(self, node: YAMLNode) -> List[Any]:
        """Convert list node to list"""
        result = []
        for child in node.children:
            if child.node_type == 'dict':
                result.append(self._node_to_dict(child))
            elif child.node_type == 'list':
                result.append(self._node_to_list(child))
            else:
                result.append(child.value)
        return result
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        # Simplified token counting
        return len(text.split()) + text.count(':') + text.count('-')

class YAMLChunker(BaseChunker):
    """Chunker specialized for YAML files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = YAMLAnalyzer(tokenizer)
        self.strategy = FormatSpecificStrategy()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from YAML file
        
        Args:
            content: YAML content as string
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze YAML structure
            structure = self.analyzer.analyze_yaml(content, file_context.path)
            
            # Check for errors
            if structure.validation_errors:
                logger.warning(f"YAML validation errors: {structure.validation_errors}")
            
            # If small enough, single chunk
            if self.count_tokens(content) <= self.max_tokens:
                return [self.create_chunk(
                    content=content,
                    chunk_type='yaml_complete',
                    metadata={
                        'format': structure.format.value,
                        'is_multi_document': structure.is_multi_document,
                        'document_count': len(structure.documents)
                    },
                    file_path=str(file_context.path)
                )]
            
            # Apply format-specific chunking
            chunk_data = self.strategy.chunk(
                structure.documents,
                self.max_tokens,
                structure.format
            )
            
            # Convert to Chunk objects
            chunks = []
            for i, data in enumerate(chunk_data):
                # Serialize content to YAML
                if isinstance(data.get('content'), (dict, list)):
                    chunk_content = yaml.dump(
                        data['content'],
                        default_flow_style=False,
                        sort_keys=False
                    )
                else:
                    chunk_content = str(data.get('content', ''))
                
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type=data.get('type', 'yaml_chunk'),
                    metadata={
                        'format': structure.format.value,
                        'chunk_index': i,
                        'total_chunks': len(chunk_data),
                        **{k: v for k, v in data.items() if k not in ['type', 'content']}
                    },
                    file_path=str(file_context.path)
                ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking YAML file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for invalid YAML"""
        logger.warning(f"Using fallback chunking for YAML file {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        current_indent = 0
        
        for line in lines:
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            line_tokens = self.count_tokens(line)
            
            # Check if we should start new chunk
            should_split = (
                current_tokens + line_tokens > self.max_tokens and
                current_chunk and
                indent == 0  # Try to split at top-level
            )
            
            if should_split:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='yaml_fallback',
                    metadata={
                        'is_fallback': True,
                        'line_count': len(current_chunk)
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
            current_indent = indent
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='yaml_fallback',
                metadata={
                    'is_fallback': True,
                    'line_count': len(current_chunk)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks