"""
Central registry for managing all chunkers
Provides automatic chunker selection based on file type and dynamic chunker loading
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple, Set
from dataclasses import dataclass, field
from src.utils.logger import get_logger
import yaml
import json
from enum import Enum
from collections import defaultdict

from src.core.base_chunker import BaseChunker, Chunk
from src.core.file_context import FileContext
from config.settings import settings

logger = get_logger(__name__)

class ChunkerPriority(Enum):
    """Priority levels for chunker selection"""
    EXACT_MATCH = 1      # Exact filename match (e.g., pom.xml)
    PATTERN_MATCH = 2    # Pattern match (e.g., *Test.java)
    EXTENSION = 3        # File extension match
    CONTENT_BASED = 4    # Content analysis match
    FALLBACK = 5         # Default/fallback chunker

@dataclass
class ChunkerInfo:
    """Information about a registered chunker"""
    name: str
    chunker_class: Type[BaseChunker]
    file_extensions: Set[str]
    file_patterns: List[str]
    exact_filenames: Set[str]
    mime_types: Set[str]
    priority: ChunkerPriority
    description: str
    version: str
    author: str
    capabilities: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class ChunkerRegistry:
    """
    Central registry for all chunkers
    Manages chunker registration, selection, and instantiation
    """
    
    def __init__(self, tokenizer=None):
        """
        Initialize chunker registry
        
        Args:
            tokenizer: Optional tokenizer for chunkers to use
        """
        self.tokenizer = tokenizer
        self._chunkers: Dict[str, ChunkerInfo] = {}
        self._extension_map: Dict[str, List[str]] = defaultdict(list)
        self._filename_map: Dict[str, str] = {}
        self._pattern_map: List[Tuple[str, str]] = []
        self._mime_map: Dict[str, List[str]] = defaultdict(list)
        
        # Chunker instances cache
        self._instances: Dict[str, BaseChunker] = {}
        
        # Load configuration
        self._load_config()
        
        # Auto-register built-in chunkers
        self._register_builtin_chunkers()
        
        # Load custom chunkers if configured
        if getattr(settings, 'enable_custom_chunkers', False):
            self._load_custom_chunkers()
    
    def _load_config(self):
        """Load registry configuration"""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'chunker_registry.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default registry configuration"""
        return {
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'default_max_tokens': 450,
            'enable_content_detection': True,
            'enable_adaptive_chunking': True,
            'chunker_timeout': 30,  # seconds
            'cache_instances': True,
            'custom_chunkers_path': 'custom_chunkers',
            'builtin_chunkers': {
                'python': {
                    'module': 'src.chunkers.code.python_chunker',
                    'class': 'PythonChunker',
                    'extensions': ['.py', '.pyw', '.pyx', '.pxd', '.pxi'],
                    'patterns': ['*.py', '*.pyw'],
                    'filenames': ['__init__.py', 'setup.py', 'manage.py'],
                    'priority': 'extension',
                    'description': 'Python code chunker with AST parsing'
                },
                'java': {
                    'module': 'src.chunkers.code.java_chunker',
                    'class': 'JavaChunker',
                    'extensions': ['.java', '.jsp', '.jspx'],
                    'patterns': ['*.java'],
                    'filenames': [],
                    'priority': 'extension',
                    'description': 'Java code chunker with AST parsing'
                },
                'javascript': {
                    'module': 'src.chunkers.code.javascript_chunker',
                    'class': 'JavaScriptChunker',
                    'extensions': ['.js', '.jsx', '.mjs', '.cjs'],
                    'patterns': ['*.js', '*.jsx'],
                    'filenames': ['index.js', 'app.js', 'server.js'],
                    'priority': 'extension',
                    'description': 'JavaScript/JSX chunker with framework detection'
                },
                'typescript': {
                    'module': 'src.chunkers.code.javascript_chunker',
                    'class': 'JavaScriptChunker',
                    'extensions': ['.ts', '.tsx', '.d.ts'],
                    'patterns': ['*.ts', '*.tsx'],
                    'filenames': ['index.ts', 'app.ts'],
                    'priority': 'extension',
                    'description': 'TypeScript chunker with type support'
                },
                'sql': {
                    'module': 'src.chunkers.database.sql_chunker',
                    'class': 'SQLChunker',
                    'extensions': ['.sql', '.ddl', '.dml'],
                    'patterns': ['*.sql', 'V*__*.sql'],  # Flyway migrations
                    'filenames': [],
                    'priority': 'extension',
                    'description': 'SQL chunker with multi-dialect support'
                },
                'plsql': {
                    'module': 'src.chunkers.database.plsql_chunker',
                    'class': 'PLSQLChunker',
                    'extensions': ['.pls', '.plb', '.pks', '.pkb', '.trg', '.fnc', '.prc'],
                    'patterns': ['*.pls', '*.plsql'],
                    'filenames': [],
                    'priority': 'extension',
                    'description': 'PL/SQL chunker for Oracle database objects'
                },
                'xml': {
                    'module': 'src.chunkers.config.xml_chunker',
                    'class': 'XMLChunker',
                    'extensions': ['.xml', '.xsd', '.xslt', '.wsdl', '.svg'],
                    'patterns': ['*.xml'],
                    'filenames': ['web.xml', 'AndroidManifest.xml'],
                    'priority': 'extension',
                    'description': 'XML chunker with namespace support'
                },
                'maven': {
                    'module': 'src.chunkers.build.maven_chunker',
                    'class': 'MavenChunker',
                    'extensions': [],
                    'patterns': [],
                    'filenames': ['pom.xml'],
                    'priority': 'exact_match',
                    'description': 'Maven POM file chunker'
                },
                'yaml': {
                    'module': 'src.chunkers.config.yaml_chunker',
                    'class': 'YAMLChunker',
                    'extensions': ['.yaml', '.yml'],
                    'patterns': ['*.yaml', '*.yml'],
                    'filenames': ['docker-compose.yml', '.gitlab-ci.yml', 'values.yaml'],
                    'priority': 'extension',
                    'description': 'YAML chunker with format detection'
                },
                'json': {
                    'module': 'src.chunkers.config.json_chunker',
                    'class': 'JSONChunker',
                    'extensions': ['.json', '.jsonl', '.geojson'],
                    'patterns': ['*.json'],
                    'filenames': ['package.json', 'tsconfig.json', 'composer.json'],
                    'priority': 'extension',
                    'description': 'JSON chunker with schema support'
                },
                'properties': {
                    'module': 'src.chunkers.config.properties_chunker',
                    'class': 'PropertiesChunker',
                    'extensions': ['.properties', '.ini', '.cfg', '.conf', '.env'],
                    'patterns': ['*.properties', '*.ini'],
                    'filenames': ['.env', '.env.local', 'application.properties'],
                    'priority': 'extension',
                    'description': 'Properties/INI/Config file chunker'
                },
                'terraform': {
                    'module': 'src.chunkers.iac.terraform_chunker',
                    'class': 'TerraformChunker',
                    'extensions': ['.tf', '.tfvars', '.hcl'],
                    'patterns': ['*.tf'],
                    'filenames': ['terraform.tfvars', 'variables.tf', 'main.tf'],
                    'priority': 'extension',
                    'description': 'Terraform HCL chunker'
                },
                'docker': {
                    'module': 'src.chunkers.build.docker_chunker',
                    'class': 'DockerChunker',
                    'extensions': [],
                    'patterns': ['Dockerfile*', 'docker-compose*.yml'],
                    'filenames': ['Dockerfile', 'docker-compose.yml'],
                    'priority': 'pattern_match',
                    'description': 'Docker and Docker Compose chunker'
                },

                'generic_code': {
                    'module': 'src.chunkers.code.generic_code_chunker',
                    'class': 'GenericCodeChunker',
                    'extensions': ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.cs', '.go',
                                  '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
                                  '.m', '.mm', '.lua', '.pl', '.sh', '.bash', '.zsh'],
                    'patterns': [],
                    'filenames': [],
                    'priority': 'extension',
                    'description': 'Generic code chunker for multiple languages'
                },

                'adaptive': {
                    'module': 'src.chunkers.adaptive.adaptive_chunker',
                    'class': 'AdaptiveChunker',
                    'extensions': [],
                    'patterns': [],
                    'filenames': [],
                    'priority': 'content_based',
                    'description': 'Adaptive chunker using content analysis'
                }
            }
        }
    
    def _register_builtin_chunkers(self):
        """Register all built-in chunkers"""
        for chunker_name, config in self.config.get('builtin_chunkers', {}).items():
            try:
                self._register_chunker_from_config(chunker_name, config)
                logger.info(f"Registered built-in chunker: {chunker_name}")
            except Exception as e:
                logger.error(f"Failed to register chunker {chunker_name}: {e}")
    
    def _register_chunker_from_config(self, name: str, config: Dict[str, Any]):
        """Register a chunker from configuration"""
        # Import chunker class
        module = importlib.import_module(config['module'])
        chunker_class = getattr(module, config['class'])
        
        # Validate it's a BaseChunker subclass
        if not issubclass(chunker_class, BaseChunker):
            raise ValueError(f"{config['class']} is not a BaseChunker subclass")
        
        # Determine priority
        priority_map = {
            'exact_match': ChunkerPriority.EXACT_MATCH,
            'pattern_match': ChunkerPriority.PATTERN_MATCH,
            'extension': ChunkerPriority.EXTENSION,
            'content_based': ChunkerPriority.CONTENT_BASED,
            'fallback': ChunkerPriority.FALLBACK
        }
        priority = priority_map.get(config.get('priority', 'extension'), ChunkerPriority.EXTENSION)
        
        # Create ChunkerInfo
        chunker_info = ChunkerInfo(
            name=name,
            chunker_class=chunker_class,
            file_extensions=set(config.get('extensions', [])),
            file_patterns=config.get('patterns', []),
            exact_filenames=set(config.get('filenames', [])),
            mime_types=set(config.get('mime_types', [])),
            priority=priority,
            description=config.get('description', ''),
            version=config.get('version', '1.0.0'),
            author=config.get('author', 'unknown'),
            capabilities=config.get('capabilities', []),
            metadata=config.get('metadata', {})
        )
        
        # Register chunker
        self.register(name, chunker_info)
    
    def register(self, name: str, chunker_info: ChunkerInfo):
        """
        Register a chunker
        
        Args:
            name: Unique name for the chunker
            chunker_info: ChunkerInfo object with chunker details
        """
        if name in self._chunkers:
            logger.warning(f"Overwriting existing chunker: {name}")
        
        self._chunkers[name] = chunker_info
        
        # Update extension map
        for ext in chunker_info.file_extensions:
            self._extension_map[ext.lower()].append(name)
        
        # Update filename map
        for filename in chunker_info.exact_filenames:
            self._filename_map[filename.lower()] = name
        
        # Update pattern map
        for pattern in chunker_info.file_patterns:
            self._pattern_map.append((pattern, name))
        
        # Update MIME type map
        for mime_type in chunker_info.mime_types:
            self._mime_map[mime_type].append(name)
        
        logger.debug(f"Registered chunker: {name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a chunker
        
        Args:
            name: Name of chunker to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if name not in self._chunkers:
            return False
        
        chunker_info = self._chunkers[name]
        
        # Remove from extension map
        for ext in chunker_info.file_extensions:
            if name in self._extension_map[ext.lower()]:
                self._extension_map[ext.lower()].remove(name)
        
        # Remove from filename map
        for filename in chunker_info.exact_filenames:
            if self._filename_map.get(filename.lower()) == name:
                del self._filename_map[filename.lower()]
        
        # Remove from pattern map
        self._pattern_map = [(p, n) for p, n in self._pattern_map if n != name]
        
        # Remove from MIME map
        for mime_type in chunker_info.mime_types:
            if name in self._mime_map[mime_type]:
                self._mime_map[mime_type].remove(name)
        
        # Remove from chunkers and instances
        del self._chunkers[name]
        if name in self._instances:
            del self._instances[name]
        
        logger.debug(f"Unregistered chunker: {name}")
        return True
    
    def get_chunker(self, file_context: FileContext, 
                   force_chunker: Optional[str] = None) -> BaseChunker:
        """
        Get appropriate chunker for a file
        
        Args:
            file_context: Context of the file to chunk
            force_chunker: Optional name of chunker to force use
            
        Returns:
            Appropriate chunker instance
        """
        # Force specific chunker if requested
        if force_chunker:
            if force_chunker in self._chunkers:
                return self._get_or_create_instance(force_chunker)
            else:
                logger.warning(f"Forced chunker {force_chunker} not found, using auto-select")
        
        # Auto-select chunker
        chunker_name = self._select_chunker(file_context)
        
        if not chunker_name:
            # Use fallback chunker
            chunker_name = 'adaptive' if 'adaptive' in self._chunkers else 'text'
        
        return self._get_or_create_instance(chunker_name)
    
    def _select_chunker(self, file_context: FileContext) -> Optional[str]:
        """
        Select the best chunker for a file
        
        Args:
            file_context: File context
            
        Returns:
            Name of selected chunker or None
        """
        candidates = []
        
        # Check exact filename match
        filename = file_context.path.name.lower() if file_context.path else ''
        if filename in self._filename_map:
            candidates.append((
                ChunkerPriority.EXACT_MATCH,
                self._filename_map[filename]
            ))
        
        # Check pattern matches
        if file_context.path:
            for pattern, chunker_name in self._pattern_map:
                if file_context.path.match(pattern):
                    candidates.append((
                        ChunkerPriority.PATTERN_MATCH,
                        chunker_name
                    ))
        
        # Check extension matches
        if file_context.path and file_context.path.suffix:
            ext = file_context.path.suffix.lower()
            for chunker_name in self._extension_map.get(ext, []):
                candidates.append((
                    ChunkerPriority.EXTENSION,
                    chunker_name
                ))
        
        # Check MIME type if available
        if file_context.mime_type:
            for chunker_name in self._mime_map.get(file_context.mime_type, []):
                candidates.append((
                    ChunkerPriority.EXTENSION,
                    chunker_name
                ))
        
        # Content-based selection if enabled
        if self.config.get('enable_content_detection') and file_context.content:
            content_chunker = self._detect_from_content(file_context.content)
            if content_chunker:
                candidates.append((
                    ChunkerPriority.CONTENT_BASED,
                    content_chunker
                ))
        
        # Select best candidate based on priority
        if candidates:
            candidates.sort(key=lambda x: x[0].value)
            selected = candidates[0][1]
            logger.debug(f"Selected chunker '{selected}' for {file_context.path}")
            return selected
        
        return None
    
    def _detect_from_content(self, content: str) -> Optional[str]:
        """
        Detect chunker from file content
        
        Args:
            content: File content
            
        Returns:
            Chunker name or None
        """
        # SQL detection
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE TABLE', 
                       'ALTER TABLE', 'DROP TABLE']
        if any(keyword in content.upper() for keyword in sql_keywords):
            return 'sql'
        
        # PL/SQL detection
        plsql_keywords = ['CREATE OR REPLACE PROCEDURE', 'CREATE OR REPLACE FUNCTION',
                         'CREATE PACKAGE', 'DECLARE', 'BEGIN', 'END;']
        if any(keyword in content.upper() for keyword in plsql_keywords):
            return 'plsql'
        
        # Terraform detection
        if 'resource "' in content or 'provider "' in content or 'module "' in content:
            return 'terraform'
        
        # Docker detection
        if content.startswith('FROM ') or 'RUN ' in content or 'COPY ' in content:
            return 'docker'
        
        # YAML detection
        if content.strip().startswith('---') or ': ' in content:
            # Could be YAML
            try:
                import yaml
                yaml.safe_load(content)
                return 'yaml'
            except:
                pass
        
        # JSON detection
        if content.strip().startswith(('{', '[')):
            try:
                import json
                json.loads(content)
                return 'json'
            except:
                pass
        
        # XML detection
        if content.strip().startswith('<?xml') or content.strip().startswith('<'):
            if '<' in content and '>' in content:
                return 'xml'
        
        # Code detection (generic)
        code_indicators = ['import ', 'function ', 'class ', 'def ', 'public ', 
                          'private ', 'const ', 'var ', 'let ', 'return ']
        if any(indicator in content for indicator in code_indicators):
            return 'generic_code'
        
        return None
    
    def _get_or_create_instance(self, chunker_name: str) -> BaseChunker:
        """
        Get or create a chunker instance
        
        Args:
            chunker_name: Name of the chunker
            
        Returns:
            Chunker instance
        """
        # Check cache if enabled
        if self.config.get('cache_instances', True) and chunker_name in self._instances:
            return self._instances[chunker_name]
        
        # Create new instance
        if chunker_name not in self._chunkers:
            raise ValueError(f"Chunker '{chunker_name}' not found")
        
        chunker_info = self._chunkers[chunker_name]
        
        # Instantiate with tokenizer
        max_tokens = self.config.get('default_max_tokens', 450)
        instance = chunker_info.chunker_class(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens
        )
        
        # Cache if enabled
        if self.config.get('cache_instances', True):
            self._instances[chunker_name] = instance
        
        return instance
    
    def chunk_file(self, file_path: Path, content: Optional[str] = None,
                  force_chunker: Optional[str] = None) -> List[Chunk]:
        """
        Chunk a file using appropriate chunker
        
        Args:
            file_path: Path to file
            content: Optional file content (will read if not provided)
            force_chunker: Optional chunker to force use
            
        Returns:
            List of chunks
        """
        # Read content if not provided
        if content is None:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        # Check file size limit
        max_size = self.config.get('max_file_size', 10 * 1024 * 1024)
        if len(content) > max_size:
            logger.warning(f"File {file_path} exceeds size limit, truncating")
            content = content[:max_size]
        
        # Create file context
        file_context = FileContext(
            path=file_path,
            name=file_path.name,
            extension=file_path.suffix.lower(),
            content=content,
            size=len(content)
        )
        
        # Get appropriate chunker
        chunker = self.get_chunker(file_context, force_chunker)
        
        # Chunk the file
        try:
            chunks = chunker.chunk(content, file_context)
            logger.info(f"Created {len(chunks)} chunks for {file_path} using {chunker.__class__.__name__}")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            # Fallback to text chunker
            if 'text' in self._chunkers:
                fallback = self._get_or_create_instance('text')
                return fallback.chunk(content, file_context)
            else:
                raise
    
    def list_chunkers(self) -> List[Dict[str, Any]]:
        """
        List all registered chunkers
        
        Returns:
            List of chunker information dictionaries
        """
        chunkers = []
        
        for name, info in self._chunkers.items():
            chunkers.append({
                'name': name,
                'class': info.chunker_class.__name__,
                'description': info.description,
                'extensions': list(info.file_extensions),
                'patterns': info.file_patterns,
                'filenames': list(info.exact_filenames),
                'priority': info.priority.name,
                'version': info.version,
                'author': info.author,
                'capabilities': info.capabilities
            })
        
        return chunkers
    
    def get_chunker_info(self, name: str) -> Optional[ChunkerInfo]:
        """
        Get information about a specific chunker
        
        Args:
            name: Chunker name
            
        Returns:
            ChunkerInfo or None if not found
        """
        return self._chunkers.get(name)
    
    def _load_custom_chunkers(self):
        """Load custom chunkers from configured path"""
        custom_path = Path(self.config.get('custom_chunkers_path', 'custom_chunkers'))
        
        if not custom_path.exists():
            return
        
        # Find all Python files in custom chunkers directory
        for py_file in custom_path.glob('**/*.py'):
            if py_file.name.startswith('_'):
                continue
            
            try:
                # Import module
                spec = importlib.util.spec_from_file_location(
                    py_file.stem,
                    py_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find BaseChunker subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseChunker) and obj != BaseChunker:
                        # Register custom chunker
                        self._register_custom_chunker(name, obj, py_file)
                        
            except Exception as e:
                logger.error(f"Failed to load custom chunker from {py_file}: {e}")
    
    def _register_custom_chunker(self, name: str, chunker_class: Type[BaseChunker], 
                                source_file: Path):
        """Register a custom chunker"""
        # Extract metadata from class
        extensions = getattr(chunker_class, 'EXTENSIONS', [])
        patterns = getattr(chunker_class, 'PATTERNS', [])
        filenames = getattr(chunker_class, 'FILENAMES', [])
        description = getattr(chunker_class, '__doc__', '') or f'Custom chunker from {source_file.name}'
        
        # Create ChunkerInfo
        chunker_info = ChunkerInfo(
            name=f'custom_{name.lower()}',
            chunker_class=chunker_class,
            file_extensions=set(extensions),
            file_patterns=patterns,
            exact_filenames=set(filenames),
            mime_types=set(),
            priority=ChunkerPriority.EXTENSION,
            description=description.strip(),
            version='custom',
            author='custom',
            capabilities=[],
            metadata={'source': str(source_file)}
        )
        
        # Register
        self.register(f'custom_{name.lower()}', chunker_info)
        logger.info(f"Registered custom chunker: custom_{name.lower()}")
    
    def register_chunker_class(self, name: str, chunker_class: Type[BaseChunker],
                              extensions: List[str] = None, patterns: List[str] = None,
                              filenames: List[str] = None, priority: str = 'extension'):
        """
        Dynamically register a chunker class
        
        Args:
            name: Name for the chunker
            chunker_class: Chunker class (must inherit from BaseChunker)
            extensions: File extensions to handle
            patterns: File patterns to match
            filenames: Exact filenames to match
            priority: Priority level (exact_match, pattern_match, extension, content_based, fallback)
        """
        if not issubclass(chunker_class, BaseChunker):
            raise ValueError(f"{chunker_class} must inherit from BaseChunker")
        
        priority_map = {
            'exact_match': ChunkerPriority.EXACT_MATCH,
            'pattern_match': ChunkerPriority.PATTERN_MATCH,
            'extension': ChunkerPriority.EXTENSION,
            'content_based': ChunkerPriority.CONTENT_BASED,
            'fallback': ChunkerPriority.FALLBACK
        }
        
        chunker_info = ChunkerInfo(
            name=name,
            chunker_class=chunker_class,
            file_extensions=set(extensions or []),
            file_patterns=patterns or [],
            exact_filenames=set(filenames or []),
            mime_types=set(),
            priority=priority_map.get(priority, ChunkerPriority.EXTENSION),
            description=chunker_class.__doc__ or '',
            version='dynamic',
            author='dynamic',
            capabilities=[]
        )
        
        self.register(name, chunker_info)
    
    def clear_cache(self):
        """Clear chunker instance cache"""
        self._instances.clear()
        logger.debug("Cleared chunker instance cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_chunkers': len(self._chunkers),
            'cached_instances': len(self._instances),
            'extension_mappings': sum(len(v) for v in self._extension_map.values()),
            'filename_mappings': len(self._filename_map),
            'pattern_mappings': len(self._pattern_map),
            'mime_mappings': sum(len(v) for v in self._mime_map.values()),
            'chunkers_by_priority': {
                priority.name: sum(1 for c in self._chunkers.values() if c.priority == priority)
                for priority in ChunkerPriority
            }
        }

# Global registry instance
_registry: Optional[ChunkerRegistry] = None

def get_registry(tokenizer=None) -> ChunkerRegistry:
    """
    Get global chunker registry instance
    
    Args:
        tokenizer: Optional tokenizer for chunkers
        
    Returns:
        ChunkerRegistry instance
    """
    global _registry
    
    if _registry is None:
        _registry = ChunkerRegistry(tokenizer)
    elif tokenizer and _registry.tokenizer != tokenizer:
        # Update tokenizer if different
        _registry.tokenizer = tokenizer
        _registry.clear_cache()
    
    return _registry

def chunk_file(file_path: Path, tokenizer=None, force_chunker: Optional[str] = None) -> List[Chunk]:
    """
    Convenience function to chunk a file
    
    Args:
        file_path: Path to file
        tokenizer: Optional tokenizer
        force_chunker: Optional chunker to force
        
    Returns:
        List of chunks
    """
    registry = get_registry(tokenizer)
    return registry.chunk_file(file_path, force_chunker=force_chunker)