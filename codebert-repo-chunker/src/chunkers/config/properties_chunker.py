"""
Properties file chunker for .properties, .ini, .conf, .cfg, and similar configuration files
Handles Java properties, INI files, environment files, and various config formats
"""

import re
import configparser
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from src.utils.logger import get_logger
from enum import Enum
import chardet
from io import StringIO

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = get_logger(__name__)

class PropertiesFormat(Enum):
    """Types of properties/configuration file formats"""
    JAVA_PROPERTIES = "java_properties"
    INI = "ini"
    ENV = "env"
    CONF = "conf"
    CFG = "cfg"
    TOML = "toml"
    HOCON = "hocon"
    YAML_PROPERTIES = "yaml_properties"
    KEY_VALUE = "key_value"
    APACHE_CONF = "apache_conf"
    NGINX_CONF = "nginx_conf"
    SYSTEMD = "systemd"
    GIT_CONFIG = "git_config"
    SSH_CONFIG = "ssh_config"
    DOCKER_ENV = "docker_env"

@dataclass
class PropertyEntry:
    """Represents a single property entry"""
    key: str
    value: str
    line_number: int
    section: Optional[str] = None
    comment: Optional[str] = None
    is_multiline: bool = False
    is_reference: bool = False
    referenced_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigSection:
    """Represents a configuration section"""
    name: str
    start_line: int
    end_line: int
    properties: List[PropertyEntry]
    comments: List[str]
    subsections: List['ConfigSection']
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PropertiesStructure:
    """Represents the structure of a properties file"""
    format: PropertiesFormat
    sections: List[ConfigSection]
    global_properties: List[PropertyEntry]
    comments: List[Tuple[int, str]]
    total_properties: int
    total_sections: int
    encoding: str
    has_references: bool
    has_environment_vars: bool
    has_multiline: bool
    namespaces: Set[str]
    profiles: Set[str]
    metadata: Dict[str, Any]

class PropertiesAnalyzer:
    """Analyzes properties file structure"""
    
    # Common property patterns
    PATTERNS = {
        # Java properties
        'java_property': re.compile(r'^([^#!=:\s][^=:]*?)\s*[=:]\s*(.*)$'),
        'java_multiline': re.compile(r'\\$'),
        
        # INI format
        'ini_section': re.compile(r'^\[([^\]]+)\]$'),
        'ini_property': re.compile(r'^([^=;\s][^=]*?)\s*=\s*(.*)$'),
        
        # Environment variables
        'env_var': re.compile(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$'),
        'env_export': re.compile(r'^export\s+([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$'),
        
        # Comments
        'hash_comment': re.compile(r'^\s*#(.*)$'),
        'semicolon_comment': re.compile(r'^\s*;(.*)$'),
        'double_slash_comment': re.compile(r'^\s*//(.*)$'),
        'exclamation_comment': re.compile(r'^\s*!(.*)$'),
        
        # References
        'property_reference': re.compile(r'\$\{([^}]+)\}'),
        'env_reference': re.compile(r'\$([A-Z_][A-Z0-9_]*)'),
        'percent_reference': re.compile(r'%([^%]+)%'),
        
        # Special patterns
        'namespace': re.compile(r'^([a-z]+(?:\.[a-z]+)+)\.'),
        'profile': re.compile(r'^([^.]+)\.properties$'),
        'array_index': re.compile(r'\[(\d+)\]'),
        'json_value': re.compile(r'^\s*[\[{].*[\]}]\s*$', re.DOTALL),
        
        # Apache/Nginx config
        'apache_directive': re.compile(r'^(\w+)\s+(.*)$'),
        'apache_section': re.compile(r'^<(\w+)(?:\s+([^>]+))?>$'),
        'apache_section_end': re.compile(r'^</(\w+)>$'),
        
        # HOCON
        'hocon_include': re.compile(r'include\s+"([^"]+)"'),
        'hocon_substitution': re.compile(r'\$\{([^}]+)\}'),
    }
    
    # Format detection helpers
    FORMAT_INDICATORS = {
        PropertiesFormat.JAVA_PROPERTIES: [
            lambda content: '.properties' in str(content),
            lambda content: bool(re.search(r'^[a-z.]+\s*=', content, re.MULTILINE))
        ],
        PropertiesFormat.INI: [
            lambda content: bool(re.search(r'^\[.+\]', content, re.MULTILINE)),
            lambda content: '.ini' in str(content) or '.cfg' in str(content)
        ],
        PropertiesFormat.ENV: [
            lambda content: bool(re.search(r'^[A-Z_]+\s*=', content, re.MULTILINE)),
            lambda content: '.env' in str(content)
        ],
        PropertiesFormat.TOML: [
            lambda content: bool(re.search(r'^\[\[.+\]\]', content, re.MULTILINE)),
            lambda content: '.toml' in str(content)
        ],
        PropertiesFormat.HOCON: [
            lambda content: 'include' in content and '{' in content,
            lambda content: bool(re.search(r'\$\{[^}]+\}', content))
        ],
        PropertiesFormat.APACHE_CONF: [
            lambda content: bool(re.search(r'^<\w+.*>', content, re.MULTILINE)),
            lambda content: 'ServerName' in content or 'DocumentRoot' in content
        ],
        PropertiesFormat.NGINX_CONF: [
            lambda content: 'server {' in content or 'location /' in content,
            lambda content: 'nginx' in str(content).lower()
        ],
        PropertiesFormat.SYSTEMD: [
            lambda content: '[Unit]' in content or '[Service]' in content,
            lambda content: '.service' in str(content) or '.unit' in str(content)
        ],
    }
    
    def __init__(self):
        self.sections = []
        self.properties = []
        self.comments = []
        self.current_section = None
        self.line_number = 0
        
    def analyze_file(self, content: str, file_path: Optional[Path] = None) -> PropertiesStructure:
        """
        Analyze properties file structure
        
        Args:
            content: File content
            file_path: Optional file path for format detection
            
        Returns:
            PropertiesStructure analysis
        """
        # Detect encoding
        encoding = self._detect_encoding(content)
        
        # Detect format
        format_type = self._detect_format(content, file_path)
        
        # Parse based on format
        if format_type == PropertiesFormat.INI:
            return self._analyze_ini(content, encoding)
        elif format_type == PropertiesFormat.JAVA_PROPERTIES:
            return self._analyze_java_properties(content, encoding)
        elif format_type == PropertiesFormat.ENV:
            return self._analyze_env(content, encoding)
        elif format_type in [PropertiesFormat.APACHE_CONF, PropertiesFormat.NGINX_CONF]:
            return self._analyze_server_config(content, encoding, format_type)
        else:
            # Generic key-value parsing
            return self._analyze_key_value(content, encoding, format_type)
    
    def _detect_encoding(self, content: str) -> str:
        """Detect file encoding"""
        if isinstance(content, bytes):
            result = chardet.detect(content)
            return result['encoding'] or 'utf-8'
        return 'utf-8'
    
    def _detect_format(self, content: str, file_path: Optional[Path]) -> PropertiesFormat:
        """Detect properties file format"""
        # Check file extension
        if file_path:
            ext = file_path.suffix.lower()
            name = file_path.name.lower()
            
            if ext == '.properties':
                return PropertiesFormat.JAVA_PROPERTIES
            elif ext in ['.ini', '.cfg']:
                return PropertiesFormat.INI
            elif ext == '.env' or name.startswith('.env'):
                return PropertiesFormat.ENV
            elif ext == '.toml':
                return PropertiesFormat.TOML
            elif ext == '.conf':
                if 'apache' in name or 'httpd' in name:
                    return PropertiesFormat.APACHE_CONF
                elif 'nginx' in name:
                    return PropertiesFormat.NGINX_CONF
                return PropertiesFormat.CONF
            elif ext in ['.service', '.unit']:
                return PropertiesFormat.SYSTEMD
        
        # Check content patterns
        for format_type, indicators in self.FORMAT_INDICATORS.items():
            if all(indicator(content) for indicator in indicators):
                return format_type
        
        # Default to key-value
        return PropertiesFormat.KEY_VALUE
    
    def _analyze_java_properties(self, content: str, encoding: str) -> PropertiesStructure:
        """Analyze Java properties file"""
        lines = content.split('\n')
        properties = []
        comments = []
        namespaces = set()
        profiles = set()
        has_references = False
        has_multiline = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_num = i + 1
            
            # Skip empty lines
            if not line.strip():
                i += 1
                continue
            
            # Check for comments
            comment_match = (
                self.PATTERNS['hash_comment'].match(line) or
                self.PATTERNS['exclamation_comment'].match(line)
            )
            
            if comment_match:
                comments.append((line_num, comment_match.group(1).strip()))
                i += 1
                continue
            
            # Check for property
            prop_match = self.PATTERNS['java_property'].match(line)
            
            if prop_match:
                key = prop_match.group(1).strip()
                value = prop_match.group(2).strip()
                
                # Handle multiline values
                while value.endswith('\\') and i + 1 < len(lines):
                    has_multiline = True
                    value = value[:-1].strip()
                    i += 1
                    next_line = lines[i].strip()
                    value += ' ' + next_line
                
                # Check for references
                ref_matches = self.PATTERNS['property_reference'].findall(value)
                if ref_matches:
                    has_references = True
                
                # Extract namespace
                namespace_match = self.PATTERNS['namespace'].match(key)
                if namespace_match:
                    namespaces.add(namespace_match.group(1))
                
                # Check for profile-specific properties
                if '.' in key:
                    parts = key.split('.')
                    if parts[0] in ['dev', 'test', 'prod', 'staging', 'local']:
                        profiles.add(parts[0])
                
                prop_entry = PropertyEntry(
                    key=key,
                    value=value,
                    line_number=line_num,
                    is_multiline=value.count('\n') > 0,
                    is_reference=bool(ref_matches),
                    referenced_keys=ref_matches,
                    metadata={
                        'namespace': namespace_match.group(1) if namespace_match else None,
                        'is_array': '[' in key and ']' in key,
                        'is_json': self.PATTERNS['json_value'].match(value) is not None
                    }
                )
                
                properties.append(prop_entry)
            
            i += 1
        
        # Group properties by namespace/prefix
        sections = self._group_properties_by_namespace(properties)
        
        return PropertiesStructure(
            format=PropertiesFormat.JAVA_PROPERTIES,
            sections=sections,
            global_properties=[p for p in properties if not p.metadata.get('namespace')],
            comments=comments,
            total_properties=len(properties),
            total_sections=len(sections),
            encoding=encoding,
            has_references=has_references,
            has_environment_vars=False,
            has_multiline=has_multiline,
            namespaces=namespaces,
            profiles=profiles,
            metadata={
                'line_count': len(lines),
                'comment_count': len(comments)
            }
        )
    
    def _analyze_ini(self, content: str, encoding: str) -> PropertiesStructure:
        """Analyze INI file using configparser"""
        try:
            parser = configparser.ConfigParser(allow_no_value=True)
            parser.read_string(content)
            
            sections = []
            global_properties = []
            
            # Process DEFAULT section
            if 'DEFAULT' in parser:
                for key, value in parser['DEFAULT'].items():
                    global_properties.append(PropertyEntry(
                        key=key,
                        value=value or '',
                        line_number=0,
                        section='DEFAULT'
                    ))
            
            # Process other sections
            for section_name in parser.sections():
                properties = []
                
                for key, value in parser[section_name].items():
                    properties.append(PropertyEntry(
                        key=key,
                        value=value or '',
                        line_number=0,
                        section=section_name
                    ))
                
                sections.append(ConfigSection(
                    name=section_name,
                    start_line=0,
                    end_line=0,
                    properties=properties,
                    comments=[],
                    subsections=[],
                    metadata={'item_count': len(properties)}
                ))
            
            return PropertiesStructure(
                format=PropertiesFormat.INI,
                sections=sections,
                global_properties=global_properties,
                comments=[],
                total_properties=sum(len(s.properties) for s in sections) + len(global_properties),
                total_sections=len(sections),
                encoding=encoding,
                has_references=False,
                has_environment_vars=False,
                has_multiline=False,
                namespaces=set(),
                profiles=set(),
                metadata={}
            )
            
        except configparser.Error:
            # Fallback to manual parsing
            return self._analyze_ini_manual(content, encoding)
    
    def _analyze_ini_manual(self, content: str, encoding: str) -> PropertiesStructure:
        """Manually parse INI file when configparser fails"""
        lines = content.split('\n')
        sections = []
        current_section = None
        global_properties = []
        comments = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            line_num = i + 1
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for comments
            if line.startswith(';') or line.startswith('#'):
                comments.append((line_num, line[1:].strip()))
                continue
            
            # Check for section
            section_match = self.PATTERNS['ini_section'].match(line)
            if section_match:
                if current_section:
                    sections.append(current_section)
                
                current_section = ConfigSection(
                    name=section_match.group(1),
                    start_line=line_num,
                    end_line=line_num,
                    properties=[],
                    comments=[],
                    subsections=[]
                )
                continue
            
            # Check for property
            prop_match = self.PATTERNS['ini_property'].match(line)
            if prop_match:
                prop_entry = PropertyEntry(
                    key=prop_match.group(1).strip(),
                    value=prop_match.group(2).strip(),
                    line_number=line_num,
                    section=current_section.name if current_section else None
                )
                
                if current_section:
                    current_section.properties.append(prop_entry)
                    current_section.end_line = line_num
                else:
                    global_properties.append(prop_entry)
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return PropertiesStructure(
            format=PropertiesFormat.INI,
            sections=sections,
            global_properties=global_properties,
            comments=comments,
            total_properties=sum(len(s.properties) for s in sections) + len(global_properties),
            total_sections=len(sections),
            encoding=encoding,
            has_references=False,
            has_environment_vars=False,
            has_multiline=False,
            namespaces=set(),
            profiles=set(),
            metadata={'manual_parse': True}
        )
    
    def _analyze_env(self, content: str, encoding: str) -> PropertiesStructure:
        """Analyze environment file (.env)"""
        lines = content.split('\n')
        properties = []
        comments = []
        has_references = False
        
        for i, line in enumerate(lines):
            line_num = i + 1
            original_line = line
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for comments
            if line.startswith('#'):
                comments.append((line_num, line[1:].strip()))
                continue
            
            # Check for export statement
            export_match = self.PATTERNS['env_export'].match(line)
            if export_match:
                key = export_match.group(1)
                value = export_match.group(2)
            else:
                # Check for regular env var
                env_match = self.PATTERNS['env_var'].match(line)
                if env_match:
                    key = env_match.group(1)
                    value = env_match.group(2)
                else:
                    continue
            
            # Handle quoted values
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            # Check for variable references
            ref_matches = self.PATTERNS['env_reference'].findall(value)
            if ref_matches:
                has_references = True
            
            properties.append(PropertyEntry(
                key=key,
                value=value,
                line_number=line_num,
                is_reference=bool(ref_matches),
                referenced_keys=ref_matches,
                metadata={
                    'is_export': export_match is not None,
                    'is_quoted': value != export_match.group(2) if export_match else False,
                    'is_path': '/' in value or '\\' in value,
                    'is_url': value.startswith(('http://', 'https://', 'ftp://'))
                }
            ))
        
        # Group by category (database, API, app settings, etc.)
        sections = self._categorize_env_vars(properties)
        
        return PropertiesStructure(
            format=PropertiesFormat.ENV,
            sections=sections,
            global_properties=properties,
            comments=comments,
            total_properties=len(properties),
            total_sections=len(sections),
            encoding=encoding,
            has_references=has_references,
            has_environment_vars=True,
            has_multiline=False,
            namespaces=set(),
            profiles=set(),
            metadata={
                'has_exports': any(p.metadata.get('is_export') for p in properties),
                'has_paths': any(p.metadata.get('is_path') for p in properties),
                'has_urls': any(p.metadata.get('is_url') for p in properties)
            }
        )
    
    def _analyze_server_config(self, content: str, encoding: str, 
                              format_type: PropertiesFormat) -> PropertiesStructure:
        """Analyze Apache or Nginx configuration"""
        lines = content.split('\n')
        sections = []
        current_section = None
        section_stack = []
        properties = []
        comments = []
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Check for comments
            if stripped.startswith('#'):
                comments.append((line_num, stripped[1:].strip()))
                continue
            
            # Check for section start (Apache)
            section_match = self.PATTERNS['apache_section'].match(stripped)
            if section_match:
                new_section = ConfigSection(
                    name=section_match.group(1),
                    start_line=line_num,
                    end_line=line_num,
                    properties=[],
                    comments=[],
                    subsections=[],
                    metadata={'args': section_match.group(2)}
                )
                
                if current_section:
                    section_stack.append(current_section)
                    current_section.subsections.append(new_section)
                else:
                    sections.append(new_section)
                
                current_section = new_section
                continue
            
            # Check for section end
            section_end_match = self.PATTERNS['apache_section_end'].match(stripped)
            if section_end_match:
                if current_section:
                    current_section.end_line = line_num
                    if section_stack:
                        current_section = section_stack.pop()
                    else:
                        current_section = None
                continue
            
            # Parse directives
            if current_section or format_type == PropertiesFormat.NGINX_CONF:
                # Split on first whitespace
                parts = stripped.split(None, 1)
                if parts:
                    key = parts[0]
                    value = parts[1] if len(parts) > 1 else ''
                    
                    prop_entry = PropertyEntry(
                        key=key,
                        value=value,
                        line_number=line_num,
                        section=current_section.name if current_section else None,
                        metadata={
                            'is_directive': True,
                            'format': format_type.value
                        }
                    )
                    
                    if current_section:
                        current_section.properties.append(prop_entry)
                    else:
                        properties.append(prop_entry)
        
        return PropertiesStructure(
            format=format_type,
            sections=sections,
            global_properties=properties,
            comments=comments,
            total_properties=len(properties) + sum(self._count_section_properties(s) for s in sections),
            total_sections=len(sections),
            encoding=encoding,
            has_references=False,
            has_environment_vars=False,
            has_multiline=False,
            namespaces=set(),
            profiles=set(),
            metadata={'config_type': format_type.value}
        )
    
    def _analyze_key_value(self, content: str, encoding: str, 
                          format_type: PropertiesFormat) -> PropertiesStructure:
        """Generic key-value parsing"""
        lines = content.split('\n')
        properties = []
        comments = []
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Check for comments (various styles)
            if any(stripped.startswith(c) for c in ['#', ';', '//', '--']):
                comments.append((line_num, stripped.lstrip('#;/-').strip()))
                continue
            
            # Try to parse as key-value
            for sep in ['=', ':', ' ']:
                if sep in stripped:
                    parts = stripped.split(sep, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        properties.append(PropertyEntry(
                            key=key,
                            value=value,
                            line_number=line_num,
                            metadata={'separator': sep}
                        ))
                        break
        
        return PropertiesStructure(
            format=format_type,
            sections=[],
            global_properties=properties,
            comments=comments,
            total_properties=len(properties),
            total_sections=0,
            encoding=encoding,
            has_references=False,
            has_environment_vars=False,
            has_multiline=False,
            namespaces=set(),
            profiles=set(),
            metadata={'generic': True}
        )
    
    def _group_properties_by_namespace(self, properties: List[PropertyEntry]) -> List[ConfigSection]:
        """Group properties by namespace/prefix"""
        namespace_groups = defaultdict(list)
        
        for prop in properties:
            namespace = prop.metadata.get('namespace')
            if namespace:
                namespace_groups[namespace].append(prop)
            else:
                # Try to extract prefix
                if '.' in prop.key:
                    prefix = prop.key.split('.')[0]
                    namespace_groups[prefix].append(prop)
        
        sections = []
        for namespace, props in namespace_groups.items():
            sections.append(ConfigSection(
                name=namespace,
                start_line=min(p.line_number for p in props),
                end_line=max(p.line_number for p in props),
                properties=props,
                comments=[],
                subsections=[],
                metadata={'namespace': True}
            ))
        
        return sections
    
    def _categorize_env_vars(self, properties: List[PropertyEntry]) -> List[ConfigSection]:
        """Categorize environment variables by type"""
        categories = {
            'database': [],
            'api': [],
            'app': [],
            'aws': [],
            'azure': [],
            'gcp': [],
            'auth': [],
            'cache': [],
            'mail': [],
            'logging': [],
            'debug': [],
            'feature': [],
            'other': []
        }
        
        for prop in properties:
            key_lower = prop.key.lower()
            
            if any(db in key_lower for db in ['db', 'database', 'mysql', 'postgres', 'mongo', 'redis']):
                categories['database'].append(prop)
            elif any(api in key_lower for api in ['api', 'endpoint', 'webhook']):
                categories['api'].append(prop)
            elif any(aws in key_lower for aws in ['aws', 's3', 'ec2', 'lambda']):
                categories['aws'].append(prop)
            elif any(azure in key_lower for azure in ['azure', 'blob']):
                categories['azure'].append(prop)
            elif any(gcp in key_lower for gcp in ['gcp', 'google', 'gcs']):
                categories['gcp'].append(prop)
            elif any(auth in key_lower for auth in ['auth', 'token', 'secret', 'key', 'password']):
                categories['auth'].append(prop)
            elif any(cache in key_lower for cache in ['cache', 'memcache']):
                categories['cache'].append(prop)
            elif any(mail in key_lower for mail in ['mail', 'smtp', 'email']):
                categories['mail'].append(prop)
            elif any(log in key_lower for log in ['log', 'sentry', 'bugsnag']):
                categories['logging'].append(prop)
            elif any(debug in key_lower for debug in ['debug', 'verbose']):
                categories['debug'].append(prop)
            elif any(feat in key_lower for feat in ['feature', 'flag', 'enable', 'disable']):
                categories['feature'].append(prop)
            elif 'app' in key_lower or 'application' in key_lower:
                categories['app'].append(prop)
            else:
                categories['other'].append(prop)
        
        sections = []
        for category, props in categories.items():
            if props:
                sections.append(ConfigSection(
                    name=category,
                    start_line=min(p.line_number for p in props),
                    end_line=max(p.line_number for p in props),
                    properties=props,
                    comments=[],
                    subsections=[],
                    metadata={'category': category, 'count': len(props)}
                ))
        
        return sections
    
    def _count_section_properties(self, section: ConfigSection) -> int:
        """Count all properties in section including subsections"""
        count = len(section.properties)
        for subsection in section.subsections:
            count += self._count_section_properties(subsection)
        return count

class PropertiesChunker(BaseChunker):
    """Chunker for properties and configuration files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = PropertiesAnalyzer()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from properties file
        
        Args:
            content: File content
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze structure
            structure = self.analyzer.analyze_file(content, file_context.path)
            
            # Choose chunking strategy based on format and size
            if self._is_small_file(content):
                return self._create_single_chunk(content, structure, file_context)
            
            # Chunk based on format
            if structure.format == PropertiesFormat.JAVA_PROPERTIES:
                return self._chunk_java_properties(content, structure, file_context)
            elif structure.format in [PropertiesFormat.INI, PropertiesFormat.SYSTEMD]:
                return self._chunk_by_sections(content, structure, file_context)
            elif structure.format == PropertiesFormat.ENV:
                return self._chunk_env_file(content, structure, file_context)
            elif structure.format in [PropertiesFormat.APACHE_CONF, PropertiesFormat.NGINX_CONF]:
                return self._chunk_server_config(content, structure, file_context)
            else:
                return self._chunk_generic(content, structure, file_context)
                
        except Exception as e:
            logger.error(f"Error chunking properties file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _is_small_file(self, content: str) -> bool:
        """Check if file is small enough for single chunk"""
        return self.count_tokens(content) <= self.max_tokens
    
    def _create_single_chunk(self, content: str, structure: PropertiesStructure,
                           file_context: FileContext) -> List[Chunk]:
        """Create single chunk for small file"""
        return [self.create_chunk(
            content=content,
            chunk_type='properties_complete',
            metadata={
                'format': structure.format.value,
                'property_count': structure.total_properties,
                'section_count': structure.total_sections,
                'has_references': structure.has_references,
                'encoding': structure.encoding
            },
            file_path=str(file_context.path)
        )]
    
    def _chunk_java_properties(self, content: str, structure: PropertiesStructure,
                              file_context: FileContext) -> List[Chunk]:
        """Chunk Java properties file"""
        chunks = []
        
        # Group by namespace/profile
        if structure.namespaces or structure.profiles:
            # Create chunks for each namespace
            namespace_chunks = self._create_namespace_chunks(structure, file_context)
            chunks.extend(namespace_chunks)
            
            # Create chunks for each profile
            profile_chunks = self._create_profile_chunks(structure, file_context)
            chunks.extend(profile_chunks)
        else:
            # Chunk by property groups
            property_chunks = self._create_property_group_chunks(
                structure.global_properties, 
                file_context
            )
            chunks.extend(property_chunks)
        
        # Add metadata chunk if there are many comments
        if len(structure.comments) > 10:
            comments_chunk = self._create_comments_chunk(structure.comments, file_context)
            if comments_chunk:
                chunks.insert(0, comments_chunk)
        
        return chunks
    
    def _chunk_by_sections(self, content: str, structure: PropertiesStructure,
                          file_context: FileContext) -> List[Chunk]:
        """Chunk INI-style file by sections"""
        chunks = []
        
        # Create chunk for global properties
        if structure.global_properties:
            global_chunk = self._create_section_chunk(
                'GLOBAL',
                structure.global_properties,
                file_context
            )
            chunks.append(global_chunk)
        
        # Create chunk for each section
        for section in structure.sections:
            section_chunk = self._create_section_chunk(
                section.name,
                section.properties,
                file_context
            )
            
            if self.count_tokens(section_chunk.content) > self.max_tokens:
                # Split large section
                split_chunks = self._split_large_section(section, file_context)
                chunks.extend(split_chunks)
            else:
                chunks.append(section_chunk)
        
        return chunks
    
    def _chunk_env_file(self, content: str, structure: PropertiesStructure,
                       file_context: FileContext) -> List[Chunk]:
        """Chunk environment file by category"""
        chunks = []
        
        # Create chunks for each category
        for section in structure.sections:
            if section.properties:
                chunk_content = self._format_env_vars(section.properties)
                
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type=f'env_{section.name}',
                    metadata={
                        'format': 'env',
                        'category': section.name,
                        'var_count': len(section.properties),
                        'has_secrets': section.name in ['auth', 'database']
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks
    
    def _chunk_server_config(self, content: str, structure: PropertiesStructure,
                            file_context: FileContext) -> List[Chunk]:
        """Chunk server configuration file"""
        chunks = []
        
        # Create chunks for each major section
        for section in structure.sections:
            chunk_content = self._reconstruct_section(section, structure.format)
            
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type=f'config_{section.name.lower()}',
                metadata={
                    'format': structure.format.value,
                    'section': section.name,
                    'directive_count': len(section.properties),
                    'has_subsections': len(section.subsections) > 0
                },
                file_path=str(file_context.path),
                start_line=section.start_line,
                end_line=section.end_line
            ))
        
        # Add global directives
        if structure.global_properties:
            global_chunk = self._create_directives_chunk(
                structure.global_properties,
                file_context
            )
            chunks.insert(0, global_chunk)
        
        return chunks
    
    def _chunk_generic(self, content: str, structure: PropertiesStructure,
                      file_context: FileContext) -> List[Chunk]:
        """Generic chunking for unknown formats"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_properties = []
        
        for prop in structure.global_properties:
            prop_line = f"{prop.key}={prop.value}"
            prop_tokens = self.count_tokens(prop_line)
            
            if current_tokens + prop_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='properties_partial',
                    metadata={
                        'format': structure.format.value,
                        'property_count': len(chunk_properties),
                        'properties': [p.key for p in chunk_properties]
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
                chunk_properties = []
            
            current_chunk.append(prop_line)
            current_tokens += prop_tokens
            chunk_properties.append(prop)
        
        # Add remaining properties
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='properties_partial',
                metadata={
                    'format': structure.format.value,
                    'property_count': len(chunk_properties),
                    'properties': [p.key for p in chunk_properties]
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _create_namespace_chunks(self, structure: PropertiesStructure,
                                file_context: FileContext) -> List[Chunk]:
        """Create chunks for each namespace"""
        chunks = []
        
        for section in structure.sections:
            if section.metadata.get('namespace'):
                chunk_content = self._format_properties(section.properties)
                
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='properties_namespace',
                    metadata={
                        'namespace': section.name,
                        'property_count': len(section.properties)
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks
    
    def _create_profile_chunks(self, structure: PropertiesStructure,
                              file_context: FileContext) -> List[Chunk]:
        """Create chunks for each profile"""
        chunks = []
        
        for profile in structure.profiles:
            profile_props = [
                p for p in structure.global_properties
                if p.key.startswith(f"{profile}.")
            ]
            
            if profile_props:
                chunk_content = self._format_properties(profile_props)
                
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='properties_profile',
                    metadata={
                        'profile': profile,
                        'property_count': len(profile_props)
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks
    
    def _create_property_group_chunks(self, properties: List[PropertyEntry],
                                     file_context: FileContext) -> List[Chunk]:
        """Create chunks for property groups"""
        chunks = []
        
        # Group properties by prefix
        prefix_groups = defaultdict(list)
        no_prefix = []
        
        for prop in properties:
            if '.' in prop.key:
                prefix = prop.key.split('.')[0]
                prefix_groups[prefix].append(prop)
            else:
                no_prefix.append(prop)
        
        # Create chunks for each prefix group
        for prefix, props in prefix_groups.items():
            if self._calculate_properties_tokens(props) <= self.max_tokens:
                chunk_content = self._format_properties(props)
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='properties_group',
                    metadata={
                        'prefix': prefix,
                        'property_count': len(props)
                    },
                    file_path=str(file_context.path)
                ))
            else:
                # Split large group
                split_chunks = self._split_property_group(props, prefix, file_context)
                chunks.extend(split_chunks)
        
        # Add properties without prefix
        if no_prefix:
            chunk_content = self._format_properties(no_prefix)
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type='properties_misc',
                metadata={'property_count': len(no_prefix)},
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _create_section_chunk(self, name: str, properties: List[PropertyEntry],
                             file_context: FileContext) -> Chunk:
        """Create chunk for a section"""
        lines = []
        
        if name != 'GLOBAL':
            lines.append(f"[{name}]")
        
        for prop in properties:
            lines.append(f"{prop.key}={prop.value}")
        
        return self.create_chunk(
            content='\n'.join(lines),
            chunk_type='properties_section',
            metadata={
                'section': name,
                'property_count': len(properties)
            },
            file_path=str(file_context.path)
        )
    
    def _create_comments_chunk(self, comments: List[Tuple[int, str]],
                              file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for comments"""
        if not comments:
            return None
        
        lines = ["# File Comments and Documentation", ""]
        
        for line_num, comment in comments:
            lines.append(f"# Line {line_num}: {comment}")
        
        return self.create_chunk(
            content='\n'.join(lines),
            chunk_type='properties_comments',
            metadata={'comment_count': len(comments)},
            file_path=str(file_context.path)
        )
    
    def _create_directives_chunk(self, properties: List[PropertyEntry],
                                file_context: FileContext) -> Chunk:
        """Create chunk for server directives"""
        lines = []
        
        for prop in properties:
            lines.append(f"{prop.key} {prop.value}")
        
        return self.create_chunk(
            content='\n'.join(lines),
            chunk_type='config_directives',
            metadata={'directive_count': len(properties)},
            file_path=str(file_context.path)
        )
    
    def _split_large_section(self, section: ConfigSection,
                            file_context: FileContext) -> List[Chunk]:
        """Split large section into smaller chunks"""
        chunks = []
        current_props = []
        current_tokens = 0
        
        section_header = f"[{section.name}]\n"
        header_tokens = self.count_tokens(section_header)
        
        for prop in section.properties:
            prop_line = f"{prop.key}={prop.value}"
            prop_tokens = self.count_tokens(prop_line)
            
            if header_tokens + current_tokens + prop_tokens > self.max_tokens and current_props:
                chunk_content = section_header + self._format_properties(current_props)
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='properties_section_part',
                    metadata={
                        'section': section.name,
                        'property_count': len(current_props),
                        'is_partial': True
                    },
                    file_path=str(file_context.path)
                ))
                current_props = []
                current_tokens = 0
            
            current_props.append(prop)
            current_tokens += prop_tokens
        
        # Add remaining properties
        if current_props:
            chunk_content = section_header + self._format_properties(current_props)
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type='properties_section_part',
                metadata={
                    'section': section.name,
                    'property_count': len(current_props),
                    'is_partial': len(chunks) > 0
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _split_property_group(self, properties: List[PropertyEntry], prefix: str,
                             file_context: FileContext) -> List[Chunk]:
        """Split large property group"""
        chunks = []
        current_props = []
        current_tokens = 0
        part = 1
        
        for prop in properties:
            prop_line = f"{prop.key}={prop.value}"
            prop_tokens = self.count_tokens(prop_line)
            
            if current_tokens + prop_tokens > self.max_tokens and current_props:
                chunk_content = self._format_properties(current_props)
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='properties_group_part',
                    metadata={
                        'prefix': prefix,
                        'part': part,
                        'property_count': len(current_props)
                    },
                    file_path=str(file_context.path)
                ))
                current_props = []
                current_tokens = 0
                part += 1
            
            current_props.append(prop)
            current_tokens += prop_tokens
        
        # Add remaining properties
        if current_props:
            chunk_content = self._format_properties(current_props)
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type='properties_group_part',
                metadata={
                    'prefix': prefix,
                    'part': part,
                    'property_count': len(current_props)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _format_properties(self, properties: List[PropertyEntry]) -> str:
        """Format properties as text"""
        lines = []
        
        for prop in properties:
            # Add comment if present
            if prop.comment:
                lines.append(f"# {prop.comment}")
            
            # Add property
            lines.append(f"{prop.key}={prop.value}")
        
        return '\n'.join(lines)
    
    def _format_env_vars(self, properties: List[PropertyEntry]) -> str:
        """Format environment variables"""
        lines = []
        
        for prop in properties:
            # Use export if it was originally exported
            if prop.metadata.get('is_export'):
                lines.append(f"export {prop.key}={prop.value}")
            else:
                lines.append(f"{prop.key}={prop.value}")
        
        return '\n'.join(lines)
    
    def _reconstruct_section(self, section: ConfigSection,
                           format_type: PropertiesFormat) -> str:
        """Reconstruct section content"""
        lines = []
        
        if format_type in [PropertiesFormat.APACHE_CONF]:
            # Apache-style section
            if section.metadata.get('args'):
                lines.append(f"<{section.name} {section.metadata['args']}>")
            else:
                lines.append(f"<{section.name}>")
            
            for prop in section.properties:
                lines.append(f"    {prop.key} {prop.value}")
            
            for subsection in section.subsections:
                sub_content = self._reconstruct_section(subsection, format_type)
                for line in sub_content.split('\n'):
                    lines.append(f"    {line}")
            
            lines.append(f"</{section.name}>")
        else:
            # Generic section
            lines.append(f"[{section.name}]")
            for prop in section.properties:
                lines.append(f"{prop.key}={prop.value}")
        
        return '\n'.join(lines)
    
    def _calculate_properties_tokens(self, properties: List[PropertyEntry]) -> int:
        """Calculate total tokens for properties"""
        total = 0
        for prop in properties:
            total += self.count_tokens(f"{prop.key}={prop.value}")
        return total
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback line-based chunking"""
        logger.warning(f"Using fallback chunking for {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='properties_fallback',
                    metadata={'is_fallback': True},
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining lines
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='properties_fallback',
                metadata={'is_fallback': True},
                file_path=str(file_context.path)
            ))
        
        return chunks