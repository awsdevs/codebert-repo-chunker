"""
Content analyzer for intelligent file type detection and content classification
Uses multiple techniques including patterns, entropy, statistical analysis, and ML features
"""

import re
import magic
import chardet
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import statistics
import math
import hashlib
import mimetypes
import json
import yaml
import xml.etree.ElementTree as ET
from datetime import datetime
import logging

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
import langdetect
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class ContentType(Enum):
    """Types of content detected"""
    # Code
    SOURCE_CODE = "source_code"
    COMPILED_CODE = "compiled_code"
    ASSEMBLY = "assembly"
    BYTECODE = "bytecode"
    
    # Data
    STRUCTURED_DATA = "structured_data"
    TABULAR_DATA = "tabular_data"
    TIME_SERIES = "time_series"
    GEOSPATIAL = "geospatial"
    
    # Documents
    DOCUMENTATION = "documentation"
    SPECIFICATION = "specification"
    API_SPEC = "api_spec"
    README = "readme"
    
    # Configuration
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"
    SETTINGS = "settings"
    CREDENTIALS = "credentials"
    
    # Logs
    APPLICATION_LOG = "application_log"
    SYSTEM_LOG = "system_log"
    ACCESS_LOG = "access_log"
    ERROR_LOG = "error_log"
    
    # Database
    SQL_SCRIPT = "sql_script"
    MIGRATION = "migration"
    SCHEMA = "schema"
    SEED_DATA = "seed_data"
    
    # Web
    MARKUP = "markup"
    STYLESHEET = "stylesheet"
    TEMPLATE = "template"
    STATIC_ASSET = "static_asset"
    
    # Binary
    EXECUTABLE = "executable"
    LIBRARY = "library"
    ARCHIVE = "archive"
    MEDIA = "media"
    
    # Text
    PLAIN_TEXT = "plain_text"
    FORMATTED_TEXT = "formatted_text"
    NATURAL_LANGUAGE = "natural_language"
    
    # Special
    ENCRYPTED = "encrypted"
    COMPRESSED = "compressed"
    SERIALIZED = "serialized"
    UNKNOWN = "unknown"

class ProgrammingLanguage(Enum):
    """Programming languages detected"""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    SQL = "sql"
    PLSQL = "plsql"
    SHELL = "shell"
    POWERSHELL = "powershell"
    PERL = "perl"
    LUA = "lua"
    HASKELL = "haskell"
    CLOJURE = "clojure"
    ERLANG = "erlang"
    ELIXIR = "elixir"
    JULIA = "julia"
    FORTRAN = "fortran"
    COBOL = "cobol"
    ASSEMBLY = "assembly"
    UNKNOWN = "unknown"

@dataclass
class ContentFeatures:
    """Features extracted from content"""
    # Basic
    size: int
    line_count: int
    avg_line_length: float
    max_line_length: int
    empty_line_ratio: float
    
    # Character statistics
    whitespace_ratio: float
    alphanumeric_ratio: float
    special_char_ratio: float
    digit_ratio: float
    uppercase_ratio: float
    
    # Entropy
    shannon_entropy: float
    normalized_entropy: float
    
    # Structural
    indentation_type: Optional[str]  # 'spaces', 'tabs', 'mixed', None
    indentation_size: Optional[int]
    has_consistent_indentation: bool
    bracket_balance: bool
    quote_balance: bool
    
    # Patterns
    has_headers: bool
    has_functions: bool
    has_classes: bool
    has_imports: bool
    has_comments: bool
    has_docstrings: bool
    has_urls: bool
    has_emails: bool
    has_paths: bool
    has_timestamps: bool
    has_uuids: bool
    has_ip_addresses: bool
    has_json_structure: bool
    has_xml_structure: bool
    has_yaml_structure: bool
    has_sql_keywords: bool
    has_html_tags: bool
    
    # Language detection
    natural_language: Optional[str]
    natural_language_confidence: float
    
    # Statistical
    word_count: int
    unique_words: int
    vocabulary_richness: float
    avg_word_length: float
    sentence_count: int
    avg_sentence_length: float
    
    # Keywords and tokens
    top_keywords: List[str]
    programming_keywords: Set[str]
    file_extensions_mentioned: Set[str]
    
    # Binary detection
    is_binary: bool
    has_null_bytes: bool
    non_printable_ratio: float
    
    # Encoding
    detected_encoding: str
    encoding_confidence: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentAnalysis:
    """Complete content analysis result"""
    content_type: ContentType
    confidence: float
    programming_language: Optional[ProgrammingLanguage]
    language_confidence: float
    features: ContentFeatures
    mime_type: Optional[str]
    file_format: Optional[str]
    suggested_chunker: Optional[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class ContentAnalyzer:
    """Analyzes file content to determine type and characteristics"""
    
    # Programming language patterns
    LANGUAGE_PATTERNS = {
        ProgrammingLanguage.PYTHON: {
            'keywords': {'def', 'class', 'import', 'from', 'if', 'elif', 'else', 
                        'for', 'while', 'with', 'as', 'try', 'except', 'finally',
                        'lambda', 'yield', 'return', 'async', 'await', '__init__'},
            'patterns': [
                re.compile(r'^import\s+\w+', re.MULTILINE),
                re.compile(r'^from\s+\w+\s+import', re.MULTILINE),
                re.compile(r'^def\s+\w+\s*\(', re.MULTILINE),
                re.compile(r'^class\s+\w+[\(\:]', re.MULTILINE),
                re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']'),
            ],
            'extensions': {'.py', '.pyw', '.pyx', '.pxd'}
        },
        ProgrammingLanguage.JAVA: {
            'keywords': {'public', 'private', 'protected', 'class', 'interface', 
                        'extends', 'implements', 'import', 'package', 'static',
                        'void', 'final', 'abstract', 'synchronized', 'volatile'},
            'patterns': [
                re.compile(r'^package\s+[\w.]+;', re.MULTILINE),
                re.compile(r'^import\s+[\w.]+;', re.MULTILINE),
                re.compile(r'public\s+class\s+\w+'),
                re.compile(r'public\s+static\s+void\s+main'),
                re.compile(r'@Override|@Autowired|@Component|@Service'),
            ],
            'extensions': {'.java', '.jsp', '.jspx'}
        },
        ProgrammingLanguage.JAVASCRIPT: {
            'keywords': {'function', 'var', 'let', 'const', 'if', 'else', 'for',
                        'while', 'return', 'async', 'await', 'import', 'export',
                        'require', 'module', 'exports', 'class', 'extends'},
            'patterns': [
                re.compile(r'function\s+\w+\s*\('),
                re.compile(r'const\s+\w+\s*=\s*(?:function|\(|\{)'),
                re.compile(r'(?:var|let|const)\s+\w+\s*='),
                re.compile(r'require\s*\(["\']'),
                re.compile(r'import\s+.*\s+from\s+["\']'),
                re.compile(r'export\s+(?:default\s+)?(?:function|class|const)'),
            ],
            'extensions': {'.js', '.jsx', '.mjs', '.cjs'}
        },
        ProgrammingLanguage.SQL: {
            'keywords': {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP',
                        'ALTER', 'TABLE', 'FROM', 'WHERE', 'JOIN', 'GROUP BY',
                        'ORDER BY', 'HAVING', 'UNION', 'INDEX', 'VIEW', 'PROCEDURE'},
            'patterns': [
                re.compile(r'\bSELECT\s+.*\s+FROM\s+', re.IGNORECASE),
                re.compile(r'\bINSERT\s+INTO\s+', re.IGNORECASE),
                re.compile(r'\bCREATE\s+TABLE\s+', re.IGNORECASE),
                re.compile(r'\bALTER\s+TABLE\s+', re.IGNORECASE),
                re.compile(r'\bDROP\s+TABLE\s+', re.IGNORECASE),
            ],
            'extensions': {'.sql', '.ddl', '.dml'}
        },
        ProgrammingLanguage.GO: {
            'keywords': {'package', 'import', 'func', 'var', 'const', 'type',
                        'struct', 'interface', 'if', 'else', 'for', 'range',
                        'switch', 'case', 'return', 'defer', 'go', 'chan'},
            'patterns': [
                re.compile(r'^package\s+\w+', re.MULTILINE),
                re.compile(r'^import\s+(?:\(|")'),
                re.compile(r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\('),
                re.compile(r'^type\s+\w+\s+struct\s*\{'),
            ],
            'extensions': {'.go'}
        },
    }
    
    # Content type indicators
    CONTENT_INDICATORS = {
        ContentType.CONFIGURATION: {
            'patterns': [
                re.compile(r'^\s*[a-z_][a-z0-9_]*\s*[=:]\s*', re.MULTILINE | re.IGNORECASE),
                re.compile(r'^\s*\[[\w\s]+\]\s*$', re.MULTILINE),  # INI sections
                re.compile(r'^[A-Z_]+=[^=]+$', re.MULTILINE),  # Environment variables
            ],
            'keywords': ['config', 'settings', 'options', 'parameters'],
            'extensions': {'.ini', '.cfg', '.conf', '.config', '.properties', '.env'}
        },
        ContentType.APPLICATION_LOG: {
            'patterns': [
                re.compile(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'),  # Timestamps
                re.compile(r'\b(ERROR|WARN|INFO|DEBUG|TRACE|FATAL)\b'),
                re.compile(r'^\[\d{4}-\d{2}-\d{2}', re.MULTILINE),
                re.compile(r'Exception|Error|Warning|Stack trace', re.IGNORECASE),
            ],
            'keywords': ['error', 'warning', 'info', 'debug', 'exception', 'stack trace'],
            'extensions': {'.log', '.out', '.err'}
        },
        ContentType.STRUCTURED_DATA: {
            'patterns': [
                re.compile(r'^\s*\{[\s\S]*\}\s*$'),  # JSON
                re.compile(r'^\s*<\?xml'),  # XML
                re.compile(r'^---\s*$', re.MULTILINE),  # YAML
            ],
            'keywords': [],
            'extensions': {'.json', '.xml', '.yaml', '.yml'}
        },
        ContentType.TABULAR_DATA: {
            'patterns': [
                re.compile(r'^[^,\t]+(?:[,\t][^,\t]+)+$', re.MULTILINE),  # CSV/TSV
                re.compile(r'^\|.*\|.*\|', re.MULTILINE),  # Markdown tables
            ],
            'keywords': [],
            'extensions': {'.csv', '.tsv', '.tab'}
        },
    }
    
    def __init__(self):
        """Initialize content analyzer"""
        self.magic = None
        try:
            self.magic = magic.Magic(mime=True)
        except:
            logger.warning("python-magic not available, MIME detection limited")
        
        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(
            max_features=20,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
    def analyze(self, content: str, file_path: Optional[Path] = None,
               check_binary: bool = True) -> ContentAnalysis:
        """
        Analyze content to determine type and characteristics
        
        Args:
            content: File content as string
            file_path: Optional file path for additional context
            check_binary: Whether to check for binary content
            
        Returns:
            ContentAnalysis object with results
        """
        try:
            # Extract features
            features = self._extract_features(content, check_binary)
            
            # Detect MIME type
            mime_type = self._detect_mime_type(content, file_path)
            
            # Determine content type
            content_type, type_confidence = self._determine_content_type(
                content, features, mime_type, file_path
            )
            
            # Detect programming language if code
            prog_language = None
            lang_confidence = 0.0
            
            if content_type == ContentType.SOURCE_CODE:
                prog_language, lang_confidence = self._detect_programming_language(
                    content, features, file_path
                )
            
            # Determine file format
            file_format = self._determine_file_format(content, mime_type, file_path)
            
            # Suggest appropriate chunker
            suggested_chunker = self._suggest_chunker(
                content_type, prog_language, file_format, features
            )
            
            # Generate warnings
            warnings = self._generate_warnings(features, content_type)
            
            # Create metadata
            metadata = {
                'analyzed_at': datetime.now().isoformat(),
                'file_path': str(file_path) if file_path else None,
                'content_hash': hashlib.md5(content.encode()).hexdigest(),
            }
            
            return ContentAnalysis(
                content_type=content_type,
                confidence=type_confidence,
                programming_language=prog_language,
                language_confidence=lang_confidence,
                features=features,
                mime_type=mime_type,
                file_format=file_format,
                suggested_chunker=suggested_chunker,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            # Return minimal analysis on error
            return ContentAnalysis(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                programming_language=None,
                language_confidence=0.0,
                features=self._get_default_features(),
                mime_type=None,
                file_format=None,
                suggested_chunker='text',
                warnings=[f"Analysis error: {str(e)}"],
                metadata={'error': str(e)}
            )
    
    def _extract_features(self, content: str, check_binary: bool = True) -> ContentFeatures:
        """Extract features from content"""
        lines = content.split('\n')
        
        # Basic metrics
        size = len(content)
        line_count = len(lines)
        line_lengths = [len(line) for line in lines]
        avg_line_length = statistics.mean(line_lengths) if line_lengths else 0
        max_line_length = max(line_lengths) if line_lengths else 0
        empty_lines = sum(1 for line in lines if not line.strip())
        empty_line_ratio = empty_lines / line_count if line_count > 0 else 0
        
        # Character statistics
        total_chars = len(content)
        whitespace = sum(1 for c in content if c.isspace())
        alphanumeric = sum(1 for c in content if c.isalnum())
        digits = sum(1 for c in content if c.isdigit())
        uppercase = sum(1 for c in content if c.isupper())
        special = total_chars - whitespace - alphanumeric
        
        whitespace_ratio = whitespace / total_chars if total_chars > 0 else 0
        alphanumeric_ratio = alphanumeric / total_chars if total_chars > 0 else 0
        special_char_ratio = special / total_chars if total_chars > 0 else 0
        digit_ratio = digits / total_chars if total_chars > 0 else 0
        uppercase_ratio = uppercase / total_chars if total_chars > 0 else 0
        
        # Entropy calculation
        shannon_entropy = self._calculate_shannon_entropy(content)
        normalized_entropy = shannon_entropy / 8.0  # Normalize to [0, 1]
        
        # Structural analysis
        indentation = self._analyze_indentation(lines)
        bracket_balance = self._check_bracket_balance(content)
        quote_balance = self._check_quote_balance(content)
        
        # Pattern detection
        patterns = self._detect_patterns(content)
        
        # Natural language detection
        natural_lang, lang_confidence = self._detect_natural_language(content)
        
        # Text statistics
        text_stats = self._calculate_text_statistics(content)
        
        # Keyword extraction
        keywords = self._extract_keywords(content)
        prog_keywords = self._detect_programming_keywords(content)
        file_exts = self._detect_file_extensions(content)
        
        # Binary detection
        is_binary = False
        has_null = b'\x00' in content.encode('utf-8', errors='ignore')
        non_printable = sum(1 for c in content if ord(c) < 32 and c not in '\t\n\r')
        non_printable_ratio = non_printable / total_chars if total_chars > 0 else 0
        
        if check_binary and (has_null or non_printable_ratio > 0.3):
            is_binary = True
        
        # Encoding detection
        encoding_info = self._detect_encoding(content)
        
        return ContentFeatures(
            size=size,
            line_count=line_count,
            avg_line_length=avg_line_length,
            max_line_length=max_line_length,
            empty_line_ratio=empty_line_ratio,
            whitespace_ratio=whitespace_ratio,
            alphanumeric_ratio=alphanumeric_ratio,
            special_char_ratio=special_char_ratio,
            digit_ratio=digit_ratio,
            uppercase_ratio=uppercase_ratio,
            shannon_entropy=shannon_entropy,
            normalized_entropy=normalized_entropy,
            indentation_type=indentation['type'],
            indentation_size=indentation['size'],
            has_consistent_indentation=indentation['consistent'],
            bracket_balance=bracket_balance,
            quote_balance=quote_balance,
            has_headers=patterns['has_headers'],
            has_functions=patterns['has_functions'],
            has_classes=patterns['has_classes'],
            has_imports=patterns['has_imports'],
            has_comments=patterns['has_comments'],
            has_docstrings=patterns['has_docstrings'],
            has_urls=patterns['has_urls'],
            has_emails=patterns['has_emails'],
            has_paths=patterns['has_paths'],
            has_timestamps=patterns['has_timestamps'],
            has_uuids=patterns['has_uuids'],
            has_ip_addresses=patterns['has_ip_addresses'],
            has_json_structure=patterns['has_json_structure'],
            has_xml_structure=patterns['has_xml_structure'],
            has_yaml_structure=patterns['has_yaml_structure'],
            has_sql_keywords=patterns['has_sql_keywords'],
            has_html_tags=patterns['has_html_tags'],
            natural_language=natural_lang,
            natural_language_confidence=lang_confidence,
            word_count=text_stats['word_count'],
            unique_words=text_stats['unique_words'],
            vocabulary_richness=text_stats['vocabulary_richness'],
            avg_word_length=text_stats['avg_word_length'],
            sentence_count=text_stats['sentence_count'],
            avg_sentence_length=text_stats['avg_sentence_length'],
            top_keywords=keywords,
            programming_keywords=prog_keywords,
            file_extensions_mentioned=file_exts,
            is_binary=is_binary,
            has_null_bytes=has_null,
            non_printable_ratio=non_printable_ratio,
            detected_encoding=encoding_info['encoding'],
            encoding_confidence=encoding_info['confidence']
        )
    
    def _calculate_shannon_entropy(self, content: str) -> float:
        """Calculate Shannon entropy of content"""
        if not content:
            return 0.0
        
        # Count character frequencies
        char_counts = Counter(content)
        total = len(content)
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _analyze_indentation(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze indentation patterns"""
        indentations = []
        
        for line in lines:
            if line and line[0] in ' \t':
                # Count leading whitespace
                spaces = 0
                tabs = 0
                for char in line:
                    if char == ' ':
                        spaces += 1
                    elif char == '\t':
                        tabs += 1
                    else:
                        break
                
                indentations.append({'spaces': spaces, 'tabs': tabs})
        
        if not indentations:
            return {'type': None, 'size': None, 'consistent': True}
        
        # Determine type
        has_spaces = any(i['spaces'] > 0 for i in indentations)
        has_tabs = any(i['tabs'] > 0 for i in indentations)
        
        if has_spaces and has_tabs:
            indent_type = 'mixed'
        elif has_tabs:
            indent_type = 'tabs'
        elif has_spaces:
            indent_type = 'spaces'
        else:
            indent_type = None
        
        # Determine size (for spaces)
        indent_size = None
        if indent_type == 'spaces':
            space_counts = [i['spaces'] for i in indentations if i['spaces'] > 0]
            if space_counts:
                # Find GCD of all indentation levels
                from math import gcd
                from functools import reduce
                indent_size = reduce(gcd, space_counts)
        
        # Check consistency
        consistent = True
        if indent_type == 'mixed':
            consistent = False
        elif indent_type == 'spaces' and indent_size:
            consistent = all(i['spaces'] % indent_size == 0 for i in indentations if i['spaces'] > 0)
        
        return {
            'type': indent_type,
            'size': indent_size,
            'consistent': consistent
        }
    
    def _check_bracket_balance(self, content: str) -> bool:
        """Check if brackets are balanced"""
        brackets = {
            '(': ')',
            '[': ']',
            '{': '}'
        }
        stack = []
        
        for char in content:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack[-1] != char:
                    return False
                stack.pop()
        
        return len(stack) == 0
    
    def _check_quote_balance(self, content: str) -> bool:
        """Check if quotes are balanced"""
        single_quotes = content.count("'")
        double_quotes = content.count('"')
        
        # Check if even number (paired)
        return single_quotes % 2 == 0 and double_quotes % 2 == 0
    
    def _detect_patterns(self, content: str) -> Dict[str, bool]:
        """Detect various patterns in content"""
        patterns = {
            'has_headers': False,
            'has_functions': False,
            'has_classes': False,
            'has_imports': False,
            'has_comments': False,
            'has_docstrings': False,
            'has_urls': False,
            'has_emails': False,
            'has_paths': False,
            'has_timestamps': False,
            'has_uuids': False,
            'has_ip_addresses': False,
            'has_json_structure': False,
            'has_xml_structure': False,
            'has_yaml_structure': False,
            'has_sql_keywords': False,
            'has_html_tags': False,
        }
        
        # Header patterns (Markdown, etc.)
        if re.search(r'^#{1,6}\s+\w+', content, re.MULTILINE):
            patterns['has_headers'] = True
        
        # Function patterns
        if re.search(r'\b(def|function|func|fn)\s+\w+\s*\(', content):
            patterns['has_functions'] = True
        
        # Class patterns
        if re.search(r'\b(class|struct|interface|type)\s+\w+', content):
            patterns['has_classes'] = True
        
        # Import patterns
        if re.search(r'\b(import|require|include|using|use)\s+[\w.\'"]+', content):
            patterns['has_imports'] = True
        
        # Comment patterns
        if re.search(r'(//|#|/\*|\*|<!--)', content):
            patterns['has_comments'] = True
        
        # Docstring patterns
        if re.search(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', content):
            patterns['has_docstrings'] = True
        
        # URLs
        if re.search(r'https?://[^\s]+', content):
            patterns['has_urls'] = True
        
        # Emails
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content):
            patterns['has_emails'] = True
        
        # File paths
        if re.search(r'([a-zA-Z]:)?[/\\][\w./\\]+', content):
            patterns['has_paths'] = True
        
        # Timestamps
        if re.search(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', content):
            patterns['has_timestamps'] = True
        
        # UUIDs
        if re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', content, re.IGNORECASE):
            patterns['has_uuids'] = True
        
        # IP addresses
        if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', content):
            patterns['has_ip_addresses'] = True
        
        # JSON structure
        if content.strip().startswith(('{', '[')) and content.strip().endswith(('}', ']')):
            try:
                json.loads(content)
                patterns['has_json_structure'] = True
            except:
                pass
        
        # XML structure
        if content.strip().startswith('<') and '>' in content:
            try:
                ET.fromstring(content)
                patterns['has_xml_structure'] = True
            except:
                pass
        
        # YAML structure
        if re.search(r'^[\w-]+:\s*[\w\s]*$', content, re.MULTILINE):
            try:
                yaml.safe_load(content)
                patterns['has_yaml_structure'] = True
            except:
                pass
        
        # SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        if any(keyword in content.upper() for keyword in sql_keywords):
            patterns['has_sql_keywords'] = True
        
        # HTML tags
        if re.search(r'<[^>]+>', content):
            patterns['has_html_tags'] = True
        
        return patterns
    
    def _detect_natural_language(self, content: str) -> Tuple[Optional[str], float]:
        """Detect natural language in content"""
        try:
            # Remove code-like patterns for better detection
            text = re.sub(r'[{}\[\]()<>]', ' ', content)
            text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
            text = ' '.join(text.split())[:500]  # Use first 500 chars
            
            if len(text.strip()) < 20:
                return None, 0.0
            
            result = langdetect.detect_langs(text)
            if result:
                best = result[0]
                return best.lang, best.prob
            
        except:
            pass
        
        return None, 0.0
    
    def _calculate_text_statistics(self, content: str) -> Dict[str, Any]:
        """Calculate text-based statistics"""
        stats = {
            'word_count': 0,
            'unique_words': 0,
            'vocabulary_richness': 0.0,
            'avg_word_length': 0.0,
            'sentence_count': 0,
            'avg_sentence_length': 0.0
        }
        
        try:
            # Tokenize words
            words = word_tokenize(content.lower())
            words = [w for w in words if w.isalpha() and len(w) > 2]
            
            if words:
                stats['word_count'] = len(words)
                stats['unique_words'] = len(set(words))
                stats['vocabulary_richness'] = stats['unique_words'] / stats['word_count']
                stats['avg_word_length'] = statistics.mean(len(w) for w in words)
            
            # Tokenize sentences
            sentences = sent_tokenize(content)
            if sentences:
                stats['sentence_count'] = len(sentences)
                stats['avg_sentence_length'] = statistics.mean(len(s.split()) for s in sentences)
            
        except:
            pass
        
        return stats
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract top keywords from content"""
        try:
            # Remove special characters
            text = re.sub(r'[^\w\s]', ' ', content)
            text = ' '.join(text.split())
            
            if not text.strip():
                return []
            
            # Use TF-IDF
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = scores.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return keywords[:10]
            
        except:
            return []
    
    def _detect_programming_keywords(self, content: str) -> Set[str]:
        """Detect programming language keywords"""
        found_keywords = set()
        
        for lang, info in self.LANGUAGE_PATTERNS.items():
            for keyword in info['keywords']:
                if re.search(r'\b' + keyword + r'\b', content):
                    found_keywords.add(keyword)
        
        return found_keywords
    
    def _detect_file_extensions(self, content: str) -> Set[str]:
        """Detect file extensions mentioned in content"""
        extensions = set()
        
        # Common file extension pattern
        ext_pattern = re.compile(r'\b\w+\.([a-z]{2,4})\b', re.IGNORECASE)
        
        for match in ext_pattern.finditer(content):
            ext = '.' + match.group(1).lower()
            # Filter to known extensions
            if ext in {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.go',
                      '.rs', '.rb', '.php', '.sql', '.html', '.css', '.xml',
                      '.json', '.yaml', '.yml', '.txt', '.md', '.log', '.csv'}:
                extensions.add(ext)
        
        return extensions
    
    def _detect_encoding(self, content: str) -> Dict[str, Any]:
        """Detect content encoding"""
        try:
            # Try to encode as bytes and detect
            result = chardet.detect(content.encode('utf-8', errors='ignore'))
            return {
                'encoding': result.get('encoding', 'utf-8'),
                'confidence': result.get('confidence', 0.0)
            }
        except:
            return {'encoding': 'utf-8', 'confidence': 0.5}
    
    def _detect_mime_type(self, content: str, file_path: Optional[Path]) -> Optional[str]:
        """Detect MIME type"""
        mime_type = None
        
        # Try from file extension
        if file_path:
            guessed = mimetypes.guess_type(str(file_path))[0]
            if guessed:
                mime_type = guessed
        
        # Try magic if available
        if self.magic and not mime_type:
            try:
                mime_type = self.magic.from_buffer(content.encode('utf-8', errors='ignore'))
            except:
                pass
        
        return mime_type
    
    def _determine_content_type(self, content: str, features: ContentFeatures,
                               mime_type: Optional[str], 
                               file_path: Optional[Path]) -> Tuple[ContentType, float]:
        """Determine content type with confidence"""
        scores = defaultdict(float)
        
        # Check binary first
        if features.is_binary:
            if features.normalized_entropy > 0.95:
                return ContentType.ENCRYPTED, 0.9
            elif features.normalized_entropy > 0.85:
                return ContentType.COMPRESSED, 0.8
            else:
                return ContentType.COMPILED_CODE, 0.7
        
        # Check each content type
        for ctype, indicators in self.CONTENT_INDICATORS.items():
            score = 0.0
            
            # Check patterns
            for pattern in indicators['patterns']:
                if pattern.search(content):
                    score += 0.3
            
            # Check keywords
            content_lower = content.lower()
            for keyword in indicators['keywords']:
                if keyword in content_lower:
                    score += 0.1
            
            # Check extensions
            if file_path and file_path.suffix in indicators['extensions']:
                score += 0.5
            
            scores[ctype] = min(score, 1.0)
        
        # Check for source code
        if features.has_functions or features.has_classes or features.has_imports:
            scores[ContentType.SOURCE_CODE] += 0.6
        
        # Check for documentation
        if features.has_headers and features.natural_language:
            scores[ContentType.DOCUMENTATION] += 0.5
        
        # Check for structured data
        if features.has_json_structure:
            scores[ContentType.STRUCTURED_DATA] += 0.8
        elif features.has_xml_structure:
            scores[ContentType.STRUCTURED_DATA] += 0.8
        elif features.has_yaml_structure:
            scores[ContentType.STRUCTURED_DATA] += 0.7
        
        # Check for SQL
        if features.has_sql_keywords:
            scores[ContentType.SQL_SCRIPT] += 0.7
        
        # Get best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            
            if confidence > 0.3:
                return best_type, confidence
        
        # Default based on other indicators
        if features.programming_keywords:
            return ContentType.SOURCE_CODE, 0.5
        elif features.natural_language and features.natural_language_confidence > 0.7:
            return ContentType.NATURAL_LANGUAGE, features.natural_language_confidence
        else:
            return ContentType.PLAIN_TEXT, 0.3
    
    def _detect_programming_language(self, content: str, features: ContentFeatures,
                                    file_path: Optional[Path]) -> Tuple[Optional[ProgrammingLanguage], float]:
        """Detect programming language"""
        scores = defaultdict(float)
        
        for lang, info in self.LANGUAGE_PATTERNS.items():
            score = 0.0
            
            # Check file extension
            if file_path and file_path.suffix in info['extensions']:
                score += 0.5
            
            # Check keywords
            content_words = set(re.findall(r'\b\w+\b', content))
            keyword_matches = len(info['keywords'].intersection(content_words))
            score += min(keyword_matches * 0.1, 0.4)
            
            # Check patterns
            for pattern in info['patterns']:
                if pattern.search(content):
                    score += 0.2
            
            scores[lang] = min(score, 1.0)
        
        # Get best match
        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = scores[best_lang]
            
            if confidence > 0.3:
                return best_lang, confidence
        
        return None, 0.0
    
    def _determine_file_format(self, content: str, mime_type: Optional[str],
                              file_path: Optional[Path]) -> Optional[str]:
        """Determine specific file format"""
        # Check file extension
        if file_path:
            ext = file_path.suffix.lower()
            if ext:
                return ext[1:]  # Remove leading dot
        
        # Check MIME type
        if mime_type:
            if 'json' in mime_type:
                return 'json'
            elif 'xml' in mime_type:
                return 'xml'
            elif 'yaml' in mime_type:
                return 'yaml'
            elif 'html' in mime_type:
                return 'html'
        
        # Content-based detection
        if content.strip().startswith('{') and content.strip().endswith('}'):
            return 'json'
        elif content.strip().startswith('<'):
            return 'xml'
        elif re.search(r'^---\s*$', content, re.MULTILINE):
            return 'yaml'
        
        return None
    
    def _suggest_chunker(self, content_type: ContentType,
                        prog_language: Optional[ProgrammingLanguage],
                        file_format: Optional[str],
                        features: ContentFeatures) -> Optional[str]:
        """Suggest appropriate chunker based on analysis"""
        # Programming language specific
        if prog_language:
            language_chunkers = {
                ProgrammingLanguage.PYTHON: 'python',
                ProgrammingLanguage.JAVA: 'java',
                ProgrammingLanguage.JAVASCRIPT: 'javascript',
                ProgrammingLanguage.TYPESCRIPT: 'typescript',
                ProgrammingLanguage.SQL: 'sql',
                ProgrammingLanguage.PLSQL: 'plsql',
            }
            if prog_language in language_chunkers:
                return language_chunkers[prog_language]
        
        # File format specific
        if file_format:
            format_chunkers = {
                'json': 'json',
                'xml': 'xml',
                'yaml': 'yaml',
                'yml': 'yaml',
                'tf': 'terraform',
                'hcl': 'terraform',
                'properties': 'properties',
                'ini': 'properties',
                'md': 'markdown',
            }
            if file_format in format_chunkers:
                return format_chunkers[file_format]
        
        # Content type specific
        content_chunkers = {
            ContentType.SOURCE_CODE: 'generic_code',
            ContentType.CONFIGURATION: 'properties',
            ContentType.STRUCTURED_DATA: 'json',
            ContentType.SQL_SCRIPT: 'sql',
            ContentType.APPLICATION_LOG: 'text',
            ContentType.DOCUMENTATION: 'markdown',
            ContentType.NATURAL_LANGUAGE: 'text',
        }
        
        if content_type in content_chunkers:
            return content_chunkers[content_type]
        
        # Adaptive chunker for unknown
        if features.normalized_entropy > 0.7:
            return 'adaptive'
        
        return 'text'
    
    def _generate_warnings(self, features: ContentFeatures, 
                          content_type: ContentType) -> List[str]:
        """Generate warnings based on analysis"""
        warnings = []
        
        # File size warnings
        if features.size > 10 * 1024 * 1024:
            warnings.append("Very large file (>10MB), chunking may be slow")
        
        # Binary content warning
        if features.is_binary:
            warnings.append("Binary content detected, text chunking may not be appropriate")
        
        # High entropy warning
        if features.normalized_entropy > 0.9:
            warnings.append("Very high entropy, content may be encrypted or compressed")
        
        # Long lines warning
        if features.max_line_length > 1000:
            warnings.append("Very long lines detected, may affect chunking quality")
        
        # Mixed indentation warning
        if features.indentation_type == 'mixed':
            warnings.append("Mixed indentation detected (tabs and spaces)")
        
        # Unbalanced brackets
        if not features.bracket_balance:
            warnings.append("Unbalanced brackets detected")
        
        # Unknown content type
        if content_type == ContentType.UNKNOWN:
            warnings.append("Content type could not be determined reliably")
        
        return warnings
    
    def _get_default_features(self) -> ContentFeatures:
        """Get default features for error cases"""
        return ContentFeatures(
            size=0,
            line_count=0,
            avg_line_length=0,
            max_line_length=0,
            empty_line_ratio=0,
            whitespace_ratio=0,
            alphanumeric_ratio=0,
            special_char_ratio=0,
            digit_ratio=0,
            uppercase_ratio=0,
            shannon_entropy=0,
            normalized_entropy=0,
            indentation_type=None,
            indentation_size=None,
            has_consistent_indentation=False,
            bracket_balance=False,
            quote_balance=False,
            has_headers=False,
            has_functions=False,
            has_classes=False,
            has_imports=False,
            has_comments=False,
            has_docstrings=False,
            has_urls=False,
            has_emails=False,
            has_paths=False,
            has_timestamps=False,
            has_uuids=False,
            has_ip_addresses=False,
            has_json_structure=False,
            has_xml_structure=False,
            has_yaml_structure=False,
            has_sql_keywords=False,
            has_html_tags=False,
            natural_language=None,
            natural_language_confidence=0,
            word_count=0,
            unique_words=0,
            vocabulary_richness=0,
            avg_word_length=0,
            sentence_count=0,
            avg_sentence_length=0,
            top_keywords=[],
            programming_keywords=set(),
            file_extensions_mentioned=set(),
            is_binary=False,
            has_null_bytes=False,
            non_printable_ratio=0,
            detected_encoding='utf-8',
            encoding_confidence=0
        )

def analyze_content(content: str, file_path: Optional[Path] = None) -> ContentAnalysis:
    """
    Convenience function to analyze content
    
    Args:
        content: Content to analyze
        file_path: Optional file path for context
        
    Returns:
        ContentAnalysis object
    """
    analyzer = ContentAnalyzer()
    return analyzer.analyze(content, file_path)