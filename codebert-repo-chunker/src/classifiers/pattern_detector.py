"""
Pattern detector for identifying code patterns, anti-patterns, and architectural patterns
Provides deep pattern analysis for code quality assessment and refactoring opportunities
"""

import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import json
import yaml
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns detected"""
    # Design Patterns
    SINGLETON = "singleton"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    FACADE = "facade"
    PROXY = "proxy"
    COMMAND = "command"
    ITERATOR = "iterator"
    TEMPLATE_METHOD = "template_method"
    
    # Anti-Patterns
    GOD_CLASS = "god_class"
    SPAGHETTI_CODE = "spaghetti_code"
    COPY_PASTE = "copy_paste"
    DEAD_CODE = "dead_code"
    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMP = "data_clump"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    SWITCH_STATEMENT = "switch_statement"
    PARALLEL_INHERITANCE = "parallel_inheritance"
    LAZY_CLASS = "lazy_class"
    SPECULATIVE_GENERALITY = "speculative_generality"
    TEMPORARY_FIELD = "temporary_field"
    MESSAGE_CHAINS = "message_chains"
    MIDDLE_MAN = "middle_man"
    
    # Code Smells
    DUPLICATE_CODE = "duplicate_code"
    LONG_PARAMETER_LIST = "long_parameter_list"
    DIVERGENT_CHANGE = "divergent_change"
    SHOTGUN_SURGERY = "shotgun_surgery"
    REFUSED_BEQUEST = "refused_bequest"
    COMMENTS_SMELL = "comments_smell"
    MAGIC_NUMBERS = "magic_numbers"
    COMPLEX_CONDITIONAL = "complex_conditional"
    
    # Security Patterns
    SQL_INJECTION = "sql_injection"
    XSS_VULNERABILITY = "xss_vulnerability"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    WEAK_ENCRYPTION = "weak_encryption"
    INSECURE_RANDOM = "insecure_random"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    
    # Performance Patterns
    N_PLUS_ONE = "n_plus_one"
    MEMORY_LEAK = "memory_leak"
    INEFFICIENT_LOOP = "inefficient_loop"
    EXCESSIVE_SYNCHRONIZATION = "excessive_synchronization"
    PREMATURE_OPTIMIZATION = "premature_optimization"
    CACHE_MISS = "cache_miss"
    
    # Architectural Patterns
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    LAYERED = "layered"
    MICROSERVICE = "microservice"
    EVENT_DRIVEN = "event_driven"
    PIPELINE = "pipeline"
    REPOSITORY = "repository"
    DEPENDENCY_INJECTION = "dependency_injection"
    
    # Testing Patterns
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    MOCK_PATTERN = "mock_pattern"
    TEST_FIXTURE = "test_fixture"
    ASSERTION_ROULETTE = "assertion_roulette"
    
    # Concurrency Patterns
    THREAD_SAFE = "thread_safe"
    RACE_CONDITION = "race_condition"
    DEADLOCK_RISK = "deadlock_risk"
    PRODUCER_CONSUMER = "producer_consumer"
    
    # Framework Patterns
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    WEBSOCKET = "websocket"
    ORM_PATTERN = "orm_pattern"
    MIDDLEWARE = "middleware"

class PatternSeverity(Enum):
    """Severity levels for patterns"""
    CRITICAL = 5  # Security vulnerabilities, major bugs
    HIGH = 4      # Performance issues, bad practices
    MEDIUM = 3    # Code smells, minor issues
    LOW = 2       # Style issues, suggestions
    INFO = 1      # Informational patterns

@dataclass
class PatternMatch:
    """Represents a pattern match in code"""
    pattern_type: PatternType
    severity: PatternSeverity
    confidence: float
    location: Dict[str, Any]  # file, line, column
    description: str
    suggestion: Optional[str]
    code_snippet: Optional[str]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternAnalysis:
    """Complete pattern analysis result"""
    file_path: Path
    patterns_found: List[PatternMatch]
    design_patterns: List[PatternMatch]
    anti_patterns: List[PatternMatch]
    code_smells: List[PatternMatch]
    security_issues: List[PatternMatch]
    performance_issues: List[PatternMatch]
    complexity_metrics: Dict[str, float]
    quality_score: float
    refactoring_suggestions: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]

class PatternDetector:
    """Detects patterns, anti-patterns, and code smells"""
    
    # Pattern detection rules
    PATTERN_RULES = {
        # Design Patterns
        PatternType.SINGLETON: {
            'indicators': [
                r'class\s+\w+.*:\s*\n.*_instance\s*=\s*None',
                r'@classmethod\s+def\s+get_instance',
                r'private\s+static\s+\w+\s+instance',
                r'if\s+instance\s*==\s*null',
            ],
            'severity': PatternSeverity.INFO,
            'description': 'Singleton pattern detected'
        },
        
        PatternType.FACTORY: {
            'indicators': [
                r'class\s+\w*Factory',
                r'def\s+create_\w+',
                r'public\s+\w+\s+create\w+\(',
                r'switch.*case.*return\s+new',
            ],
            'severity': PatternSeverity.INFO,
            'description': 'Factory pattern detected'
        },
        
        # Anti-Patterns
        PatternType.GOD_CLASS: {
            'metrics': {
                'lines': 1000,
                'methods': 30,
                'attributes': 20
            },
            'severity': PatternSeverity.HIGH,
            'description': 'Class has too many responsibilities'
        },
        
        PatternType.LONG_METHOD: {
            'metrics': {
                'lines': 100,
                'complexity': 15
            },
            'severity': PatternSeverity.MEDIUM,
            'description': 'Method is too long and complex'
        },
        
        PatternType.COPY_PASTE: {
            'indicators': [
                # Detected through similarity analysis
            ],
            'severity': PatternSeverity.MEDIUM,
            'description': 'Duplicate code detected'
        },
        
        # Security Patterns
        PatternType.SQL_INJECTION: {
            'indicators': [
                r'query\s*=\s*["\'].*\+.*["\']',
                r'execute\(["\'].*%s.*["\'].*%',
                r'SELECT.*\+\s*\w+\s*\+',
                r'f["\']SELECT.*{.*}',
            ],
            'severity': PatternSeverity.CRITICAL,
            'description': 'Potential SQL injection vulnerability'
        },
        
        PatternType.HARDCODED_CREDENTIALS: {
            'indicators': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            'severity': PatternSeverity.CRITICAL,
            'description': 'Hardcoded credentials detected'
        },
        
        PatternType.XSS_VULNERABILITY: {
            'indicators': [
                r'innerHTML\s*=.*\+',
                r'document\.write\(.*\+',
                r'eval\(.*user',
                r'dangerouslySetInnerHTML',
            ],
            'severity': PatternSeverity.CRITICAL,
            'description': 'Potential XSS vulnerability'
        },
        
        # Performance Patterns
        PatternType.N_PLUS_ONE: {
            'indicators': [
                r'for.*:\s*\n.*\.objects\.get\(',
                r'for.*:\s*\n.*\.query\(',
                r'\.map\(.*async.*await.*fetch',
            ],
            'severity': PatternSeverity.HIGH,
            'description': 'N+1 query problem detected'
        },
        
        PatternType.INEFFICIENT_LOOP: {
            'indicators': [
                r'for.*in.*range\(len\(',
                r'while.*len\(.*\)',
                r'for.*:\s*\n.*append\(',
            ],
            'severity': PatternSeverity.MEDIUM,
            'description': 'Inefficient loop pattern'
        },
        
        # Code Smells
        PatternType.MAGIC_NUMBERS: {
            'indicators': [
                r'if\s+\w+\s*[><=]+\s*\d{2,}',
                r'return\s+\d{2,}',
                r'=\s*\d{2,}[^0-9]',
            ],
            'severity': PatternSeverity.LOW,
            'description': 'Magic numbers without constants'
        },
        
        PatternType.COMPLEX_CONDITIONAL: {
            'indicators': [
                r'if.*and.*and.*and',
                r'if.*or.*or.*or',
                r'if.*\(.*\(.*\(.*\)',
            ],
            'severity': PatternSeverity.MEDIUM,
            'description': 'Complex conditional logic'
        },
    }
    
    # Complexity thresholds
    COMPLEXITY_THRESHOLDS = {
        'cyclomatic_complexity': 10,
        'cognitive_complexity': 15,
        'nesting_depth': 4,
        'parameter_count': 5,
        'return_count': 5,
        'line_count': 50,
        'branch_count': 10,
    }
    
    # Similarity threshold for duplicate detection
    SIMILARITY_THRESHOLD = 0.85
    
    def __init__(self):
        """Initialize pattern detector"""
        self.statistics = defaultdict(int)
        self._init_advanced_patterns()
        
    def _init_advanced_patterns(self):
        """Initialize advanced pattern detection"""
        # Framework-specific patterns
        self.framework_patterns = {
            'django': {
                'models': r'class\s+\w+\(.*Model.*\):',
                'views': r'class\s+\w+View\(.*View.*\):',
                'serializers': r'class\s+\w+Serializer\(',
            },
            'flask': {
                'routes': r'@app\.route\(',
                'blueprints': r'Blueprint\(',
            },
            'spring': {
                'controllers': r'@Controller|@RestController',
                'services': r'@Service',
                'repositories': r'@Repository',
            },
            'react': {
                'components': r'class\s+\w+\s+extends\s+(?:React\.)?Component',
                'hooks': r'use[A-Z]\w+',
            },
        }
        
        # Security vulnerability patterns
        self.vulnerability_patterns = {
            'path_traversal': r'\.\./|\.\.\\',
            'command_injection': r'os\.system|subprocess\.call.*shell=True|eval\(|exec\(',
            'weak_random': r'random\.random\(\)|Math\.random\(\)',
            'unsafe_yaml': r'yaml\.load\(|pickle\.loads?\(',
            'open_redirect': r'redirect\(request\..*\)',
        }
        
    def analyze(self, content: str, file_path: Optional[Path] = None,
               language: Optional[str] = None) -> PatternAnalysis:
        """
        Analyze content for patterns
        
        Args:
            content: File content
            file_path: Optional file path
            language: Optional programming language
            
        Returns:
            PatternAnalysis object
        """
        try:
            # Detect language if not provided
            if not language and file_path:
                language = self._detect_language(file_path)
            
            # Find all patterns
            patterns_found = []
            
            # Detect design patterns
            design_patterns = self._detect_design_patterns(content, language)
            patterns_found.extend(design_patterns)
            
            # Detect anti-patterns
            anti_patterns = self._detect_anti_patterns(content, language)
            patterns_found.extend(anti_patterns)
            
            # Detect code smells
            code_smells = self._detect_code_smells(content, language)
            patterns_found.extend(code_smells)
            
            # Detect security issues
            security_issues = self._detect_security_issues(content, language)
            patterns_found.extend(security_issues)
            
            # Detect performance issues
            performance_issues = self._detect_performance_issues(content, language)
            patterns_found.extend(performance_issues)
            
            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity(content, language)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                patterns_found, complexity_metrics
            )
            
            # Generate refactoring suggestions
            refactoring_suggestions = self._generate_refactoring_suggestions(
                patterns_found, complexity_metrics
            )
            
            # Generate statistics
            statistics = self._generate_statistics(patterns_found)
            
            # Create metadata
            metadata = {
                'analyzed_at': datetime.now().isoformat(),
                'language': language,
                'file_path': str(file_path) if file_path else None,
                'content_size': len(content),
                'line_count': content.count('\n') + 1,
            }
            
            return PatternAnalysis(
                file_path=file_path,
                patterns_found=patterns_found,
                design_patterns=design_patterns,
                anti_patterns=anti_patterns,
                code_smells=code_smells,
                security_issues=security_issues,
                performance_issues=performance_issues,
                complexity_metrics=complexity_metrics,
                quality_score=quality_score,
                refactoring_suggestions=refactoring_suggestions,
                statistics=statistics,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return self._create_empty_analysis(file_path, str(e))
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file path"""
        extension = file_path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
        }
        
        return language_map.get(extension)
    
    def _detect_design_patterns(self, content: str, 
                               language: Optional[str]) -> List[PatternMatch]:
        """Detect design patterns"""
        patterns = []
        
        design_pattern_types = [
            PatternType.SINGLETON,
            PatternType.FACTORY,
            PatternType.BUILDER,
            PatternType.OBSERVER,
            PatternType.STRATEGY,
            PatternType.DECORATOR,
            PatternType.REPOSITORY,
            PatternType.DEPENDENCY_INJECTION,
        ]
        
        for pattern_type in design_pattern_types:
            if pattern_type in self.PATTERN_RULES:
                rule = self.PATTERN_RULES[pattern_type]
                matches = self._find_pattern_matches(content, pattern_type, rule)
                patterns.extend(matches)
        
        # Language-specific pattern detection
        if language == 'python':
            patterns.extend(self._detect_python_patterns(content))
        elif language == 'java':
            patterns.extend(self._detect_java_patterns(content))
        elif language == 'javascript':
            patterns.extend(self._detect_javascript_patterns(content))
        
        return patterns
    
    def _detect_anti_patterns(self, content: str,
                            language: Optional[str]) -> List[PatternMatch]:
        """Detect anti-patterns"""
        patterns = []
        lines = content.split('\n')
        
        # God class detection
        if self._is_class_file(content, language):
            class_metrics = self._analyze_class_metrics(content, language)
            if class_metrics['lines'] > 1000 or class_metrics['methods'] > 30:
                patterns.append(PatternMatch(
                    pattern_type=PatternType.GOD_CLASS,
                    severity=PatternSeverity.HIGH,
                    confidence=0.8,
                    location={'line': 1, 'column': 0},
                    description='Class is too large and has too many responsibilities',
                    suggestion='Consider breaking this class into smaller, focused classes',
                    code_snippet=None,
                    metrics=class_metrics
                ))
        
        # Long method detection
        methods = self._extract_methods(content, language)
        for method in methods:
            if method['lines'] > 50:
                patterns.append(PatternMatch(
                    pattern_type=PatternType.LONG_METHOD,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.9,
                    location={'line': method['start_line'], 'column': 0},
                    description=f"Method '{method['name']}' is too long ({method['lines']} lines)",
                    suggestion='Consider extracting parts into smaller methods',
                    code_snippet=None,
                    metrics={'lines': method['lines']}
                ))
        
        # Duplicate code detection
        duplicates = self._find_duplicates(content)
        for duplicate in duplicates:
            patterns.append(PatternMatch(
                pattern_type=PatternType.DUPLICATE_CODE,
                severity=PatternSeverity.MEDIUM,
                confidence=duplicate['similarity'],
                location={'line': duplicate['line'], 'column': 0},
                description='Duplicate code block detected',
                suggestion='Extract common code into a reusable function',
                code_snippet=duplicate['snippet'],
                metrics={'similarity': duplicate['similarity']}
            ))
        
        return patterns
    
    def _detect_code_smells(self, content: str,
                          language: Optional[str]) -> List[PatternMatch]:
        """Detect code smells"""
        patterns = []
        
        # Magic numbers
        magic_number_pattern = re.compile(r'(?<!["\'])\b(?!0|1|2|10|100|1000)\d{2,}\b(?!["\'])')
        for match in magic_number_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            patterns.append(PatternMatch(
                pattern_type=PatternType.MAGIC_NUMBERS,
                severity=PatternSeverity.LOW,
                confidence=0.7,
                location={'line': line_num, 'column': match.start()},
                description=f'Magic number {match.group()} without constant',
                suggestion='Define as a named constant',
                code_snippet=match.group(),
                metrics={'value': match.group()}
            ))
        
        # Complex conditionals
        complex_condition_pattern = re.compile(
            r'if\s*\((?:[^()]*\([^()]*\))*[^()]*(?:&&|\|\||and|or).*(?:&&|\|\||and|or).*\)'
        )
        for match in complex_condition_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            patterns.append(PatternMatch(
                pattern_type=PatternType.COMPLEX_CONDITIONAL,
                severity=PatternSeverity.MEDIUM,
                confidence=0.8,
                location={'line': line_num, 'column': match.start()},
                description='Complex conditional expression',
                suggestion='Consider extracting into a well-named boolean method',
                code_snippet=match.group()[:100],
                metrics={'complexity': match.group().count('and') + match.group().count('or')}
            ))
        
        # Long parameter lists
        patterns.extend(self._detect_long_parameter_lists(content, language))
        
        # Comments smell (TODO, FIXME, HACK)
        comment_smell_pattern = re.compile(
            r'(#|//|/\*)\s*(TODO|FIXME|HACK|XXX|BUG)[\s:]+(.+)',
            re.IGNORECASE
        )
        for match in comment_smell_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            patterns.append(PatternMatch(
                pattern_type=PatternType.COMMENTS_SMELL,
                severity=PatternSeverity.LOW,
                confidence=1.0,
                location={'line': line_num, 'column': match.start()},
                description=f'{match.group(2)} comment found: {match.group(3)[:50]}',
                suggestion='Address the issue or create a proper ticket',
                code_snippet=match.group(),
                metrics={'type': match.group(2)}
            ))
        
        return patterns
    
    def _detect_security_issues(self, content: str,
                              language: Optional[str]) -> List[PatternMatch]:
        """Detect security vulnerabilities"""
        patterns = []
        
        # SQL injection
        sql_patterns = [
            r'query\s*=\s*["\'].*\+.*["\']',
            r'execute\(["\'].*%\s*["\'].*%\s*\(',
            r'f["\'](?:SELECT|INSERT|UPDATE|DELETE).*{',
        ]
        
        for pattern in sql_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count('\n') + 1
                patterns.append(PatternMatch(
                    pattern_type=PatternType.SQL_INJECTION,
                    severity=PatternSeverity.CRITICAL,
                    confidence=0.8,
                    location={'line': line_num, 'column': match.start()},
                    description='Potential SQL injection vulnerability',
                    suggestion='Use parameterized queries or prepared statements',
                    code_snippet=match.group()[:100],
                    metrics={'pattern': pattern}
                ))
        
        # Hardcoded credentials
        credential_patterns = [
            r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{6,}["\']',
            r'(?:api[_-]?key|apikey)\s*=\s*["\'][^"\']{10,}["\']',
            r'(?:secret|token)\s*=\s*["\'][^"\']{10,}["\']',
        ]
        
        for pattern in credential_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Skip if it looks like a placeholder
                value = match.group()
                if not any(placeholder in value.lower() for placeholder in 
                          ['example', 'your_', 'xxx', '***', 'change_me', 'placeholder']):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.HARDCODED_CREDENTIALS,
                        severity=PatternSeverity.CRITICAL,
                        confidence=0.7,
                        location={'line': line_num, 'column': match.start()},
                        description='Hardcoded credentials detected',
                        suggestion='Use environment variables or secure credential storage',
                        code_snippet=re.sub(r'["\'][^"\']+["\']', '"***"', match.group()),
                        metrics={'type': 'credential'}
                    ))
        
        # XSS vulnerabilities
        if language in ['javascript', 'typescript']:
            xss_patterns = [
                r'innerHTML\s*=.*(?:\+|`)',
                r'document\.write\([^)]*(?:\+|`)',
                r'eval\([^)]*(?:user|input|request)',
            ]
            
            for pattern in xss_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.XSS_VULNERABILITY,
                        severity=PatternSeverity.CRITICAL,
                        confidence=0.7,
                        location={'line': line_num, 'column': match.start()},
                        description='Potential XSS vulnerability',
                        suggestion='Sanitize user input and use safe DOM manipulation methods',
                        code_snippet=match.group()[:100],
                        metrics={'pattern': pattern}
                    ))
        
        # Check for other vulnerabilities
        for vuln_type, pattern in self.vulnerability_patterns.items():
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                patterns.append(PatternMatch(
                    pattern_type=PatternType.COMMAND_INJECTION,
                    severity=PatternSeverity.HIGH,
                    confidence=0.6,
                    location={'line': line_num, 'column': match.start()},
                    description=f'Potential {vuln_type.replace("_", " ")} vulnerability',
                    suggestion='Review and secure this code',
                    code_snippet=match.group()[:100],
                    metrics={'vulnerability': vuln_type}
                ))
        
        return patterns
    
    def _detect_performance_issues(self, content: str,
                                  language: Optional[str]) -> List[PatternMatch]:
        """Detect performance issues"""
        patterns = []
        
        # N+1 query problem
        n_plus_one_patterns = [
            r'for\s+.*:\s*\n\s*.*\.(get|filter|all|select)\(',  # ORM in loop
            r'\.map\([^)]*=>[^}]*fetch\(',  # Fetch in map
            r'while.*:\s*\n\s*.*query\(',  # Query in while loop
        ]
        
        for pattern in n_plus_one_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[:match.start()].count('\n') + 1
                patterns.append(PatternMatch(
                    pattern_type=PatternType.N_PLUS_ONE,
                    severity=PatternSeverity.HIGH,
                    confidence=0.7,
                    location={'line': line_num, 'column': match.start()},
                    description='Potential N+1 query problem',
                    suggestion='Use eager loading or batch queries',
                    code_snippet=match.group()[:100],
                    metrics={'pattern': pattern}
                ))
        
        # Inefficient loops
        if language == 'python':
            # String concatenation in loop
            string_concat = re.compile(r'for.*:\s*\n\s*.*\+=\s*["\']')
            for match in string_concat.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                patterns.append(PatternMatch(
                    pattern_type=PatternType.INEFFICIENT_LOOP,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.8,
                    location={'line': line_num, 'column': match.start()},
                    description='String concatenation in loop',
                    suggestion='Use list.append() and "".join() instead',
                    code_snippet=match.group()[:100],
                    metrics={'type': 'string_concatenation'}
                ))
        
        # Memory leak patterns
        memory_leak_patterns = [
            r'addEventListener\([^)]+\)',  # Event listeners without removal
            r'setInterval\([^)]+\)',  # Intervals without clearInterval
            r'new\s+\w+\([^)]*\)(?!.*(?:delete|dispose|close))',  # Objects without cleanup
        ]
        
        if language in ['javascript', 'typescript']:
            for pattern in memory_leak_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.MEMORY_LEAK,
                        severity=PatternSeverity.MEDIUM,
                        confidence=0.5,
                        location={'line': line_num, 'column': match.start()},
                        description='Potential memory leak',
                        suggestion='Ensure proper cleanup/disposal',
                        code_snippet=match.group()[:100],
                        metrics={'pattern': pattern}
                    ))
        
        return patterns
    
    def _find_pattern_matches(self, content: str, pattern_type: PatternType,
                            rule: Dict[str, Any]) -> List[PatternMatch]:
        """Find matches for a specific pattern"""
        matches = []
        
        if 'indicators' in rule:
            for indicator in rule['indicators']:
                pattern = re.compile(indicator, re.MULTILINE | re.IGNORECASE)
                for match in pattern.finditer(content):
                    line_num = content[:match.start()].count('\n') + 1
                    matches.append(PatternMatch(
                        pattern_type=pattern_type,
                        severity=rule.get('severity', PatternSeverity.INFO),
                        confidence=0.7,
                        location={'line': line_num, 'column': match.start()},
                        description=rule.get('description', f'{pattern_type.value} detected'),
                        suggestion=rule.get('suggestion'),
                        code_snippet=match.group()[:100],
                        metrics={'indicator': indicator}
                    ))
        
        return matches
    
    def _detect_python_patterns(self, content: str) -> List[PatternMatch]:
        """Detect Python-specific patterns"""
        patterns = []
        
        try:
            # Parse AST for more accurate detection
            tree = ast.parse(content)
            
            # Detect decorators (which might indicate patterns)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if decorator.id in ['property', 'staticmethod', 'classmethod']:
                                patterns.append(PatternMatch(
                                    pattern_type=PatternType.DECORATOR,
                                    severity=PatternSeverity.INFO,
                                    confidence=1.0,
                                    location={'line': node.lineno, 'column': node.col_offset},
                                    description=f'Decorator pattern: @{decorator.id}',
                                    suggestion=None,
                                    code_snippet=None,
                                    metrics={'decorator': decorator.id}
                                ))
                
                # Detect context managers (with statement)
                elif isinstance(node, ast.With):
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.TEMPLATE_METHOD,
                        severity=PatternSeverity.INFO,
                        confidence=0.8,
                        location={'line': node.lineno, 'column': node.col_offset},
                        description='Context manager pattern detected',
                        suggestion=None,
                        code_snippet=None,
                        metrics={'type': 'context_manager'}
                    ))
        except:
            pass  # AST parsing failed
        
        return patterns
    
    def _detect_java_patterns(self, content: str) -> List[PatternMatch]:
        """Detect Java-specific patterns"""
        patterns = []
        
        # Spring annotations
        spring_annotations = [
            '@Controller', '@Service', '@Repository', '@Component',
            '@Autowired', '@Bean', '@Configuration'
        ]
        
        for annotation in spring_annotations:
            if annotation in content:
                line_num = content.index(annotation)
                line_num = content[:line_num].count('\n') + 1
                patterns.append(PatternMatch(
                    pattern_type=PatternType.DEPENDENCY_INJECTION,
                    severity=PatternSeverity.INFO,
                    confidence=1.0,
                    location={'line': line_num, 'column': 0},
                    description=f'Spring {annotation} pattern',
                    suggestion=None,
                    code_snippet=annotation,
                    metrics={'annotation': annotation}
                ))
        
        return patterns
    
    def _detect_javascript_patterns(self, content: str) -> List[PatternMatch]:
        """Detect JavaScript-specific patterns"""
        patterns = []
        
        # React patterns
        react_patterns = [
            (r'useState\(', 'React Hook pattern'),
            (r'useEffect\(', 'React Effect Hook'),
            (r'useContext\(', 'React Context pattern'),
            (r'createContext\(', 'React Context Provider pattern'),
        ]
        
        for pattern, description in react_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                patterns.append(PatternMatch(
                    pattern_type=PatternType.OBSERVER,
                    severity=PatternSeverity.INFO,
                    confidence=0.9,
                    location={'line': line_num, 'column': match.start()},
                    description=description,
                    suggestion=None,
                    code_snippet=match.group(),
                    metrics={'pattern': pattern}
                ))
        
        return patterns
    
    def _is_class_file(self, content: str, language: Optional[str]) -> bool:
        """Check if content contains a class definition"""
        if language == 'python':
            return bool(re.search(r'^class\s+\w+', content, re.MULTILINE))
        elif language in ['java', 'csharp']:
            return bool(re.search(r'(?:public|private|protected)?\s*class\s+\w+', content))
        elif language == 'javascript':
            return bool(re.search(r'class\s+\w+', content))
        
        return False
    
    def _analyze_class_metrics(self, content: str, 
                              language: Optional[str]) -> Dict[str, int]:
        """Analyze class metrics"""
        metrics = {
            'lines': content.count('\n') + 1,
            'methods': 0,
            'attributes': 0,
            'imports': 0
        }
        
        if language == 'python':
            metrics['methods'] = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
            metrics['attributes'] = len(re.findall(r'^\s*self\.\w+\s*=', content, re.MULTILINE))
            metrics['imports'] = len(re.findall(r'^(?:import|from)', content, re.MULTILINE))
        elif language == 'java':
            metrics['methods'] = len(re.findall(r'(?:public|private|protected).*\s+\w+\s*\(', content))
            metrics['attributes'] = len(re.findall(r'(?:public|private|protected).*\s+\w+\s*;', content))
            metrics['imports'] = len(re.findall(r'^import\s+', content, re.MULTILINE))
        
        return metrics
    
    def _extract_methods(self, content: str, 
                        language: Optional[str]) -> List[Dict[str, Any]]:
        """Extract method information"""
        methods = []
        
        if language == 'python':
            pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\([^)]*\):', re.MULTILINE)
            lines = content.split('\n')
            
            for match in pattern.finditer(content):
                indent = len(match.group(1))
                name = match.group(2)
                start_line = content[:match.start()].count('\n') + 1
                
                # Find method end
                end_line = start_line
                for i in range(start_line, len(lines)):
                    if lines[i].strip() and not lines[i].startswith(' ' * (indent + 1)):
                        break
                    end_line = i
                
                methods.append({
                    'name': name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'lines': end_line - start_line + 1
                })
        
        return methods
    
    def _find_duplicates(self, content: str) -> List[Dict[str, Any]]:
        """Find duplicate code blocks"""
        duplicates = []
        lines = content.split('\n')
        
        # Simple duplicate detection (can be enhanced with more sophisticated algorithms)
        block_size = 5  # Minimum block size to consider
        seen_blocks = {}
        
        for i in range(len(lines) - block_size):
            block = '\n'.join(lines[i:i + block_size])
            block_hash = hashlib.md5(block.encode()).hexdigest()
            
            if block_hash in seen_blocks:
                duplicates.append({
                    'line': i + 1,
                    'similarity': 1.0,
                    'snippet': block[:200],
                    'original_line': seen_blocks[block_hash]
                })
            else:
                seen_blocks[block_hash] = i + 1
        
        return duplicates
    
    def _detect_long_parameter_lists(self, content: str,
                                    language: Optional[str]) -> List[PatternMatch]:
        """Detect methods with long parameter lists"""
        patterns = []
        
        if language == 'python':
            method_pattern = re.compile(r'def\s+\w+\s*\(([^)]+)\):', re.MULTILINE)
        elif language in ['java', 'csharp']:
            method_pattern = re.compile(r'(?:public|private|protected).*\s+\w+\s*\(([^)]+)\)')
        elif language == 'javascript':
            method_pattern = re.compile(r'function\s+\w+\s*\(([^)]+)\)')
        else:
            return patterns
        
        for match in method_pattern.finditer(content):
            params = match.group(1)
            param_count = len([p.strip() for p in params.split(',') if p.strip()])
            
            if param_count > 5:
                line_num = content[:match.start()].count('\n') + 1
                patterns.append(PatternMatch(
                    pattern_type=PatternType.LONG_PARAMETER_LIST,
                    severity=PatternSeverity.MEDIUM,
                    confidence=1.0,
                    location={'line': line_num, 'column': match.start()},
                    description=f'Method has {param_count} parameters',
                    suggestion='Consider using parameter object or builder pattern',
                    code_snippet=match.group()[:100],
                    metrics={'parameter_count': param_count}
                ))
        
        return patterns
    
    def _calculate_complexity(self, content: str,
                            language: Optional[str]) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        metrics = {
            'cyclomatic_complexity': 0.0,
            'cognitive_complexity': 0.0,
            'nesting_depth': 0.0,
            'lines_of_code': content.count('\n') + 1,
            'comment_ratio': 0.0,
        }
        
        # Cyclomatic complexity (simplified)
        control_flow = ['if', 'elif', 'else', 'for', 'while', 'case', 'catch', 'except']
        for keyword in control_flow:
            metrics['cyclomatic_complexity'] += content.count(f' {keyword} ') + content.count(f' {keyword}(')
        
        # Nesting depth
        max_indent = 0
        for line in content.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        metrics['nesting_depth'] = max_indent / 4  # Assuming 4-space indents
        
        # Comment ratio
        comment_lines = 0
        if language == 'python':
            comment_lines = len(re.findall(r'^\s*#', content, re.MULTILINE))
        elif language in ['java', 'javascript', 'csharp']:
            comment_lines = len(re.findall(r'^\s*//', content, re.MULTILINE))
        
        total_lines = metrics['lines_of_code']
        metrics['comment_ratio'] = comment_lines / total_lines if total_lines > 0 else 0
        
        return metrics
    
    def _calculate_quality_score(self, patterns: List[PatternMatch],
                                complexity_metrics: Dict[str, float]) -> float:
        """Calculate overall code quality score"""
        score = 100.0
        
        # Deduct points for patterns based on severity
        for pattern in patterns:
            if pattern.severity == PatternSeverity.CRITICAL:
                score -= 20
            elif pattern.severity == PatternSeverity.HIGH:
                score -= 10
            elif pattern.severity == PatternSeverity.MEDIUM:
                score -= 5
            elif pattern.severity == PatternSeverity.LOW:
                score -= 2
        
        # Deduct for high complexity
        if complexity_metrics['cyclomatic_complexity'] > 20:
            score -= 10
        elif complexity_metrics['cyclomatic_complexity'] > 10:
            score -= 5
        
        if complexity_metrics['nesting_depth'] > 5:
            score -= 5
        
        # Bonus for good comment ratio
        if 0.1 <= complexity_metrics['comment_ratio'] <= 0.3:
            score += 5
        
        return max(0.0, min(100.0, score))
    
    def _generate_refactoring_suggestions(self, patterns: List[PatternMatch],
                                         complexity_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate refactoring suggestions based on patterns"""
        suggestions = []
        
        # Group patterns by type
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)
        
        # Generate suggestions based on pattern groups
        if PatternType.GOD_CLASS in pattern_groups:
            suggestions.append({
                'type': 'class_decomposition',
                'priority': 'high',
                'description': 'Break down large class into smaller, focused classes',
                'techniques': ['Single Responsibility Principle', 'Extract Class', 'Move Method']
            })
        
        if PatternType.LONG_METHOD in pattern_groups:
            suggestions.append({
                'type': 'method_extraction',
                'priority': 'medium',
                'description': 'Extract long methods into smaller, focused methods',
                'techniques': ['Extract Method', 'Replace Temp with Query']
            })
        
        if PatternType.DUPLICATE_CODE in pattern_groups:
            suggestions.append({
                'type': 'code_consolidation',
                'priority': 'medium',
                'description': 'Eliminate duplicate code',
                'techniques': ['Extract Method', 'Pull Up Method', 'Form Template Method']
            })
        
        if PatternType.COMPLEX_CONDITIONAL in pattern_groups:
            suggestions.append({
                'type': 'conditional_simplification',
                'priority': 'low',
                'description': 'Simplify complex conditional logic',
                'techniques': ['Decompose Conditional', 'Consolidate Conditional', 'Extract Method']
            })
        
        # Complexity-based suggestions
        if complexity_metrics['cyclomatic_complexity'] > 15:
            suggestions.append({
                'type': 'reduce_complexity',
                'priority': 'high',
                'description': 'Reduce cyclomatic complexity',
                'techniques': ['Extract Method', 'Replace Conditional with Polymorphism']
            })
        
        return suggestions
    
    def _generate_statistics(self, patterns: List[PatternMatch]) -> Dict[str, Any]:
        """Generate statistics about patterns found"""
        stats = {
            'total_patterns': len(patterns),
            'by_severity': Counter(p.severity.name for p in patterns),
            'by_type': Counter(p.pattern_type.name for p in patterns),
            'critical_count': sum(1 for p in patterns if p.severity == PatternSeverity.CRITICAL),
            'high_priority_count': sum(1 for p in patterns if p.severity in [PatternSeverity.CRITICAL, PatternSeverity.HIGH]),
            'average_confidence': sum(p.confidence for p in patterns) / len(patterns) if patterns else 0
        }
        
        return stats
    
    def _create_empty_analysis(self, file_path: Optional[Path], 
                              error: str) -> PatternAnalysis:
        """Create empty analysis for error cases"""
        return PatternAnalysis(
            file_path=file_path,
            patterns_found=[],
            design_patterns=[],
            anti_patterns=[],
            code_smells=[],
            security_issues=[],
            performance_issues=[],
            complexity_metrics={},
            quality_score=0.0,
            refactoring_suggestions=[],
            statistics={'error': error},
            metadata={'error': error}
        )

def analyze_patterns(content: str, file_path: Optional[Path] = None,
                    language: Optional[str] = None) -> PatternAnalysis:
    """
    Convenience function to analyze patterns
    
    Args:
        content: File content
        file_path: Optional file path
        language: Optional programming language
        
    Returns:
        PatternAnalysis object
    """
    detector = PatternDetector()
    return detector.analyze(content, file_path, language)