"""
File classifier for categorizing files based on purpose, domain, and importance
Provides multi-dimensional classification for intelligent processing prioritization
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json
import yaml
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FileCategory(Enum):
    """High-level file categories"""
    SOURCE_CODE = "source_code"
    TEST_CODE = "test_code"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    BUILD_SCRIPT = "build_script"
    DEPLOYMENT = "deployment"
    DATABASE = "database"
    INTERFACE = "interface"
    RESOURCE = "resource"
    GENERATED = "generated"
    VENDOR = "vendor"
    DATA = "data"
    LOG = "log"
    BACKUP = "backup"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"

class FileDomain(Enum):
    """Domain/layer classification"""
    PRESENTATION = "presentation"          # UI, views, templates
    APPLICATION = "application"            # Business logic, services
    DOMAIN = "domain"                      # Core domain models
    INFRASTRUCTURE = "infrastructure"      # Database, external services
    CONFIGURATION = "configuration"        # Config files
    TESTING = "testing"                   # Test files
    DOCUMENTATION = "documentation"        # Docs
    DEPLOYMENT = "deployment"              # CI/CD, containers
    UTILITY = "utility"                   # Helpers, utils
    INTEGRATION = "integration"           # API, interfaces
    SECURITY = "security"                 # Auth, crypto
    MONITORING = "monitoring"             # Logs, metrics
    ASSETS = "assets"                    # Static resources
    UNKNOWN = "unknown"

class FileImportance(Enum):
    """File importance levels"""
    CRITICAL = 5      # Core business logic, main entry points
    HIGH = 4          # Important features, key services
    MEDIUM = 3        # Standard implementation files
    LOW = 2           # Utilities, helpers
    MINIMAL = 1       # Generated, temporary, logs

class FilePurpose(Enum):
    """Specific file purposes"""
    # Entry points
    MAIN_ENTRY = "main_entry"
    API_ENDPOINT = "api_endpoint"
    CLI_INTERFACE = "cli_interface"
    
    # Core components
    MODEL = "model"
    SERVICE = "service"
    CONTROLLER = "controller"
    REPOSITORY = "repository"
    
    # Configuration
    APP_CONFIG = "app_config"
    ENV_CONFIG = "env_config"
    BUILD_CONFIG = "build_config"
    DEPLOY_CONFIG = "deploy_config"
    
    # Testing
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    E2E_TEST = "e2e_test"
    TEST_FIXTURE = "test_fixture"
    
    # Documentation
    README = "readme"
    API_DOC = "api_doc"
    USER_GUIDE = "user_guide"
    CHANGELOG = "changelog"
    
    # Database
    SCHEMA = "schema"
    MIGRATION = "migration"
    SEED_DATA = "seed_data"
    QUERY = "query"
    
    # Frontend
    COMPONENT = "component"
    VIEW = "view"
    STYLE = "style"
    ASSET = "asset"
    
    # Infrastructure
    CONTAINER = "container"
    ORCHESTRATION = "orchestration"
    CI_CD = "ci_cd"
    MONITORING = "monitoring"
    
    # Other
    UTILITY = "utility"
    HELPER = "helper"
    CONSTANT = "constant"
    TYPE_DEFINITION = "type_definition"
    INTERFACE = "interface"
    EXCEPTION = "exception"
    MIDDLEWARE = "middleware"
    PLUGIN = "plugin"
    LIBRARY = "library"
    VENDOR = "vendor"
    GENERATED = "generated"
    UNKNOWN = "unknown"

class TechnologyStack(Enum):
    """Technology stack classification"""
    # Languages
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    
    # Frameworks
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    REACT = "react"
    ANGULAR = "angular"
    VUE = "vue"
    SPRING = "spring"
    DOTNET = "dotnet"
    EXPRESS = "express"
    RAILS = "rails"
    LARAVEL = "laravel"
    
    # Databases
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    
    # Infrastructure
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    
    # Tools
    WEBPACK = "webpack"
    GRADLE = "gradle"
    MAVEN = "maven"
    NPM = "npm"
    YARN = "yarn"
    POETRY = "poetry"
    
    UNKNOWN = "unknown"

@dataclass
class FileClassification:
    """Complete file classification result"""
    path: Path
    category: FileCategory
    domain: FileDomain
    purpose: FilePurpose
    importance: FileImportance
    technology_stack: Set[TechnologyStack]
    is_test: bool
    is_generated: bool
    is_vendor: bool
    is_config: bool
    is_documentation: bool
    is_entry_point: bool
    size_bytes: int
    line_count: int
    complexity_score: float
    dependencies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

class FileClassifier:
    """Classifies files based on multiple dimensions"""
    
    # File patterns by category
    CATEGORY_PATTERNS = {
        FileCategory.SOURCE_CODE: {
            'extensions': {'.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.rb', '.php'},
            'patterns': [r'\.(?:py|js|java|cpp|c|cs|go|rs|rb|php)$'],
            'exclude_patterns': [r'test', r'spec', r'_test\.', r'\.test\.'],
        },
        FileCategory.TEST_CODE: {
            'extensions': {'.py', '.js', '.java', '.cpp', '.cs', '.go'},
            'patterns': [r'test_.*\.py$', r'.*_test\.py$', r'.*\.test\.js$', r'.*\.spec\.js$',
                        r'.*Test\.java$', r'.*_test\.go$'],
            'path_patterns': [r'/test/', r'/tests/', r'/spec/', r'/__tests__/'],
        },
        FileCategory.CONFIGURATION: {
            'extensions': {'.json', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.properties', '.env'},
            'patterns': [r'\.(?:json|ya?ml|ini|cfg|conf|properties|env)$'],
            'filenames': {'config.json', 'settings.py', 'application.properties', '.env'},
        },
        FileCategory.DOCUMENTATION: {
            'extensions': {'.md', '.rst', '.txt', '.adoc'},
            'patterns': [r'\.(?:md|rst|txt|adoc)$'],
            'filenames': {'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE'},
        },
        FileCategory.BUILD_SCRIPT: {
            'extensions': {'.gradle', '.xml', '.yml'},
            'filenames': {'pom.xml', 'build.gradle', 'package.json', 'Makefile', 'setup.py'},
            'patterns': [r'build\.', r'setup\.'],
        },
        FileCategory.DATABASE: {
            'extensions': {'.sql', '.ddl', '.dml'},
            'patterns': [r'\.(?:sql|ddl|dml)$', r'migration', r'schema'],
            'path_patterns': [r'/migrations/', r'/db/', r'/database/', r'/sql/'],
        },
        FileCategory.DEPLOYMENT: {
            'extensions': {'.dockerfile', '.yml', '.yaml'},
            'filenames': {'Dockerfile', 'docker-compose.yml', 'k8s.yaml', '.gitlab-ci.yml'},
            'patterns': [r'Dockerfile', r'\.dockerfile$', r'deployment\.ya?ml$'],
        },
    }
    
    # Domain classification patterns
    DOMAIN_PATTERNS = {
        FileDomain.PRESENTATION: {
            'keywords': ['view', 'template', 'ui', 'frontend', 'component', 'page', 'screen'],
            'paths': ['/views/', '/templates/', '/frontend/', '/ui/', '/components/', '/pages/'],
            'extensions': {'.jsx', '.tsx', '.vue', '.html', '.css', '.scss'},
        },
        FileDomain.APPLICATION: {
            'keywords': ['service', 'controller', 'handler', 'manager', 'processor', 'business'],
            'paths': ['/services/', '/controllers/', '/handlers/', '/business/', '/application/'],
        },
        FileDomain.DOMAIN: {
            'keywords': ['model', 'entity', 'domain', 'core', 'aggregate'],
            'paths': ['/models/', '/entities/', '/domain/', '/core/'],
        },
        FileDomain.INFRASTRUCTURE: {
            'keywords': ['repository', 'dao', 'adapter', 'gateway', 'client'],
            'paths': ['/infrastructure/', '/repositories/', '/adapters/', '/persistence/'],
        },
        FileDomain.TESTING: {
            'keywords': ['test', 'spec', 'mock', 'fixture', 'stub'],
            'paths': ['/test/', '/tests/', '/spec/', '/__tests__/'],
        },
        FileDomain.SECURITY: {
            'keywords': ['auth', 'security', 'crypto', 'permission', 'role', 'jwt', 'oauth'],
            'paths': ['/auth/', '/security/', '/permissions/'],
        },
        FileDomain.INTEGRATION: {
            'keywords': ['api', 'rest', 'graphql', 'grpc', 'webhook', 'integration'],
            'paths': ['/api/', '/integration/', '/webhooks/'],
        },
    }
    
    # Purpose detection patterns
    PURPOSE_PATTERNS = {
        FilePurpose.MAIN_ENTRY: {
            'filenames': ['main.py', 'app.py', 'index.js', 'server.js', 'Main.java', 'Program.cs'],
            'patterns': [r'__main__', r'if __name__', r'main\s*\(', r'app\.listen'],
        },
        FilePurpose.API_ENDPOINT: {
            'patterns': [r'@app\.route', r'@router\.', r'@RequestMapping', r'@GetMapping', 
                        r'@PostMapping', r'router\.get', r'router\.post'],
            'keywords': ['endpoint', 'route', 'api', 'rest'],
        },
        FilePurpose.MODEL: {
            'patterns': [r'class.*Model', r'@Entity', r'@Table', r'Schema\s*=', r'model\.Model'],
            'keywords': ['model', 'entity', 'schema'],
        },
        FilePurpose.SERVICE: {
            'patterns': [r'class.*Service', r'@Service', r'@Injectable'],
            'keywords': ['service', 'business', 'logic'],
        },
        FilePurpose.CONTROLLER: {
            'patterns': [r'class.*Controller', r'@Controller', r'@RestController'],
            'keywords': ['controller', 'handler'],
        },
        FilePurpose.REPOSITORY: {
            'patterns': [r'class.*Repository', r'@Repository', r'class.*DAO'],
            'keywords': ['repository', 'dao', 'persistence'],
        },
        FilePurpose.UNIT_TEST: {
            'patterns': [r'test_', r'_test\.', r'\.test\.', r'\.spec\.', r'@Test', r'describe\('],
            'keywords': ['test', 'spec', 'should', 'expect', 'assert'],
        },
        FilePurpose.MIGRATION: {
            'patterns': [r'migration', r'migrate', r'upgrade\(\)', r'downgrade\(\)'],
            'keywords': ['migration', 'upgrade', 'downgrade'],
        },
        FilePurpose.COMPONENT: {
            'patterns': [r'Component\(', r'extends Component', r'React\.Component', r'@Component'],
            'keywords': ['component', 'widget'],
        },
    }
    
    # Technology stack detection
    TECH_PATTERNS = {
        TechnologyStack.PYTHON: {
            'extensions': {'.py', '.pyw'},
            'patterns': [r'import\s+\w+', r'from\s+\w+\s+import'],
            'keywords': ['python', 'pip', 'poetry'],
        },
        TechnologyStack.DJANGO: {
            'patterns': [r'from\s+django', r'INSTALLED_APPS', r'models\.Model'],
            'filenames': ['manage.py', 'settings.py', 'urls.py'],
        },
        TechnologyStack.REACT: {
            'patterns': [r'import\s+React', r'from\s+[\'"]react[\'"]', r'useState', r'useEffect'],
            'extensions': {'.jsx', '.tsx'},
        },
        TechnologyStack.SPRING: {
            'patterns': [r'@SpringBootApplication', r'@Autowired', r'import\s+org\.springframework'],
            'filenames': ['application.properties', 'application.yml'],
        },
        TechnologyStack.DOCKER: {
            'filenames': ['Dockerfile', 'docker-compose.yml', '.dockerignore'],
            'patterns': [r'FROM\s+\w+', r'RUN\s+', r'COPY\s+', r'EXPOSE\s+'],
        },
        TechnologyStack.KUBERNETES: {
            'patterns': [r'apiVersion:', r'kind:\s*(?:Pod|Service|Deployment|ConfigMap)'],
            'keywords': ['kubernetes', 'k8s', 'kubectl'],
        },
        TechnologyStack.TERRAFORM: {
            'extensions': {'.tf', '.tfvars'},
            'patterns': [r'resource\s+"', r'provider\s+"', r'module\s+"'],
        },
    }
    
    # Importance scoring factors
    IMPORTANCE_FACTORS = {
        'main_entry': 5.0,
        'api_endpoint': 4.0,
        'model': 4.0,
        'service': 3.5,
        'controller': 3.5,
        'repository': 3.0,
        'configuration': 3.0,
        'test': 2.0,
        'documentation': 2.0,
        'utility': 1.5,
        'generated': 0.5,
        'vendor': 0.5,
    }
    
    def __init__(self):
        """Initialize file classifier"""
        self.stats = defaultdict(int)
        
    def classify(self, file_path: Path, content: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> FileClassification:
        """
        Classify a file based on path, content, and metadata
        
        Args:
            file_path: Path to file
            content: Optional file content
            metadata: Optional additional metadata
            
        Returns:
            FileClassification object
        """
        try:
            # Get file info
            file_info = self._get_file_info(file_path, content)
            
            # Determine category
            category = self._classify_category(file_path, content)
            
            # Determine domain
            domain = self._classify_domain(file_path, content)
            
            # Determine purpose
            purpose = self._classify_purpose(file_path, content)
            
            # Detect technology stack
            tech_stack = self._detect_technology_stack(file_path, content)
            
            # Calculate importance
            importance = self._calculate_importance(
                category, domain, purpose, file_path, content
            )
            
            # Detect special flags
            is_test = self._is_test_file(file_path, content)
            is_generated = self._is_generated_file(file_path, content)
            is_vendor = self._is_vendor_file(file_path)
            is_config = category == FileCategory.CONFIGURATION
            is_documentation = category == FileCategory.DOCUMENTATION
            is_entry_point = purpose == FilePurpose.MAIN_ENTRY
            
            # Calculate complexity
            complexity = self._calculate_complexity(content, category, purpose)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(content, file_path)
            
            # Generate warnings
            warnings = self._generate_warnings(
                file_path, category, domain, is_vendor, is_generated
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                category, domain, purpose, tech_stack
            )
            
            # Update statistics
            self._update_statistics(category, domain, purpose)
            
            return FileClassification(
                path=file_path,
                category=category,
                domain=domain,
                purpose=purpose,
                importance=importance,
                technology_stack=tech_stack,
                is_test=is_test,
                is_generated=is_generated,
                is_vendor=is_vendor,
                is_config=is_config,
                is_documentation=is_documentation,
                is_entry_point=is_entry_point,
                size_bytes=file_info['size'],
                line_count=file_info['lines'],
                complexity_score=complexity,
                dependencies=dependencies,
                metadata=metadata or {},
                confidence=confidence,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error classifying file {file_path}: {e}")
            return self._create_unknown_classification(file_path, str(e))
    
    def _get_file_info(self, file_path: Path, content: Optional[str]) -> Dict[str, Any]:
        """Get basic file information"""
        info = {
            'size': 0,
            'lines': 0,
            'extension': file_path.suffix.lower(),
            'name': file_path.name,
            'parent': file_path.parent.name if file_path.parent else '',
        }
        
        if file_path.exists():
            info['size'] = file_path.stat().st_size
        
        if content:
            info['lines'] = content.count('\n') + 1
            if not info['size']:
                info['size'] = len(content)
        elif file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    info['lines'] = sum(1 for _ in f)
            except:
                pass
        
        return info
    
    def _classify_category(self, file_path: Path, content: Optional[str]) -> FileCategory:
        """Classify file category"""
        path_str = str(file_path).replace('\\', '/').lower()
        extension = file_path.suffix.lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            # Check extensions
            if 'extensions' in patterns and extension in patterns['extensions']:
                # Check exclusions for source code
                if category == FileCategory.SOURCE_CODE:
                    if any(re.search(p, path_str) for p in patterns.get('exclude_patterns', [])):
                        continue
                return category
            
            # Check patterns
            if 'patterns' in patterns:
                for pattern in patterns['patterns']:
                    if re.search(pattern, path_str):
                        # Check exclusions
                        if category == FileCategory.SOURCE_CODE:
                            if any(re.search(p, path_str) for p in patterns.get('exclude_patterns', [])):
                                continue
                        return category
            
            # Check filenames
            if 'filenames' in patterns:
                if file_path.name in patterns['filenames']:
                    return category
            
            # Check path patterns
            if 'path_patterns' in patterns:
                for pattern in patterns['path_patterns']:
                    if re.search(pattern, path_str):
                        return category
        
        # Content-based classification
        if content:
            # Check for test indicators
            if any(indicator in content for indicator in ['test_', '@Test', 'describe(', 'it(']):
                return FileCategory.TEST_CODE
            
            # Check for SQL
            if any(keyword in content.upper() for keyword in ['SELECT', 'INSERT', 'CREATE TABLE']):
                return FileCategory.DATABASE
        
        # Check for vendor/dependencies
        if any(vendor in path_str for vendor in ['/vendor/', '/node_modules/', '/lib/', '/libs/']):
            return FileCategory.VENDOR
        
        # Check for logs
        if extension in {'.log', '.out', '.err'}:
            return FileCategory.LOG
        
        # Check for backup
        if extension in {'.bak', '.backup', '.old'} or '~' in file_path.name:
            return FileCategory.BACKUP
        
        # Check for temporary
        if extension in {'.tmp', '.temp', '.swp'} or file_path.name.startswith('.'):
            return FileCategory.TEMPORARY
        
        return FileCategory.UNKNOWN
    
    def _classify_domain(self, file_path: Path, content: Optional[str]) -> FileDomain:
        """Classify file domain/layer"""
        path_str = str(file_path).replace('\\', '/').lower()
        filename = file_path.name.lower()
        
        # Path-based classification
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            # Check paths
            if 'paths' in patterns:
                for path_pattern in patterns['paths']:
                    if path_pattern in path_str:
                        return domain
            
            # Check keywords in filename
            if 'keywords' in patterns:
                for keyword in patterns['keywords']:
                    if keyword in filename:
                        return domain
            
            # Check extensions
            if 'extensions' in patterns:
                if file_path.suffix.lower() in patterns['extensions']:
                    return domain
        
        # Content-based classification
        if content:
            # Count domain indicators
            domain_scores = defaultdict(int)
            
            for domain, patterns in self.DOMAIN_PATTERNS.items():
                if 'keywords' in patterns:
                    for keyword in patterns['keywords']:
                        if keyword in content.lower():
                            domain_scores[domain] += 1
            
            if domain_scores:
                return max(domain_scores, key=domain_scores.get)
        
        return FileDomain.UNKNOWN
    
    def _classify_purpose(self, file_path: Path, content: Optional[str]) -> FilePurpose:
        """Classify specific file purpose"""
        filename = file_path.name
        
        # Filename-based classification
        for purpose, patterns in self.PURPOSE_PATTERNS.items():
            if 'filenames' in patterns:
                if filename in patterns['filenames']:
                    return purpose
        
        # Content-based classification
        if content:
            for purpose, patterns in self.PURPOSE_PATTERNS.items():
                # Check patterns
                if 'patterns' in patterns:
                    for pattern in patterns['patterns']:
                        if re.search(pattern, content):
                            return purpose
                
                # Check keywords
                if 'keywords' in patterns:
                    content_lower = content.lower()
                    matches = sum(1 for kw in patterns['keywords'] if kw in content_lower)
                    if matches >= 2:  # At least 2 keyword matches
                        return purpose
        
        # Fallback based on common patterns
        name_lower = filename.lower()
        
        if 'test' in name_lower or 'spec' in name_lower:
            return FilePurpose.UNIT_TEST
        elif 'model' in name_lower:
            return FilePurpose.MODEL
        elif 'service' in name_lower:
            return FilePurpose.SERVICE
        elif 'controller' in name_lower:
            return FilePurpose.CONTROLLER
        elif 'repository' in name_lower or 'dao' in name_lower:
            return FilePurpose.REPOSITORY
        elif 'util' in name_lower or 'helper' in name_lower:
            return FilePurpose.UTILITY
        elif 'config' in name_lower or 'settings' in name_lower:
            return FilePurpose.APP_CONFIG
        elif 'migration' in name_lower:
            return FilePurpose.MIGRATION
        elif 'component' in name_lower:
            return FilePurpose.COMPONENT
        elif 'interface' in name_lower:
            return FilePurpose.INTERFACE
        elif 'exception' in name_lower or 'error' in name_lower:
            return FilePurpose.EXCEPTION
        elif 'constant' in name_lower or 'enum' in name_lower:
            return FilePurpose.CONSTANT
        elif 'middleware' in name_lower:
            return FilePurpose.MIDDLEWARE
        
        return FilePurpose.UNKNOWN
    
    def _detect_technology_stack(self, file_path: Path, 
                                 content: Optional[str]) -> Set[TechnologyStack]:
        """Detect technology stack used"""
        tech_stack = set()
        extension = file_path.suffix.lower()
        
        for tech, patterns in self.TECH_PATTERNS.items():
            # Check extensions
            if 'extensions' in patterns and extension in patterns['extensions']:
                tech_stack.add(tech)
            
            # Check filenames
            if 'filenames' in patterns and file_path.name in patterns['filenames']:
                tech_stack.add(tech)
            
            # Check content patterns
            if content and 'patterns' in patterns:
                for pattern in patterns['patterns']:
                    if re.search(pattern, content):
                        tech_stack.add(tech)
                        break
            
            # Check keywords
            if content and 'keywords' in patterns:
                content_lower = content.lower()
                if any(keyword in content_lower for keyword in patterns['keywords']):
                    tech_stack.add(tech)
        
        # Add base language if framework detected
        if TechnologyStack.DJANGO in tech_stack or TechnologyStack.FLASK in tech_stack:
            tech_stack.add(TechnologyStack.PYTHON)
        elif TechnologyStack.REACT in tech_stack or TechnologyStack.ANGULAR in tech_stack:
            tech_stack.add(TechnologyStack.JAVASCRIPT)
        elif TechnologyStack.SPRING in tech_stack:
            tech_stack.add(TechnologyStack.JAVA)
        
        return tech_stack if tech_stack else {TechnologyStack.UNKNOWN}
    
    def _calculate_importance(self, category: FileCategory, domain: FileDomain,
                             purpose: FilePurpose, file_path: Path,
                             content: Optional[str]) -> FileImportance:
        """Calculate file importance"""
        score = 3.0  # Default medium
        
        # Purpose-based scoring
        purpose_scores = {
            FilePurpose.MAIN_ENTRY: 5.0,
            FilePurpose.API_ENDPOINT: 4.5,
            FilePurpose.MODEL: 4.0,
            FilePurpose.SERVICE: 4.0,
            FilePurpose.CONTROLLER: 3.5,
            FilePurpose.REPOSITORY: 3.5,
            FilePurpose.APP_CONFIG: 3.5,
            FilePurpose.MIGRATION: 3.0,
            FilePurpose.UNIT_TEST: 2.5,
            FilePurpose.COMPONENT: 3.0,
            FilePurpose.UTILITY: 2.0,
            FilePurpose.GENERATED: 1.0,
            FilePurpose.VENDOR: 1.0,
        }
        
        if purpose in purpose_scores:
            score = purpose_scores[purpose]
        
        # Category adjustments
        if category == FileCategory.TEST_CODE:
            score = min(score, 2.5)
        elif category == FileCategory.VENDOR:
            score = min(score, 1.5)
        elif category == FileCategory.GENERATED:
            score = min(score, 1.5)
        elif category == FileCategory.DOCUMENTATION:
            score = min(score, 2.5)
        elif category == FileCategory.TEMPORARY:
            score = min(score, 1.0)
        
        # Domain adjustments
        if domain == FileDomain.DOMAIN:
            score += 0.5  # Core domain is more important
        elif domain == FileDomain.INFRASTRUCTURE:
            score -= 0.5  # Infrastructure is less important
        
        # Special files
        special_files = {
            'main.py': 5.0,
            'app.py': 5.0,
            'index.js': 5.0,
            'Main.java': 5.0,
            'settings.py': 4.0,
            'config.py': 4.0,
            'package.json': 3.5,
            'requirements.txt': 3.0,
            'README.md': 2.5,
        }
        
        if file_path.name in special_files:
            score = max(score, special_files[file_path.name])
        
        # Content-based adjustments
        if content:
            # Large files might be more important
            if len(content) > 10000:
                score += 0.5
            
            # Files with many imports might be central
            import_count = content.count('import ') + content.count('from ')
            if import_count > 20:
                score += 0.5
        
        # Convert score to enum
        if score >= 4.5:
            return FileImportance.CRITICAL
        elif score >= 3.5:
            return FileImportance.HIGH
        elif score >= 2.5:
            return FileImportance.MEDIUM
        elif score >= 1.5:
            return FileImportance.LOW
        else:
            return FileImportance.MINIMAL
    
    def _is_test_file(self, file_path: Path, content: Optional[str]) -> bool:
        """Check if file is a test file"""
        path_str = str(file_path).replace('\\', '/').lower()
        filename = file_path.name.lower()
        
        # Path indicators
        test_paths = ['/test/', '/tests/', '/spec/', '/__tests__/', '/test_']
        if any(test_path in path_str for test_path in test_paths):
            return True
        
        # Filename indicators
        test_patterns = [
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'.*\.test\.\w+$',
            r'.*\.spec\.\w+$',
            r'.*Test\.\w+$',
            r'.*Spec\.\w+$',
        ]
        
        for pattern in test_patterns:
            if re.match(pattern, filename):
                return True
        
        # Content indicators
        if content:
            test_indicators = [
                '@Test', '@test', 'unittest', 'pytest',
                'describe(', 'it(', 'test(', 'expect(',
                'assert', 'should', 'TestCase'
            ]
            
            for indicator in test_indicators:
                if indicator in content:
                    return True
        
        return False
    
    def _is_generated_file(self, file_path: Path, content: Optional[str]) -> bool:
        """Check if file is generated"""
        filename = file_path.name.lower()
        
        # Common generated file patterns
        generated_patterns = [
            r'.*\.generated\.\w+$',
            r'.*\.g\.\w+$',
            r'.*\.pb\.\w+$',  # Protobuf
            r'.*\.min\.\w+$',  # Minified
        ]
        
        for pattern in generated_patterns:
            if re.match(pattern, filename):
                return True
        
        # Check for generation markers in content
        if content:
            generated_markers = [
                'auto-generated',
                'autogenerated',
                'automatically generated',
                'do not edit',
                'do not modify',
                '<autogenerated />',
                '@generated',
            ]
            
            # Check first few lines
            first_lines = '\n'.join(content.split('\n')[:10]).lower()
            for marker in generated_markers:
                if marker in first_lines:
                    return True
        
        return False
    
    def _is_vendor_file(self, file_path: Path) -> bool:
        """Check if file is vendor/third-party"""
        path_str = str(file_path).replace('\\', '/').lower()
        
        vendor_indicators = [
            '/vendor/',
            '/node_modules/',
            '/bower_components/',
            '/lib/',
            '/libs/',
            '/third_party/',
            '/third-party/',
            '/external/',
            '/packages/',
            '/.bundle/',
            '/site-packages/',
        ]
        
        return any(indicator in path_str for indicator in vendor_indicators)
    
    def _calculate_complexity(self, content: Optional[str], 
                            category: FileCategory,
                            purpose: FilePurpose) -> float:
        """Calculate file complexity score"""
        if not content:
            return 0.0
        
        complexity = 0.0
        
        # Size complexity
        lines = content.count('\n') + 1
        complexity += min(lines / 1000, 2.0)  # Max 2 points for 1000+ lines
        
        # Nesting complexity (count indentation levels)
        max_indent = 0
        for line in content.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        complexity += min(max_indent / 20, 1.0)  # Max 1 point for deep nesting
        
        # Control flow complexity
        control_flow_keywords = [
            'if', 'else', 'elif', 'switch', 'case', 'for', 'while',
            'try', 'catch', 'except', 'finally'
        ]
        
        for keyword in control_flow_keywords:
            complexity += content.count(f' {keyword} ') * 0.1
        
        # Method/function count
        function_patterns = [
            r'\bdef\s+\w+',
            r'\bfunction\s+\w+',
            r'\bfunc\s+\w+',
            r'public\s+\w+\s+\w+\s*\(',
            r'private\s+\w+\s+\w+\s*\(',
        ]
        
        for pattern in function_patterns:
            matches = len(re.findall(pattern, content))
            complexity += matches * 0.2
        
        # Class count
        class_patterns = [
            r'\bclass\s+\w+',
            r'public\s+class\s+\w+',
            r'struct\s+\w+',
            r'interface\s+\w+',
        ]
        
        for pattern in class_patterns:
            matches = len(re.findall(pattern, content))
            complexity += matches * 0.3
        
        # Import/dependency complexity
        import_count = content.count('import ') + content.count('from ')
        complexity += min(import_count * 0.05, 1.0)
        
        # Adjust for category
        if category == FileCategory.TEST_CODE:
            complexity *= 0.7  # Tests are usually simpler
        elif category == FileCategory.CONFIGURATION:
            complexity *= 0.5  # Config files are simple
        
        return min(complexity, 10.0)  # Cap at 10
    
    def _extract_dependencies(self, content: Optional[str], 
                            file_path: Path) -> List[str]:
        """Extract file dependencies"""
        dependencies = []
        
        if not content:
            return dependencies
        
        extension = file_path.suffix.lower()
        
        # Python imports
        if extension == '.py':
            # Standard imports
            for match in re.finditer(r'^import\s+([\w.]+)', content, re.MULTILINE):
                dependencies.append(match.group(1))
            
            # From imports
            for match in re.finditer(r'^from\s+([\w.]+)\s+import', content, re.MULTILINE):
                dependencies.append(match.group(1))
        
        # JavaScript/TypeScript imports
        elif extension in {'.js', '.jsx', '.ts', '.tsx'}:
            # ES6 imports
            for match in re.finditer(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', content):
                dependencies.append(match.group(1))
            
            # Require statements
            for match in re.finditer(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]', content):
                dependencies.append(match.group(1))
        
        # Java imports
        elif extension == '.java':
            for match in re.finditer(r'^import\s+([\w.]+);', content, re.MULTILINE):
                dependencies.append(match.group(1))
        
        # Go imports
        elif extension == '.go':
            import_block = re.search(r'import\s*\((.*?)\)', content, re.DOTALL)
            if import_block:
                for line in import_block.group(1).split('\n'):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        # Remove quotes
                        dep = line.strip('"')
                        if dep:
                            dependencies.append(dep)
        
        # C/C++ includes
        elif extension in {'.c', '.cpp', '.cc', '.h', '.hpp'}:
            for match in re.finditer(r'#include\s+[<"]([^>"]+)[>"]', content):
                dependencies.append(match.group(1))
        
        return list(set(dependencies))  # Remove duplicates
    
    def _generate_warnings(self, file_path: Path, category: FileCategory,
                          domain: FileDomain, is_vendor: bool,
                          is_generated: bool) -> List[str]:
        """Generate classification warnings"""
        warnings = []
        
        # Check for potential issues
        if category == FileCategory.UNKNOWN:
            warnings.append("Could not determine file category")
        
        if domain == FileDomain.UNKNOWN:
            warnings.append("Could not determine file domain")
        
        if is_vendor and category == FileCategory.SOURCE_CODE:
            warnings.append("Vendor file classified as source code")
        
        if is_generated and category == FileCategory.SOURCE_CODE:
            warnings.append("Generated file classified as source code")
        
        # Check for suspicious paths
        path_str = str(file_path).lower()
        if 'temp' in path_str or 'tmp' in path_str:
            warnings.append("File appears to be in temporary directory")
        
        if file_path.name.startswith('.'):
            warnings.append("Hidden file detected")
        
        # Check file size
        if file_path.exists():
            size = file_path.stat().st_size
            if size > 10 * 1024 * 1024:  # 10MB
                warnings.append("Very large file (>10MB)")
            elif size == 0:
                warnings.append("Empty file")
        
        return warnings
    
    def _calculate_confidence(self, category: FileCategory, domain: FileDomain,
                            purpose: FilePurpose,
                            tech_stack: Set[TechnologyStack]) -> float:
        """Calculate classification confidence"""
        confidence = 0.0
        
        # Category confidence
        if category != FileCategory.UNKNOWN:
            confidence += 0.3
        
        # Domain confidence
        if domain != FileDomain.UNKNOWN:
            confidence += 0.2
        
        # Purpose confidence
        if purpose != FilePurpose.UNKNOWN:
            confidence += 0.3
        
        # Technology stack confidence
        if tech_stack and TechnologyStack.UNKNOWN not in tech_stack:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _update_statistics(self, category: FileCategory, domain: FileDomain,
                          purpose: FilePurpose):
        """Update classification statistics"""
        self.stats[f'category_{category.value}'] += 1
        self.stats[f'domain_{domain.value}'] += 1
        self.stats[f'purpose_{purpose.value}'] += 1
        self.stats['total_classified'] += 1
    
    def _create_unknown_classification(self, file_path: Path, 
                                      error: str) -> FileClassification:
        """Create classification for unknown/error cases"""
        return FileClassification(
            path=file_path,
            category=FileCategory.UNKNOWN,
            domain=FileDomain.UNKNOWN,
            purpose=FilePurpose.UNKNOWN,
            importance=FileImportance.LOW,
            technology_stack={TechnologyStack.UNKNOWN},
            is_test=False,
            is_generated=False,
            is_vendor=False,
            is_config=False,
            is_documentation=False,
            is_entry_point=False,
            size_bytes=0,
            line_count=0,
            complexity_score=0.0,
            dependencies=[],
            metadata={'error': error},
            confidence=0.0,
            warnings=[f"Classification error: {error}"]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return dict(self.stats)
    
    def classify_directory(self, directory: Path) -> Dict[Path, FileClassification]:
        """
        Classify all files in a directory
        
        Args:
            directory: Directory path
            
        Returns:
            Dictionary of file paths to classifications
        """
        classifications = {}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                try:
                    classification = self.classify(file_path)
                    classifications[file_path] = classification
                except Exception as e:
                    logger.error(f"Error classifying {file_path}: {e}")
        
        return classifications

def classify_file(file_path: Path, content: Optional[str] = None) -> FileClassification:
    """
    Convenience function to classify a file
    
    Args:
        file_path: Path to file
        content: Optional file content
        
    Returns:
        FileClassification object
    """
    classifier = FileClassifier()
    return classifier.classify(file_path, content)