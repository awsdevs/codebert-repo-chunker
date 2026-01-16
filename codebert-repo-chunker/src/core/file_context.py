"""
File context module providing comprehensive file information and metadata
Handles file analysis, project context, and repository understanding
"""

import os
import re
import json
import yaml
import hashlib
import mimetypes
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from src.utils.logger import get_logger
import chardet
import git
import ast
import tokenize
import io

logger = get_logger(__name__)

class FileType(Enum):
    """File type classifications"""
    # Source code
    SOURCE = "source"
    HEADER = "header"
    IMPLEMENTATION = "implementation"
    
    # Tests
    TEST = "test"
    BENCHMARK = "benchmark"
    FIXTURE = "fixture"
    
    # Documentation
    DOCUMENTATION = "documentation"
    README = "readme"
    LICENSE = "license"
    CHANGELOG = "changelog"
    
    # Configuration
    CONFIG = "config"
    MANIFEST = "manifest"
    LOCKFILE = "lockfile"
    ENVIRONMENT = "environment"
    
    # Build/Deploy
    BUILD = "build"
    DEPLOYMENT = "deployment"
    DOCKERFILE = "dockerfile"
    CI_CD = "ci_cd"
    
    # Data
    DATA = "data"
    SCHEMA = "schema"
    MIGRATION = "migration"
    SEED = "seed"
    
    # Assets
    ASSET = "asset"
    RESOURCE = "resource"
    TEMPLATE = "template"
    
    # Generated
    GENERATED = "generated"
    COMPILED = "compiled"
    MINIFIED = "minified"
    
    # Other
    VENDOR = "vendor"
    BINARY = "binary"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"

class ProjectType(Enum):
    """Project type classifications"""
    PYTHON_PACKAGE = "python_package"
    PYTHON_APPLICATION = "python_application"
    JAVASCRIPT_LIBRARY = "javascript_library"
    JAVASCRIPT_APPLICATION = "javascript_application"
    JAVA_PROJECT = "java_project"
    SPRING_BOOT = "spring_boot"
    REACT_APP = "react_app"
    ANGULAR_APP = "angular_app"
    VUE_APP = "vue_app"
    DJANGO_PROJECT = "django_project"
    FLASK_APP = "flask_app"
    FASTAPI_APP = "fastapi_app"
    EXPRESS_APP = "express_app"
    DOTNET_PROJECT = "dotnet_project"
    GO_MODULE = "go_module"
    RUST_CRATE = "rust_crate"
    RUBY_GEM = "ruby_gem"
    RAILS_APP = "rails_app"
    PHP_PROJECT = "php_project"
    LARAVEL_APP = "laravel_app"
    MOBILE_APP = "mobile_app"
    MICROSERVICE = "microservice"
    MONOREPO = "monorepo"
    LIBRARY = "library"
    CLI_TOOL = "cli_tool"
    API_SERVICE = "api_service"
    UNKNOWN = "unknown"

@dataclass
class FileEncoding:
    """File encoding information"""
    encoding: str
    confidence: float
    bom: Optional[str] = None
    line_ending: str = "LF"  # LF, CRLF, CR
    
    @classmethod
    def detect(cls, file_path: Path) -> 'FileEncoding':
        """Detect file encoding"""
        try:
            # Read raw bytes
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
            
            # Check for BOM
            bom = None
            if raw_data.startswith(b'\xff\xfe\x00\x00'):
                return cls('utf-32-le', 1.0, 'UTF-32-LE')
            elif raw_data.startswith(b'\x00\x00\xfe\xff'):
                return cls('utf-32-be', 1.0, 'UTF-32-BE')
            elif raw_data.startswith(b'\xff\xfe'):
                return cls('utf-16-le', 1.0, 'UTF-16-LE')
            elif raw_data.startswith(b'\xfe\xff'):
                return cls('utf-16-be', 1.0, 'UTF-16-BE')
            elif raw_data.startswith(b'\xef\xbb\xbf'):
                bom = 'UTF-8'
                raw_data = raw_data[3:]  # Remove BOM for detection
            
            # Detect encoding
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0.5)
            
            # Detect line ending
            line_ending = 'LF'
            if b'\r\n' in raw_data:
                line_ending = 'CRLF'
            elif b'\r' in raw_data:
                line_ending = 'CR'
            
            return cls(encoding, confidence, bom, line_ending)
            
        except Exception as e:
            logger.warning(f"Failed to detect encoding: {e}")
            return cls('utf-8', 0.5)

@dataclass
class FileStatistics:
    """File statistics and metrics"""
    size_bytes: int = 0
    lines_total: int = 0
    lines_code: int = 0
    lines_comment: int = 0
    lines_blank: int = 0
    characters: int = 0
    words: int = 0
    tokens: int = 0
    functions: int = 0
    classes: int = 0
    imports: int = 0
    complexity: float = 0.0
    
    @classmethod
    def calculate(cls, content: str, language: Optional[str] = None) -> 'FileStatistics':
        """Calculate file statistics"""
        stats = cls()
        
        stats.size_bytes = len(content.encode('utf-8'))
        stats.characters = len(content)
        stats.words = len(content.split())
        
        lines = content.split('\n')
        stats.lines_total = len(lines)
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                stats.lines_blank += 1
            elif language:
                if cls._is_comment(stripped, language):
                    stats.lines_comment += 1
                else:
                    stats.lines_code += 1
            else:
                stats.lines_code += 1
        
        # Language-specific analysis
        if language == 'python':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        stats.functions += 1
                    elif isinstance(node, ast.ClassDef):
                        stats.classes += 1
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        stats.imports += 1
            except:
                pass
        
        return stats
    
    @staticmethod
    def _is_comment(line: str, language: str) -> bool:
        """Check if line is a comment"""
        comment_markers = {
            'python': ['#'],
            'javascript': ['//', '/*', '*/', '*'],
            'java': ['//', '/*', '*/', '*'],
            'cpp': ['//', '/*', '*/', '*'],
            'c': ['//', '/*', '*/', '*'],
            'ruby': ['#'],
            'go': ['//', '/*', '*/', '*'],
            'rust': ['//', '/*', '*/', '*'],
            'php': ['//', '/*', '*/', '#'],
            'sql': ['--', '/*', '*/'],
            'shell': ['#'],
        }
        
        markers = comment_markers.get(language, [])
        return any(line.startswith(marker) for marker in markers)

@dataclass
class GitInfo:
    """Git repository information"""
    is_tracked: bool = False
    branch: Optional[str] = None
    commit: Optional[str] = None
    remote: Optional[str] = None
    is_modified: bool = False
    is_staged: bool = False
    last_commit_date: Optional[datetime] = None
    last_commit_author: Optional[str] = None
    commit_count: int = 0
    contributors: List[str] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'GitInfo':
        """Get git info for a file"""
        info = cls()
        
        try:
            repo = git.Repo(file_path.parent, search_parent_directories=True)
            info.is_tracked = True
            
            # Branch info
            try:
                info.branch = repo.active_branch.name
            except:
                info.branch = repo.head.commit.hexsha[:7]  # Detached HEAD
            
            # Current commit
            info.commit = repo.head.commit.hexsha[:7]
            
            # Remote info
            if repo.remotes:
                info.remote = repo.remotes[0].url
            
            # File status
            changed_files = [item.a_path for item in repo.index.diff(None)]
            staged_files = [item.a_path for item in repo.index.diff('HEAD')]
            
            rel_path = str(file_path.relative_to(repo.working_dir))
            info.is_modified = rel_path in changed_files
            info.is_staged = rel_path in staged_files
            
            # File history
            commits = list(repo.iter_commits(paths=rel_path))
            if commits:
                info.commit_count = len(commits)
                info.last_commit_date = datetime.fromtimestamp(
                    commits[0].committed_date,
                    tz=timezone.utc
                )
                info.last_commit_author = commits[0].author.name
                
                # Get unique contributors
                contributors = set()
                for commit in commits:
                    contributors.add(commit.author.name)
                info.contributors = list(contributors)
            
        except Exception as e:
            logger.debug(f"Not a git repository or git not available: {e}")
        
        return info

@dataclass
class DependencyInfo:
    """File dependency information"""
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    external_deps: List[str] = field(default_factory=list)
    internal_deps: List[str] = field(default_factory=list)
    
    @classmethod
    def extract(cls, content: str, language: Optional[str] = None) -> 'DependencyInfo':
        """Extract dependency information"""
        info = cls()
        
        if not language:
            return info
        
        if language == 'python':
            info._extract_python_deps(content)
        elif language in ['javascript', 'typescript']:
            info._extract_javascript_deps(content)
        elif language == 'java':
            info._extract_java_deps(content)
        elif language == 'go':
            info._extract_go_deps(content)
        
        return info
    
    def _extract_python_deps(self, content: str):
        """Extract Python dependencies"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        self.imports.append(full_name)
            
            # Classify as internal/external
            for imp in self.imports:
                if imp.startswith('.'):
                    self.internal_deps.append(imp)
                elif not imp.split('.')[0] in ['os', 'sys', 'json', 'math', 're']:
                    self.external_deps.append(imp)
        except:
            pass
    
    def _extract_javascript_deps(self, content: str):
        """Extract JavaScript dependencies"""
        # Import statements
        import_pattern = re.compile(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]')
        for match in import_pattern.finditer(content):
            self.imports.append(match.group(1))
        
        # Require statements
        require_pattern = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]')
        for match in require_pattern.finditer(content):
            self.imports.append(match.group(1))
        
        # Export statements
        export_pattern = re.compile(r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)')
        for match in export_pattern.finditer(content):
            self.exports.append(match.group(1))
        
        # Classify dependencies
        for imp in self.imports:
            if imp.startswith('.'):
                self.internal_deps.append(imp)
            else:
                self.external_deps.append(imp)
    
    def _extract_java_deps(self, content: str):
        """Extract Java dependencies"""
        import_pattern = re.compile(r'import\s+([\w.]+);')
        for match in import_pattern.finditer(content):
            self.imports.append(match.group(1))
        
        # Classify by package
        for imp in self.imports:
            if imp.startswith('java.') or imp.startswith('javax.'):
                pass  # Standard library
            else:
                self.external_deps.append(imp)
    
    def _extract_go_deps(self, content: str):
        """Extract Go dependencies"""
        import_pattern = re.compile(r'import\s+(?:\(([^)]+)\)|"([^"]+)")')
        for match in import_pattern.finditer(content):
            if match.group(1):  # Multiple imports
                for line in match.group(1).split('\n'):
                    line = line.strip().strip('"')
                    if line:
                        self.imports.append(line)
            elif match.group(2):  # Single import
                self.imports.append(match.group(2))

@dataclass
class ProjectContext:
    """Project-level context information"""
    root_path: Path
    project_type: ProjectType = ProjectType.UNKNOWN
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    languages: Dict[str, int] = field(default_factory=dict)  # language -> file count
    framework: Optional[str] = None
    frameworks: List[str] = field(default_factory=list)
    build_tool: Optional[str] = None
    package_manager: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    entry_point: Optional[str] = None
    test_framework: Optional[str] = None
    config_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def detect(cls, path: Path) -> 'ProjectContext':
        """Detect project context from path"""
        # Find project root
        root = cls._find_project_root(path)
        context = cls(root_path=root)
        
        # Detect project type and metadata
        context._detect_python_project()
        context._detect_javascript_project()
        context._detect_java_project()
        context._detect_go_project()
        context._detect_rust_project()
        context._detect_ruby_project()
        context._detect_dotnet_project()
        
        # Detect languages used
        context._detect_languages()
        
        # Determine primary project type
        context._determine_project_type()
        
        return context
    
    @staticmethod
    def _find_project_root(path: Path) -> Path:
        """Find project root directory"""
        if path.is_file():
            path = path.parent
        
        # Look for root indicators
        root_indicators = [
            '.git', 'package.json', 'setup.py', 'pyproject.toml',
            'pom.xml', 'build.gradle', 'go.mod', 'Cargo.toml',
            'Gemfile', '*.csproj', '*.sln'
        ]
        
        current = path
        while current != current.parent:
            for indicator in root_indicators:
                if (current / indicator).exists() or list(current.glob(indicator)):
                    return current
            current = current.parent
        
        return path
    
    def _detect_python_project(self):
        """Detect Python project details"""
        # Check for setup.py
        setup_py = self.root_path / 'setup.py'
        if setup_py.exists():
            self.language = 'python'
            self.build_tool = 'setuptools'
            self._parse_setup_py(setup_py)
        
        # Check for pyproject.toml
        pyproject = self.root_path / 'pyproject.toml'
        if pyproject.exists():
            self.language = 'python'
            self._parse_pyproject_toml(pyproject)
        
        # Check for requirements.txt
        requirements = self.root_path / 'requirements.txt'
        if requirements.exists():
            self.language = 'python'
            self.package_manager = 'pip'
            self._parse_requirements_txt(requirements)
        
        # Check for Pipfile
        pipfile = self.root_path / 'Pipfile'
        if pipfile.exists():
            self.language = 'python'
            self.package_manager = 'pipenv'
        
        # Check for Django
        if (self.root_path / 'manage.py').exists():
            self.framework = 'django'
            self.frameworks.append('django')
        
        # Check for Flask
        if 'flask' in self.dependencies:
            self.framework = 'flask'
            self.frameworks.append('flask')
        
        # Check for FastAPI
        if 'fastapi' in self.dependencies:
            self.framework = 'fastapi'
            self.frameworks.append('fastapi')
    
    def _detect_javascript_project(self):
        """Detect JavaScript project details"""
        # Check for package.json
        package_json = self.root_path / 'package.json'
        if package_json.exists():
            if not self.language:
                self.language = 'javascript'
            self.package_manager = 'npm'
            self._parse_package_json(package_json)
        
        # Check for yarn.lock
        if (self.root_path / 'yarn.lock').exists():
            self.package_manager = 'yarn'
        
        # Check for pnpm-lock.yaml
        if (self.root_path / 'pnpm-lock.yaml').exists():
            self.package_manager = 'pnpm'
        
        # Check for React
        if 'react' in self.dependencies:
            self.framework = 'react'
            self.frameworks.append('react')
        
        # Check for Angular
        if '@angular/core' in self.dependencies:
            self.framework = 'angular'
            self.frameworks.append('angular')
        
        # Check for Vue
        if 'vue' in self.dependencies:
            self.framework = 'vue'
            self.frameworks.append('vue')
        
        # Check for Express
        if 'express' in self.dependencies:
            if not self.framework:
                self.framework = 'express'
            self.frameworks.append('express')
    
    def _detect_java_project(self):
        """Detect Java project details"""
        # Check for pom.xml (Maven)
        pom = self.root_path / 'pom.xml'
        if pom.exists():
            self.language = 'java'
            self.build_tool = 'maven'
            self._parse_pom_xml(pom)
        
        # Check for build.gradle (Gradle)
        build_gradle = self.root_path / 'build.gradle'
        if build_gradle.exists():
            self.language = 'java'
            self.build_tool = 'gradle'
            self._parse_build_gradle(build_gradle)
        
        # Check for Spring Boot
        if any('spring-boot' in str(dep) for dep in self.dependencies):
            self.framework = 'spring-boot'
            self.frameworks.append('spring-boot')
    
    def _detect_go_project(self):
        """Detect Go project details"""
        go_mod = self.root_path / 'go.mod'
        if go_mod.exists():
            self.language = 'go'
            self.package_manager = 'go modules'
            self._parse_go_mod(go_mod)
    
    def _detect_rust_project(self):
        """Detect Rust project details"""
        cargo_toml = self.root_path / 'Cargo.toml'
        if cargo_toml.exists():
            self.language = 'rust'
            self.build_tool = 'cargo'
            self._parse_cargo_toml(cargo_toml)
    
    def _detect_ruby_project(self):
        """Detect Ruby project details"""
        gemfile = self.root_path / 'Gemfile'
        if gemfile.exists():
            self.language = 'ruby'
            self.package_manager = 'bundler'
            
            # Check for Rails
            if (self.root_path / 'config' / 'application.rb').exists():
                self.framework = 'rails'
                self.frameworks.append('rails')
    
    def _detect_dotnet_project(self):
        """Detect .NET project details"""
        csproj_files = list(self.root_path.glob('*.csproj'))
        if csproj_files:
            self.language = 'csharp'
            self.build_tool = 'msbuild'
            self.package_manager = 'nuget'
    
    def _detect_languages(self):
        """Detect all languages used in project"""
        language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
        }
        
        for ext, lang in language_extensions.items():
            count = len(list(self.root_path.rglob(f'*{ext}')))
            if count > 0:
                self.languages[lang] = count
    
    def _determine_project_type(self):
        """Determine the primary project type"""
        if self.framework == 'django':
            self.project_type = ProjectType.DJANGO_PROJECT
        elif self.framework == 'flask':
            self.project_type = ProjectType.FLASK_APP
        elif self.framework == 'fastapi':
            self.project_type = ProjectType.FASTAPI_APP
        elif self.framework == 'react':
            self.project_type = ProjectType.REACT_APP
        elif self.framework == 'angular':
            self.project_type = ProjectType.ANGULAR_APP
        elif self.framework == 'vue':
            self.project_type = ProjectType.VUE_APP
        elif self.framework == 'express':
            self.project_type = ProjectType.EXPRESS_APP
        elif self.framework == 'spring-boot':
            self.project_type = ProjectType.SPRING_BOOT
        elif self.framework == 'rails':
            self.project_type = ProjectType.RAILS_APP
        elif self.language == 'python':
            if (self.root_path / '__init__.py').exists():
                self.project_type = ProjectType.PYTHON_PACKAGE
            else:
                self.project_type = ProjectType.PYTHON_APPLICATION
        elif self.language == 'javascript':
            if self.name and 'lib' in str(self.root_path):
                self.project_type = ProjectType.JAVASCRIPT_LIBRARY
            else:
                self.project_type = ProjectType.JAVASCRIPT_APPLICATION
        elif self.language == 'go':
            self.project_type = ProjectType.GO_MODULE
        elif self.language == 'rust':
            self.project_type = ProjectType.RUST_CRATE
    
    def _parse_setup_py(self, setup_py: Path):
        """Parse setup.py file"""
        try:
            content = setup_py.read_text()
            
            # Extract name
            name_match = re.search(r'name\s*=\s*[\'"]([^\'"]+)', content)
            if name_match:
                self.name = name_match.group(1)
            
            # Extract version
            version_match = re.search(r'version\s*=\s*[\'"]([^\'"]+)', content)
            if version_match:
                self.version = version_match.group(1)
            
            # Extract description
            desc_match = re.search(r'description\s*=\s*[\'"]([^\'"]+)', content)
            if desc_match:
                self.description = desc_match.group(1)
        except Exception as e:
            logger.debug(f"Failed to parse setup.py: {e}")
    
    def _parse_pyproject_toml(self, pyproject: Path):
        """Parse pyproject.toml file"""
        try:
            import toml
            data = toml.load(pyproject)
            
            # Poetry section
            if 'tool' in data and 'poetry' in data['tool']:
                poetry = data['tool']['poetry']
                self.name = poetry.get('name')
                self.version = poetry.get('version')
                self.description = poetry.get('description')
                self.package_manager = 'poetry'
                
                if 'dependencies' in poetry:
                    for dep, version in poetry['dependencies'].items():
                        if dep != 'python':
                            self.dependencies[dep] = version
                
                if 'dev-dependencies' in poetry:
                    self.dev_dependencies = poetry['dev-dependencies']
            
            # Build system
            if 'build-system' in data:
                build = data['build-system']
                if 'requires' in build:
                    if 'setuptools' in str(build['requires']):
                        self.build_tool = 'setuptools'
                    elif 'poetry' in str(build['requires']):
                        self.build_tool = 'poetry'
        except Exception as e:
            logger.debug(f"Failed to parse pyproject.toml: {e}")
    
    def _parse_requirements_txt(self, requirements: Path):
        """Parse requirements.txt file"""
        try:
            content = requirements.read_text()
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse requirement
                    if '==' in line:
                        name, version = line.split('==', 1)
                        self.dependencies[name] = version
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                        self.dependencies[name] = f'>={version}'
                    else:
                        self.dependencies[line] = '*'
        except Exception as e:
            logger.debug(f"Failed to parse requirements.txt: {e}")
    
    def _parse_package_json(self, package_json: Path):
        """Parse package.json file"""
        try:
            with open(package_json) as f:
                data = json.load(f)
            
            self.name = data.get('name')
            self.version = data.get('version')
            self.description = data.get('description')
            
            if 'main' in data:
                self.entry_point = data['main']
            
            if 'dependencies' in data:
                self.dependencies = data['dependencies']
            
            if 'devDependencies' in data:
                self.dev_dependencies = data['devDependencies']
            
            # Check for TypeScript
            if 'typescript' in self.dev_dependencies or 'typescript' in self.dependencies:
                self.language = 'typescript'
        except Exception as e:
            logger.debug(f"Failed to parse package.json: {e}")
    
    def _parse_pom_xml(self, pom: Path):
        """Parse pom.xml file"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(pom)
            root = tree.getroot()
            
            # Handle namespace
            ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
            
            # Extract artifact info
            group_id = root.find('m:groupId', ns)
            artifact_id = root.find('m:artifactId', ns)
            version = root.find('m:version', ns)
            
            if artifact_id is not None:
                self.name = artifact_id.text
            if version is not None:
                self.version = version.text
            
            # Extract dependencies
            dependencies = root.find('m:dependencies', ns)
            if dependencies is not None:
                for dep in dependencies.findall('m:dependency', ns):
                    dep_artifact = dep.find('m:artifactId', ns)
                    dep_version = dep.find('m:version', ns)
                    if dep_artifact is not None:
                        name = dep_artifact.text
                        ver = dep_version.text if dep_version is not None else '*'
                        self.dependencies[name] = ver
        except Exception as e:
            logger.debug(f"Failed to parse pom.xml: {e}")
    
    def _parse_build_gradle(self, build_gradle: Path):
        """Parse build.gradle file"""
        try:
            content = build_gradle.read_text()
            
            # Extract dependencies (simple pattern matching)
            dep_pattern = re.compile(r'implementation\s+[\'"]([^\'"]+):([^\'"]+):([^\'"]+)')
            for match in dep_pattern.finditer(content):
                artifact = match.group(2)
                version = match.group(3)
                self.dependencies[artifact] = version
        except Exception as e:
            logger.debug(f"Failed to parse build.gradle: {e}")
    
    def _parse_go_mod(self, go_mod: Path):
        """Parse go.mod file"""
        try:
            content = go_mod.read_text()
            
            # Extract module name
            module_match = re.search(r'^module\s+(.+)$', content, re.MULTILINE)
            if module_match:
                self.name = module_match.group(1)
            
            # Extract Go version
            go_match = re.search(r'^go\s+(.+)$', content, re.MULTILINE)
            if go_match:
                self.metadata['go_version'] = go_match.group(1)
            
            # Extract dependencies
            require_block = re.search(r'require\s*\((.*?)\)', content, re.DOTALL)
            if require_block:
                for line in require_block.group(1).split('\n'):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            self.dependencies[parts[0]] = parts[1]
        except Exception as e:
            logger.debug(f"Failed to parse go.mod: {e}")
    
    def _parse_cargo_toml(self, cargo_toml: Path):
        """Parse Cargo.toml file"""
        try:
            import toml
            data = toml.load(cargo_toml)
            
            if 'package' in data:
                pkg = data['package']
                self.name = pkg.get('name')
                self.version = pkg.get('version')
                self.description = pkg.get('description')
            
            if 'dependencies' in data:
                self.dependencies = data['dependencies']
            
            if 'dev-dependencies' in data:
                self.dev_dependencies = data['dev-dependencies']
        except Exception as e:
            logger.debug(f"Failed to parse Cargo.toml: {e}")

@dataclass
class FileContext:
    """Comprehensive file context information"""
    # Basic info
    path: Path
    name: str
    extension: str
    file_type: FileType = FileType.UNKNOWN
    
    # Content
    content: Optional[str] = None
    size: int = 0
    hash: Optional[str] = None
    
    # Encoding
    encoding: FileEncoding = field(default_factory=lambda: FileEncoding('utf-8', 1.0))
    
    # Statistics
    statistics: FileStatistics = field(default_factory=FileStatistics)
    
    # Language/Framework
    language: Optional[str] = None
    framework: Optional[str] = None
    
    # Git info
    git_info: GitInfo = field(default_factory=GitInfo)
    
    # Dependencies
    dependencies: DependencyInfo = field(default_factory=DependencyInfo)
    
    # Project context
    project_context: Optional[ProjectContext] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    
    # Flags
    is_test: bool = False
    is_generated: bool = False
    is_vendor: bool = False
    is_binary: bool = False
    is_minified: bool = False
    
    # Related files
    related_files: List[Path] = field(default_factory=list)
    test_files: List[Path] = field(default_factory=list)
    
    # Metadata
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path], 
                  read_content: bool = True,
                  analyze_project: bool = True) -> 'FileContext':
        """Create FileContext from file path"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        context = cls(
            path=path,
            name=path.name,
            extension=path.suffix.lower()
        )
        
        # Get file stats
        stat = path.stat()
        context.size = stat.st_size
        context.created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        context.modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        context.accessed_at = datetime.fromtimestamp(stat.st_atime, tz=timezone.utc)
        
        # Detect encoding
        context.encoding = FileEncoding.detect(path)
        
        # Detect MIME type
        context.mime_type = mimetypes.guess_type(str(path))[0]
        
        # Read content if requested
        if read_content and context.size < 10 * 1024 * 1024:  # Skip files > 10MB
            try:
                context.content = path.read_text(encoding=context.encoding.encoding)
                context.hash = hashlib.sha256(context.content.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"Failed to read file content: {e}")
                context.is_binary = True
        
        # Detect language
        context._detect_language()
        
        # Calculate statistics
        if context.content:
            context.statistics = FileStatistics.calculate(context.content, context.language)
        
        # Extract dependencies
        if context.content and context.language:
            context.dependencies = DependencyInfo.extract(context.content, context.language)
        
        # Get git info
        context.git_info = GitInfo.from_file(path)
        
        # Get project context
        if analyze_project:
            context.project_context = ProjectContext.detect(path)
            if context.project_context.framework:
                context.framework = context.project_context.framework
        
        # Classify file type
        context._classify_file_type()
        
        # Detect related files
        context._find_related_files()
        
        return context
    
    def _detect_language(self):
        """Detect programming language"""
        # Extension-based detection
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'objectivec',
            '.mm': 'objectivecpp',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'bash',
            '.ps1': 'powershell',
            '.lua': 'lua',
            '.pl': 'perl',
        }
        
        self.language = extension_map.get(self.extension)
        
        # Shebang detection for scripts
        if not self.language and self.content:
            first_line = self.content.split('\n')[0]
            if first_line.startswith('#!'):
                if 'python' in first_line:
                    self.language = 'python'
                elif 'node' in first_line:
                    self.language = 'javascript'
                elif 'bash' in first_line or 'sh' in first_line:
                    self.language = 'shell'
                elif 'ruby' in first_line:
                    self.language = 'ruby'
                elif 'perl' in first_line:
                    self.language = 'perl'
    
    def _classify_file_type(self):
        """Classify the file type"""
        name_lower = self.name.lower()
        
        # Test files
        if any(pattern in name_lower for pattern in ['test', 'spec', '_test.', '.test.']):
            self.file_type = FileType.TEST
            self.is_test = True
        
        # Documentation
        elif self.extension in ['.md', '.rst', '.txt', '.adoc']:
            self.file_type = FileType.DOCUMENTATION
            if 'readme' in name_lower:
                self.file_type = FileType.README
            elif 'license' in name_lower:
                self.file_type = FileType.LICENSE
            elif 'changelog' in name_lower or 'history' in name_lower:
                self.file_type = FileType.CHANGELOG
        
        # Configuration
        elif self.extension in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']:
            self.file_type = FileType.CONFIG
        elif name_lower in ['makefile', 'dockerfile', 'jenkinsfile']:
            self.file_type = FileType.BUILD
        elif 'docker' in name_lower:
            self.file_type = FileType.DOCKERFILE
        elif name_lower.endswith('.env'):
            self.file_type = FileType.ENVIRONMENT
        
        # Build files
        elif name_lower in ['pom.xml', 'build.gradle', 'setup.py', 'package.json']:
            self.file_type = FileType.BUILD
        elif name_lower in ['requirements.txt', 'pipfile', 'gemfile', 'cargo.toml']:
            self.file_type = FileType.MANIFEST
        elif name_lower.endswith('.lock'):
            self.file_type = FileType.LOCKFILE
        
        # Database
        elif self.extension in ['.sql', '.ddl', '.dml']:
            self.file_type = FileType.SCHEMA
            if 'migration' in name_lower:
                self.file_type = FileType.MIGRATION
            elif 'seed' in name_lower:
                self.file_type = FileType.SEED
        
        # Source code
        elif self.language:
            self.file_type = FileType.SOURCE
            if self.extension in ['.h', '.hpp', '.hxx']:
                self.file_type = FileType.HEADER
        
        # Generated files
        if any(pattern in name_lower for pattern in ['generated', '.g.', '.pb.']):
            self.is_generated = True
            self.file_type = FileType.GENERATED
        
        # Minified files
        if '.min.' in name_lower:
            self.is_minified = True
            self.file_type = FileType.MINIFIED
        
        # Vendor files
        if any(vendor in str(self.path) for vendor in ['vendor/', 'node_modules/', 'lib/', 'third_party/']):
            self.is_vendor = True
            self.file_type = FileType.VENDOR
    
    def _find_related_files(self):
        """Find related files"""
        if not self.path.parent.exists():
            return
        
        stem = self.path.stem
        parent = self.path.parent
        
        # Look for test files
        test_patterns = [
            f'{stem}_test.*',
            f'test_{stem}.*',
            f'{stem}.test.*',
            f'{stem}.spec.*'
        ]
        
        for pattern in test_patterns:
            for test_file in parent.glob(pattern):
                if test_file != self.path:
                    self.test_files.append(test_file)
        
        # Look for related implementation/header files
        if self.extension in ['.h', '.hpp']:
            # Look for implementation
            impl_extensions = ['.cpp', '.cc', '.cxx', '.c']
            for ext in impl_extensions:
                impl_file = parent / f'{stem}{ext}'
                if impl_file.exists():
                    self.related_files.append(impl_file)
        
        elif self.extension in ['.cpp', '.cc', '.cxx', '.c']:
            # Look for header
            header_extensions = ['.h', '.hpp', '.hxx']
            for ext in header_extensions:
                header_file = parent / f'{stem}{ext}'
                if header_file.exists():
                    self.related_files.append(header_file)
    
    def get_relative_path(self, base: Optional[Path] = None) -> Path:
        """Get relative path from base or project root"""
        if base is None and self.project_context:
            base = self.project_context.root_path
        
        if base:
            try:
                return self.path.relative_to(base)
            except ValueError:
                return self.path
        
        return self.path
    
    def is_importable(self) -> bool:
        """Check if file is importable as a module"""
        if not self.language:
            return False
        
        if self.language == 'python':
            return not self.name.startswith('_') and self.extension == '.py'
        elif self.language in ['javascript', 'typescript']:
            return self.extension in ['.js', '.ts', '.jsx', '.tsx']
        elif self.language == 'java':
            return self.extension == '.java' and 'public class' in (self.content or '')
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'path': str(self.path),
            'name': self.name,
            'extension': self.extension,
            'file_type': self.file_type.value,
            'size': self.size,
            'hash': self.hash,
            'language': self.language,
            'framework': self.framework,
            'encoding': self.encoding.encoding,
            'statistics': asdict(self.statistics),
            'dependencies': asdict(self.dependencies),
            'git_info': asdict(self.git_info),
            'project_context': asdict(self.project_context) if self.project_context else None,
            'is_test': self.is_test,
            'is_generated': self.is_generated,
            'is_vendor': self.is_vendor,
            'is_binary': self.is_binary,
            'is_minified': self.is_minified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'modified_at': self.modified_at.isoformat() if self.modified_at else None,
            'mime_type': self.mime_type,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return f"FileContext(path={self.path}, type={self.file_type.value}, language={self.language})"

def analyze_file(file_path: Union[str, Path]) -> FileContext:
    """Convenience function to analyze a file"""
    return FileContext.from_file(file_path)

def analyze_directory(directory: Union[str, Path]) -> List[FileContext]:
    """Analyze all files in a directory"""
    dir_path = Path(directory) if isinstance(directory, str) else directory
    contexts = []
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            try:
                context = FileContext.from_file(file_path, read_content=False)
                contexts.append(context)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
    
    return contexts