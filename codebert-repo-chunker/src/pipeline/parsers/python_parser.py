import re
try:
    import tomllib
except ImportError:
    import toml as tomllib # Fallback if installed, otherwise handled
    
from pathlib import Path
from typing import List, Dict, Any

from src.pipeline.parsers.base_parser import BaseManifestParser, Dependency

class PythonParser(BaseManifestParser):
    """Parser for Python dependency files"""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.name in [
            'requirements.txt', 
            'setup.py', 
            'pyproject.toml', 
            'Pipfile'
        ] or (file_path.name.endswith('.txt') and 'requirements' in file_path.name) or file_path.suffix == '.py'

    def parse(self, file_path: Path, content: str) -> List[Dependency]:
        filename = file_path.name
        
        if 'requirements' in filename or filename.endswith('.txt'):
            return self._parse_requirements(content, filename)
        elif filename == 'setup.py':
            return self._parse_setup_py(content, filename)
        elif filename == 'pyproject.toml':
            return self._parse_pyproject(content, filename)
        elif file_path.suffix == '.py':
            return self._parse_python_source(content, filename)
        
        return []

    def _parse_python_source(self, content: str, filename: str) -> List[Dependency]:
        """Parse imports from python source files"""
        deps = []
        try:
            from src.utils.import_extractor import ImportExtractor
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            
            imports = ImportExtractor.extract_imports(content, 'python')
            
            for i, module_name in enumerate(imports):
                deps.append(Dependency(
                    name=module_name,
                    version='*',
                    type='import',
                    source_file=filename,
                    line_number=0 # ImportExtractor doesn't return line numbers yet
                ))
        except Exception as e:
            logger.warning(f"Failed to parse python source {filename}: {e}")
            
        return deps

    def _parse_requirements(self, content: str, filename: str) -> List[Dependency]:
        deps = []
        # Regex for "package>=1.2.3"
        # Groups: 1=name, 2=operator, 3=version
        pattern = re.compile(r'^([a-zA-Z0-9_\-]+)\s*([<>=!~]+)?\s*([0-9a-zA-Z_\.\*]+)?')
        
        for i, line in enumerate(content.splitlines()):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            match = pattern.match(line)
            if match:
                deps.append(Dependency(
                    name=match.group(1),
                    version=f"{match.group(2) or ''}{match.group(3) or '*'}",
                    type='runtime',
                    source_file=filename,
                    line_number=i+1
                ))
        return deps

    def _parse_setup_py(self, content: str, filename: str) -> List[Dependency]:
        deps = []
        # Basic regex scan for install_requires=[...]
        # This is simplified; setup.py is executable so parsing is hard without executing
        block_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if block_match:
            block = block_match.group(1)
            for raw_dep in re.findall(r'[\'"]([^\'"]+)[\'"]', block):
                # reuse requirements parsing logic for "pkg>=ver" string
                parsed = self._parse_requirements(raw_dep, filename)
                if parsed:
                    deps.extend(parsed)
        return deps

    def _parse_pyproject(self, content: str, filename: str) -> List[Dependency]:
        deps = []
        try:
            # Simple TOML parsing if library available
            # If standard tomllib (py3.11+) or toml package not present, fallback to regex
            # Here assuming simple regex fallback for robustness if imports fail or partial parsing needed
            
            # [project] dependencies
            in_dependencies = False
            for i, line in enumerate(content.splitlines()):
                line = line.strip()
                if line.startswith('[project]') or line.startswith('[tool.poetry.dependencies]'):
                    in_dependencies = True
                    continue
                if line.startswith('[') and in_dependencies:
                    in_dependencies = False
                
                if in_dependencies and '=' in line:
                    parts = line.split('=', 1)
                    name = parts[0].strip().strip('"\'')
                    version = parts[1].strip().strip('"\'')
                    deps.append(Dependency(
                        name=name, 
                        version=version, 
                        type='runtime', 
                        source_file=filename,
                        line_number=i+1
                    ))
        except Exception:
            pass
        return deps
