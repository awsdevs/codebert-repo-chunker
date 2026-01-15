from typing import List, Dict, Any, Optional, Type
from pathlib import Path
import logging

from src.pipeline.parsers.base_parser import BaseManifestParser, Dependency

logger = logging.getLogger(__name__)

class DependencyResolver:
    """
    Orchestrates dependency resolution by delegating to specific parsers.
    Acts as the Registry for all manifest parsers.
    """
    
    def __init__(self):
        self._parsers: List[BaseManifestParser] = []
        self._register_default_parsers()
        
    def _register_default_parsers(self):
        """Register built-in parsers"""
        try:
            from src.pipeline.parsers.python_parser import PythonParser
            from src.pipeline.parsers.node_parser import NodeParser
            from src.pipeline.parsers.java_parser import JavaParser
            from src.pipeline.parsers.sql_parser import SQLParser
            from src.pipeline.parsers.terraform_parser import TerraformParser
            
            self.register(PythonParser())
            self.register(NodeParser())
            self.register(JavaParser())
            self.register(SQLParser())
            self.register(TerraformParser())
        except ImportError as e:
            logger.warning(f"Could not register default parsers: {e}")
        
    def register(self, parser: BaseManifestParser):
        """Register a new parser"""
        self._parsers.append(parser)
        logger.debug(f"Registered dependency parser: {parser.__class__.__name__}")
        
    def resolve(self, file_path: Path, content: str) -> List[Dependency]:
        """
        Resolve dependencies for a given file.
        Returns empty list if no parser supports this file.
        """
        deps = []
        parsed = False
        
        for parser in self._parsers:
            if parser.can_parse(file_path):
                try:
                    parser_deps = parser.parse(file_path, content)
                    deps.extend(parser_deps)
                    parsed = True
                except Exception as e:
                    logger.error(f"Error parsing {file_path} with {parser.__class__.__name__}: {e}")
                    
        if parsed:
            logger.info(f"Resolved {len(deps)} dependencies from {file_path.name}")
            
        return deps

    def analyze_repository(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Analyze dependencies for a list of files.
        Returns a dictionary with dependency graph and import map.
        """
        graph = {} # file -> list of dependencies
        imports = {} # library -> list of files importing it
        
        for file_path in file_paths:
            try:
                if not file_path.exists():
                    continue
                    
                # Optimization: Check if any parser *can* parse it first
                supported = False
                for parser in self._parsers:
                    if parser.can_parse(file_path):
                        supported = True
                        break
                
                if not supported:
                    continue
                    
                try:
                    content = file_path.read_text(errors='ignore')
                except Exception:
                    continue
                    
                deps = self.resolve(file_path, content)
                if deps:
                    graph[str(file_path)] = [d.to_dict() for d in deps]
                    
                    for dep in deps:
                        if dep.name not in imports:
                            imports[dep.name] = []
                        imports[dep.name].append(str(file_path))
                        
            except Exception as e:
                logger.warning(f"Failed to analyze dependencies for {file_path}: {e}")
                
        return {
            "dependency_graph": graph,
            "imports_map": imports,
            "total_dependencies": sum(len(d) for d in graph.values())
        }
