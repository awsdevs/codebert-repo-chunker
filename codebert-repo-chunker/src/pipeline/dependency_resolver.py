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
        
        # 1. Build a map of potential module names to file paths
        # e.g., "src.utils.metrics" -> "/abs/path/to/src/utils/metrics.py"
        module_map = {}
        
        # Find the root source directory to calculate relative module names
        # Heuristic: Use common prefix or assume 'src' is root if present
        # strictly speaking, we should look for __init__.py but we can try simple heuristics
        
        # We'll map "filename_stem" -> path (weak) and "dir.filename" -> path (stronger)
        for fp in file_paths:
            if not fp.exists(): continue
            
            # map "src.utils.metrics"
            try:
                # varied attempts to guess python module path
                parts = fp.parts
                if 'src' in parts:
                    src_index = parts.index('src')
                    rel_parts = parts[src_index:]
                    module_name = '.'.join(rel_parts).replace('.py', '')
                    module_map[module_name] = str(fp)
                
                # Also just map the filename for simple local imports
                module_map[fp.stem] = str(fp)
            except Exception:
                pass

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
                    # Resolve internal dependencies
                    resolved_deps = []
                    for d in deps:
                        # If this dependency matches a known internal file, point to it
                        if d.name in module_map:
                            # It's an internal dependency!
                            # We can mark it or just keep the name, but for the graph logic 
                            # (which usually uses names), we might need to handle this downstream.
                            # actually, let's keep the dependency object but maybe add a metadata field?
                            # Or simpler: The report generator uses the name. 
                            # If we want the graph to connect, we need the "target" of the edge to match the "id" of the file node.
                            # The node IDs in report_generator are str(file_path)
                            pass
                            
                    graph[str(file_path)] = [d.to_dict() for d in deps]
                    
                    for dep in deps:
                        target = dep.name
                        # Link to internal file if possible
                        if dep.name in module_map:
                             # This is key for the graph visualization to link nodes
                             # The ReportGenerator likely uses 'name' as target. 
                             # We should seemingly update the explicit dependency list passed to ReportGenerator?
                             # Or we rely on ReportGenerator to know about this mapping?
                             # Since ReportGenerator receives this graph, let's leave it here but maybe note it.
                             pass

                        if target not in imports:
                            imports[target] = []
                        imports[target].append(str(file_path))
                        
            except Exception as e:
                logger.error(f"Error analyzing dependencies for {file_path}: {e}")
                
        return {
            "graph": graph,
            "imports": imports,
            "module_map": module_map, # Return this so ReportGenerator can link them!
            "total_dependencies": sum(len(d) for d in graph.values())
        }

