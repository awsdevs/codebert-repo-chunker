import json
from pathlib import Path
from typing import List, Dict, Any

from src.pipeline.parsers.base_parser import BaseManifestParser, Dependency

class NodeParser(BaseManifestParser):
    """Parser for Node.js package.json"""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.name == 'package.json'

    def parse(self, file_path: Path, content: str) -> List[Dependency]:
        deps = []
        try:
            data = json.loads(content)
            
            # Runtime dependencies
            if 'dependencies' in data:
                for name, version in data['dependencies'].items():
                    deps.append(Dependency(
                        name=name,
                        version=version,
                        type='runtime',
                        source_file='package.json'
                    ))
            
            # Dev dependencies
            if 'devDependencies' in data:
                for name, version in data['devDependencies'].items():
                    deps.append(Dependency(
                        name=name,
                        version=version,
                        type='dev',
                        source_file='package.json'
                    ))
                    
        except json.JSONDecodeError:
            pass
            
        return deps
