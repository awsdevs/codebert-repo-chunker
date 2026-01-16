import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import List

from src.pipeline.parsers.base_parser import BaseManifestParser, Dependency

class JavaParser(BaseManifestParser):
    """Parser for Java Maven (pom.xml) and Gradle (build.gradle)"""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.name in ['pom.xml', 'build.gradle']

    def parse(self, file_path: Path, content: str) -> List[Dependency]:
        if file_path.name == 'pom.xml':
            return self._parse_maven(content)
        elif file_path.name == 'build.gradle':
            return self._parse_gradle(content)
        return []

    def _parse_maven(self, content: str) -> List[Dependency]:
        deps = []
        try:
            # Strip namespace for easier parsing
            content = re.sub(r'xmlns="[^"]+"', '', content, count=1)
            root = ET.fromstring(content)
            
            for dep in root.findall(".//dependency"):
                group_id = dep.find('groupId')
                artifact_id = dep.find('artifactId')
                version = dep.find('version')
                scope = dep.find('scope')
                
                if group_id is not None and artifact_id is not None:
                    name = f"{group_id.text}:{artifact_id.text}"
                    ver = version.text if version is not None else "*"
                    dep_type = scope.text if scope is not None else 'compile'
                    
                    deps.append(Dependency(
                        name=name,
                        version=ver,
                        type=dep_type,
                        source_file='pom.xml'
                    ))
        except ET.ParseError as e:
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Failed to parse maven {file_path}: {e}")
        return deps

    def _parse_gradle(self, content: str) -> List[Dependency]:
        deps = []
        # Basic Regex for "implementation 'group:name:version'"
        # Groups: 1=configuration, 2=quote, 3=dep string
        pattern = re.compile(r'(\w+)\s+([\'"])([^:"\']+:[^:"\']+:[^:"\']+)([\'"])')
        
        for i, line in enumerate(content.splitlines()):
            line = line.strip()
            if line.startswith('//'): continue
            
            match = pattern.search(line)
            if match:
                config = match.group(1) # implementation, testImplementation
                dep_str = match.group(3)
                
                parts = dep_str.split(':')
                if len(parts) >= 2:
                    name = f"{parts[0]}:{parts[1]}"
                    version = parts[2] if len(parts) > 2 else "*"
                    
                    deps.append(Dependency(
                        name=name,
                        version=version,
                        type=config,
                        source_file='build.gradle',
                        line_number=i+1
                    ))
        return deps
