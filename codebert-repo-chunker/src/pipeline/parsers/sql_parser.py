from pathlib import Path
from typing import List, Set
import re
import logging
import yaml

from src.pipeline.parsers.base_parser import BaseManifestParser, Dependency

logger = logging.getLogger(__name__)

class SQLParser(BaseManifestParser):
    """
    Parser for SQL files and dbt projects.
    Detects dbt refs, sources, and basic table references.
    """
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.sql' or file_path.name in ('dbt_project.yml', 'dbt_project.yaml')

    def parse(self, file_path: Path, content: str) -> List[Dependency]:
        deps = []
        
        # 1. dbt project configuration
        if file_path.name in ('dbt_project.yml', 'dbt_project.yaml'):
            try:
                config = yaml.safe_load(content)
                if config and 'name' in config:
                    # The project itself is not a dependency, but we can extract 'require-dbt-version'
                    # or packages (usually in packages.yml, but sometimes hinted here)
                    if 'require-dbt-version' in config:
                        deps.append(Dependency(
                            name="dbt-core", 
                            version=str(config['require-dbt-version']),
                            type="tool",
                            source_file=file_path.name
                        ))
            except Exception as e:
                logger.warning(f"Error parsing dbt project: {e}")
                
        # 2. SQL files (dbt style)
        elif file_path.suffix.lower() == '.sql':
            # Detect dbt ref()
            # ref('model_name') or ref("model_name")
            refs = re.findall(r"ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", content)
            for ref in refs:
                deps.append(Dependency(
                    name=ref,
                    version="latest",
                    type="dbt_model",
                    source_file=file_path.name
                ))
                
            # Detect dbt source()
            # source('source_name', 'table_name')
            sources = re.findall(r"source\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)", content)
            for source, table in sources:
                deps.append(Dependency(
                    name=f"{source}.{table}",
                    version="latest",
                    type="dbt_source",
                    source_file=file_path.name
                ))
            
            # Detect standard SQL JOIN/FROM (Basic heuristic)
            # FROM table_name or JOIN table_name
            # This is very loose and might catch aliases, keywords etc. defaulting to minimal extraction
            # Focusing on Fully Qualified Names (schema.table) or obvious tables
            sql_refs = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z0-9_"\.]+)', content, re.IGNORECASE)
            for ref in sql_refs:
                # Filter out obvious keywords if they sneaked in (though regex requires space before)
                if '.' in ref: # Assume qualified names are interesting external deps
                    clean_ref = ref.strip('"`')
                    deps.append(Dependency(
                        name=clean_ref,
                        version="unknown",
                        type="sql_table",
                        source_file=file_path.name
                    ))

        return deps
