from pathlib import Path
from typing import List
import re
import logging

from src.pipeline.parsers.base_parser import BaseManifestParser, Dependency

logger = logging.getLogger(__name__)

class TerraformParser(BaseManifestParser):
    """
    Parser for Terraform (.tf) files.
    Detects modules and providers.
    """
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix == '.tf'

    def parse(self, file_path: Path, content: str) -> List[Dependency]:
        deps = []
        
        # 1. Detect Modules
        # module "name" { source = "..." version = "..." }
        # Simplified regex to capture module name and optionally source
        # This regex is multiline-aware but simple
        
        # Find module blocks
        module_blocks = re.finditer(r'module\s+"([^"]+)"\s*\{', content)
        
        for match in module_blocks:
            module_name = match.group(1)
            # Try to find source within the block (heuristic: look ahead limited chars)
            start = match.end()
            block_sample = content[start:start+500] # Look at next 500 chars
            
            source_match = re.search(r'source\s*=\s*"([^"]+)"', block_sample)
            version_match = re.search(r'version\s*=\s*"([^"]+)"', block_sample)
            
            source = source_match.group(1) if source_match else "unknown"
            version = version_match.group(1) if version_match else None
            
            deps.append(Dependency(
                name=source if source != "unknown" else module_name, # source is the real dep
                version=version or "latest",
                type="terraform_module",
                source_file=file_path.name
            ))

        # 2. Detect Providers
        # provider "aws" { ... }
        # plugin "aws" is implied
        providers = re.findall(r'provider\s+"([^"]+)"', content)
        for provider in providers:
            deps.append(Dependency(
                name=provider,
                version="latest",
                type="terraform_provider",
                source_file=file_path.name
            ))

        # 3. Required Providers (terraform block)
        # required_providers { aws = { source = "hashicorp/aws" ... } }
        # This is harder with regex, but we can look for "hashicorp/name" strings commonly
        
        return deps
