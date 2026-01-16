"""
src/utils/import_extractor.py
Unified utility for extracting imports from various languages.
Replaces duplicate logic in ChunkProcessor, PythonParser, and RelationshipBuilder.
"""
import re
import ast
from typing import List, Set
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ImportExtractor:
    """
    Extracts import statements from code content for supported languages.
    Uses AST parsing where possible (Python) and robust regex for others.
    """
    
    SUPPORTED_LANGUAGES = {
        '.py', 
        '.java', 
        '.js', '.jsx', '.ts', '.tsx', 
        '.go', 
        '.c', '.cpp', '.h', '.hpp'
    }
    
    @staticmethod
    def extract_imports(content: str, language: str) -> List[str]:
        """
        Extract imports from code content.
        
        Args:
            content (str): Source code content
            language (str): Language identifier (python, java, javascript, etc.)
            
        Returns:
            List[str]: List of imported module/package names
        """
        if not content or not language:
            return []
            
        language = language.lower()
        imports = set()
        
        try:
            if language == 'python':
                imports.update(ImportExtractor._extract_python_imports(content))
            elif language == 'java':
                imports.update(ImportExtractor._extract_java_imports(content))
            elif language in ['javascript', 'typescript', 'ts', 'js']:
                imports.update(ImportExtractor._extract_js_imports(content))
            elif language == 'go':
                imports.update(ImportExtractor._extract_go_imports(content))
            elif language in ['c', 'cpp', 'c++']:
                imports.update(ImportExtractor._extract_cpp_imports(content))
        except Exception as e:
            logger.warning(f"Failed to extract imports for {language}: {e}")
            
        return sorted(list(imports))

    @staticmethod
    def _extract_python_imports(content: str) -> Set[str]:
        imports = set()
        # 1. Try AST Parsing (Most robust)
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
                    elif node.level > 0:
                        # Relative import (e.g. from . import utils) -> skip or handle?
                        # For repo-level graph, relative imports are tricky without file path context.
                        # We'll skip bare relative imports for now or treat as local.
                        pass
        except SyntaxError:
            # Fallback to Regex if AST fails (e.g. invalid syntax in chunk)
            pass
        except Exception as e:
            logger.debug(f"Python AST parse failed: {e}")

        # 2. Regex Fallback (Catches things AST might miss in partial chunks)
        # matches: import numpy; from os import path
        regex_imports = re.findall(r'^(?:from\s+(\S+)|import\s+(\S+))', content, re.MULTILINE)
        for match in regex_imports:
            module = match[0] or match[1]
            if module:
                imports.add(module.split('.')[0])
                
        return imports

    @staticmethod
    def _extract_java_imports(content: str) -> Set[str]:
        # import java.util.List;
        pattern = re.compile(r'^\s*import\s+[\w\.]+\.?(\w+)\s*;', re.MULTILINE)
        # We might want the full package for Java? 
        # Reviewer mentioned "duplicate extraction logic" - let's check what was verified.
        # Usually full package is better for Java: java.util.List
        full_pattern = re.compile(r'^\s*import\s+([\w\.]+)\s*;', re.MULTILINE)
        return set(full_pattern.findall(content))

    @staticmethod
    def _extract_js_imports(content: str) -> Set[str]:
        imports = set()
        # import X from 'Y'
        pattern1 = re.compile(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]')
        # require('Y')
        pattern2 = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]')
        # import('Y')
        pattern3 = re.compile(r'import\s*\(\s*[\'"]([^\'"]+)[\'"]')
        
        for p in [pattern1, pattern2, pattern3]:
            imports.update(p.findall(content))
        return imports

    @staticmethod
    def _extract_go_imports(content: str) -> Set[str]:
        imports = set()
        # import "fmt"
        # import ( "fmt" \n "os" )
        
        # Single line
        pattern_single = re.compile(r'import\s+"([^"]+)"')
        imports.update(pattern_single.findall(content))
        
        # Multi-line block
        pattern_block = re.compile(r'import\s+\(([^)]+)\)', re.DOTALL)
        blocks = pattern_block.findall(content)
        for block in blocks:
            # Extract items inside quotes
            block_imports = re.findall(r'"([^"]+)"', block)
            imports.update(block_imports)
            
        return imports

    @staticmethod
    def _extract_cpp_imports(content: str) -> Set[str]:
        # #include <iostream> or #include "header.h"
        pattern = re.compile(r'#include\s+[<"]([^>"]+)[>"]')
        return set(pattern.findall(content))
