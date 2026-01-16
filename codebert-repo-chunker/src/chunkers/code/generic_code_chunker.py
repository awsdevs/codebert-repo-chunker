"""
Generic code chunker for programming languages without specific chunkers
Uses common code patterns, syntax highlighting, and structure detection
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from src.utils.logger import get_logger
from enum import Enum
import pygments
from pygments.lexers import get_lexer_for_filename, guess_lexer, get_lexer_by_name
from pygments.token import Token
from pygments.util import ClassNotFound

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = get_logger(__name__)

class CodePattern(Enum):
    """Common code patterns across languages"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    MODULE = "module"
    NAMESPACE = "namespace"
    BLOCK = "block"
    COMMENT_BLOCK = "comment_block"
    IMPORT = "import"
    VARIABLE = "variable"
    CONSTANT = "constant"

@dataclass
class GenericCodeBlock:
    """Generic code block representation"""
    pattern: CodePattern
    name: Optional[str]
    content: str
    start_line: int
    end_line: int
    indent_level: int
    tokens: List[Tuple[Token, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['GenericCodeBlock'] = field(default_factory=list)
    parent: Optional['GenericCodeBlock'] = None

class SyntaxPatternDetector:
    """Detects common syntax patterns across languages"""
    
    # Common function/method patterns
    FUNCTION_PATTERNS = [
        # C-style: return_type function_name(params)
        r'^\s*(?:(?:public|private|protected|static|virtual|async|export|const)\s+)*'
        r'(?:[\w<>\[\]]+\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*{',
        
        # Python/Ruby style: def function_name(params):
        r'^\s*(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:',
        r'^\s*def\s+(\w+)\s*\([^)]*\)',
        
        # JavaScript/TypeScript arrow functions
        r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
        r'^\s*(\w+)\s*:\s*(?:async\s+)?\([^)]*\)\s*=>',
        
        # Go style: func (receiver) name(params) returns
        r'^\s*func\s+(?:\(\w+\s+[\w\*]+\)\s+)?(\w+)\s*\([^)]*\)',
        
        # Rust style: fn name(params) -> return_type
        r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\([^)]*\)',
        
        # VB.NET style
        r'^\s*(?:Public|Private|Protected)\s+(?:Shared\s+)?(?:Function|Sub)\s+(\w+)',
        
        # Swift style
        r'^\s*(?:public|private|internal|open)?\s*func\s+(\w+)\s*\([^)]*\)',
        
        # R style
        r'^\s*(\w+)\s*<-\s*function\s*\(',
        
        # Kotlin
        r'^\s*(?:public|private|protected|internal)?\s*fun\s+(\w+)\s*\([^)]*\)',
        
        # Scala
        r'^\s*def\s+(\w+)\s*(?:\[[^\]]*\])?\s*\([^)]*\)',
        
        # PHP
        r'^\s*(?:public|private|protected)?\s*function\s+(\w+)\s*\(',
        
        # Perl
        r'^\s*sub\s+(\w+)\s*{',
        
        # Lua
        r'^\s*(?:local\s+)?function\s+(\w+)\s*\(',
        r'^\s*(\w+)\s*=\s*function\s*\('
    ]
    
    # Common class patterns
    CLASS_PATTERNS = [
        # C++/C#/Java style
        r'^\s*(?:public|private|protected)?\s*(?:abstract|final|static)?\s*class\s+(\w+)',
        
        # Python
        r'^\s*class\s+(\w+)\s*(?:\([^)]*\))?\s*:',
        
        # JavaScript/TypeScript
        r'^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)',
        
        # Ruby
        r'^\s*class\s+(\w+)(?:\s*<\s*\w+)?',
        
        # Swift
        r'^\s*(?:public|private|internal|open)?\s*class\s+(\w+)',
        
        # Kotlin
        r'^\s*(?:public|private|protected|internal)?\s*(?:data\s+)?class\s+(\w+)',
        
        # Go (struct as class)
        r'^\s*type\s+(\w+)\s+struct\s*{',
        
        # Rust
        r'^\s*(?:pub\s+)?struct\s+(\w+)',
        
        # VB.NET
        r'^\s*(?:Public|Private)\s+Class\s+(\w+)',
        
        # PHP
        r'^\s*(?:abstract\s+)?(?:final\s+)?class\s+(\w+)',
        
        # Perl (package)
        r'^\s*package\s+(\w+);'
    ]
    
    # Interface patterns
    INTERFACE_PATTERNS = [
        r'^\s*(?:public|private)?\s*interface\s+(\w+)',
        r'^\s*type\s+(\w+)\s+interface\s*{',  # Go
        r'^\s*(?:export\s+)?interface\s+(\w+)',  # TypeScript
        r'^\s*protocol\s+(\w+)',  # Swift
        r'^\s*trait\s+(\w+)',  # Rust/Scala
    ]
    
    # Import/Include patterns
    IMPORT_PATTERNS = [
        r'^\s*import\s+.*',
        r'^\s*from\s+\S+\s+import\s+.*',
        r'^\s*#include\s*[<"].*[>"]',
        r'^\s*using\s+.*',
        r'^\s*require\s*\(.*\)',
        r'^\s*require_once\s*\(.*\)',
        r'^\s*use\s+.*',
        r'^\s*package\s+.*',
        r'^\s*const\s+\w+\s*=\s*require\(',
        r'^\s*load\s*\(.*\)',
        r'^\s*source\s*\(.*\)',
    ]
    
    # Variable/Constant declaration patterns
    VARIABLE_PATTERNS = [
        # Type-annotated variables
        r'^\s*(?:public|private|protected)?\s*(?:static|final|const)?\s*([\w<>\[\]]+)\s+(\w+)\s*[=;]',
        
        # Dynamic languages
        r'^\s*(?:const|let|var)\s+(\w+)\s*=',
        r'^\s*(\w+)\s*=\s*[^=]',  # Simple assignment
        r'^\s*(\w+)\s*:=',  # Go short declaration
        
        # Constant declarations
        r'^\s*#define\s+(\w+)',
        r'^\s*const\s+(\w+)',
        r'^\s*final\s+.*\s+(\w+)',
    ]
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[CodePattern, List[re.Pattern]]:
        """Compile regex patterns for efficiency"""
        return {
            CodePattern.FUNCTION: [re.compile(p, re.MULTILINE) for p in self.FUNCTION_PATTERNS],
            CodePattern.CLASS: [re.compile(p, re.MULTILINE) for p in self.CLASS_PATTERNS],
            CodePattern.INTERFACE: [re.compile(p, re.MULTILINE) for p in self.INTERFACE_PATTERNS],
            CodePattern.IMPORT: [re.compile(p, re.MULTILINE) for p in self.IMPORT_PATTERNS],
            CodePattern.VARIABLE: [re.compile(p, re.MULTILINE) for p in self.VARIABLE_PATTERNS],
        }
    
    def detect_pattern(self, line: str) -> Optional[Tuple[CodePattern, str]]:
        """
        Detect code pattern in a line
        
        Returns:
            Tuple of (pattern_type, matched_name) or None
        """
        for pattern_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    # Extract name if captured
                    name = match.group(1) if match.groups() else None
                    return (pattern_type, name)
        return None

class IndentationAnalyzer:
    """Analyzes code indentation patterns"""
    
    def __init__(self):
        self.indent_char = None
        self.indent_width = None
        self.indent_style = None
    
    def analyze_indentation(self, lines: List[str]) -> Dict[str, Any]:
        """
        Analyze indentation style of code
        
        Returns:
            Dictionary with indentation info
        """
        indent_counts = defaultdict(int)
        space_widths = []
        
        for line in lines:
            if not line or not line[0].isspace():
                continue
            
            # Count leading whitespace
            indent = len(line) - len(line.lstrip())
            indent_char = '\t' if line[0] == '\t' else ' '
            
            if indent_char == ' ':
                space_widths.append(indent)
                indent_counts['spaces'] += 1
            else:
                indent_counts['tabs'] += 1
        
        # Determine indent style
        if indent_counts['tabs'] > indent_counts['spaces']:
            self.indent_style = 'tabs'
            self.indent_char = '\t'
            self.indent_width = 1
        else:
            self.indent_style = 'spaces'
            self.indent_char = ' '
            # Find common space width (likely 2, 4, or 8)
            if space_widths:
                from math import gcd
                from functools import reduce
                # Find GCD of all indentation widths
                common_width = reduce(gcd, space_widths)
                self.indent_width = common_width if common_width > 0 else 4
            else:
                self.indent_width = 4
        
        return {
            'style': self.indent_style,
            'char': self.indent_char,
            'width': self.indent_width
        }
    
    def get_indent_level(self, line: str) -> int:
        """Get indentation level of a line"""
        if not line or not line[0].isspace():
            return 0
        
        indent = len(line) - len(line.lstrip())
        
        if self.indent_width:
            return indent // self.indent_width
        else:
            return indent

class BracketTracker:
    """Tracks bracket/brace pairs for scope detection"""
    
    OPENING_BRACKETS = {'(', '[', '{', '<'}
    CLOSING_BRACKETS = {')', ']', '}', '>'}
    BRACKET_PAIRS = {'(': ')', '[': ']', '{': '}', '<': '>'}
    
    def __init__(self):
        self.stack = []
        self.depth = 0
    
    def process_line(self, line: str) -> Dict[str, Any]:
        """
        Process a line and track brackets
        
        Returns:
            Dictionary with bracket info
        """
        in_string = False
        in_comment = False
        escape_next = False
        string_char = None
        
        for i, char in enumerate(line):
            # Handle escape sequences
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            # Handle strings
            if char in ['"', "'", '`'] and not in_comment:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue
            
            # Skip if in string
            if in_string:
                continue
            
            # Handle comments (simplified)
            if i < len(line) - 1:
                two_char = line[i:i+2]
                if two_char in ['//', '--', '#']:
                    in_comment = True
                    break
                elif two_char == '/*':
                    in_comment = True
                    continue
                elif two_char == '*/':
                    in_comment = False
                    continue
            
            # Skip if in comment
            if in_comment:
                continue
            
            # Track brackets
            if char in self.OPENING_BRACKETS:
                self.stack.append(char)
                self.depth += 1
            elif char in self.CLOSING_BRACKETS:
                if self.stack:
                    # Check for matching bracket
                    expected_opener = None
                    for opener, closer in self.BRACKET_PAIRS.items():
                        if closer == char:
                            expected_opener = opener
                            break
                    
                    if self.stack[-1] == expected_opener:
                        self.stack.pop()
                        self.depth -= 1
        
        return {
            'depth': self.depth,
            'has_opening': any(c in line for c in self.OPENING_BRACKETS),
            'has_closing': any(c in line for c in self.CLOSING_BRACKETS),
            'balanced': len(self.stack) == 0
        }
    
    def is_balanced(self) -> bool:
        """Check if all brackets are balanced"""
        return len(self.stack) == 0
    
    def reset(self):
        """Reset tracker state"""
        self.stack = []
        self.depth = 0

class TokenAnalyzer:
    """Analyzes code using Pygments tokenization"""
    
    def __init__(self, lexer):
        self.lexer = lexer
    
    def tokenize(self, content: str) -> List[Tuple[Token, str]]:
        """
        Tokenize code content
        
        Returns:
            List of (token_type, text) tuples
        """
        try:
            return list(self.lexer.get_tokens(content))
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return []
    
    def extract_structure(self, tokens: List[Tuple[Token, str]]) -> Dict[str, List[str]]:
        """
        Extract code structure from tokens
        
        Returns:
            Dictionary of structural elements
        """
        structure = {
            'functions': [],
            'classes': [],
            'variables': [],
            'comments': [],
            'strings': [],
            'keywords': [],
            'operators': []
        }
        
        for token_type, text in tokens:
            if token_type in Token.Name.Function:
                structure['functions'].append(text)
            elif token_type in Token.Name.Class:
                structure['classes'].append(text)
            elif token_type in Token.Name.Variable or token_type in Token.Name:
                structure['variables'].append(text)
            elif token_type in Token.Comment:
                structure['comments'].append(text)
            elif token_type in Token.String:
                structure['strings'].append(text)
            elif token_type in Token.Keyword:
                structure['keywords'].append(text)
            elif token_type in Token.Operator:
                structure['operators'].append(text)
        
        # Remove duplicates
        for key in structure:
            structure[key] = list(set(structure[key]))
        
        return structure
    
    def identify_blocks(self, tokens: List[Tuple[Token, str]]) -> List[Tuple[int, int]]:
        """
        Identify code blocks from tokens
        
        Returns:
            List of (start_pos, end_pos) tuples
        """
        blocks = []
        current_pos = 0
        block_stack = []
        
        for token_type, text in tokens:
            if text == '{':
                block_stack.append(current_pos)
            elif text == '}' and block_stack:
                start = block_stack.pop()
                blocks.append((start, current_pos))
            
            current_pos += len(text)
        
        return blocks

class CommentExtractor:
    """Extracts and classifies comments"""
    
    # Comment patterns for various languages
    COMMENT_PATTERNS = {
        'line_double_slash': re.compile(r'//.*$', re.MULTILINE),
        'line_hash': re.compile(r'#.*$', re.MULTILINE),
        'line_dash': re.compile(r'--.*$', re.MULTILINE),
        'line_percent': re.compile(r'%.*$', re.MULTILINE),
        'block_c': re.compile(r'/\*.*?\*/', re.DOTALL),
        'block_html': re.compile(r'<!--.*?-->', re.DOTALL),
        'block_python': re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL),
        'block_lua': re.compile(r'--\[\[.*?\]\]', re.DOTALL),
        'javadoc': re.compile(r'/\*\*.*?\*/', re.DOTALL),
    }
    
    def extract_comments(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract all comments from code
        
        Returns:
            List of comment dictionaries
        """
        comments = []
        lines = content.split('\n')
        
        # Extract single-line comments
        for i, line in enumerate(lines):
            for pattern_name, pattern in self.COMMENT_PATTERNS.items():
                if 'line' in pattern_name:
                    match = pattern.search(line)
                    if match:
                        comments.append({
                            'type': 'line',
                            'content': match.group(),
                            'line': i + 1,
                            'pattern': pattern_name
                        })
        
        # Extract block comments
        for pattern_name, pattern in self.COMMENT_PATTERNS.items():
            if 'block' in pattern_name or 'javadoc' in pattern_name:
                for match in pattern.finditer(content):
                    # Find line number
                    start_line = content[:match.start()].count('\n') + 1
                    end_line = content[:match.end()].count('\n') + 1
                    
                    comments.append({
                        'type': 'block',
                        'content': match.group(),
                        'start_line': start_line,
                        'end_line': end_line,
                        'pattern': pattern_name,
                        'is_documentation': 'javadoc' in pattern_name or '"""' in match.group()
                    })
        
        return comments

class GenericCodeChunker(BaseChunker):
    """Generic code chunker for any programming language"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.pattern_detector = SyntaxPatternDetector()
        self.indentation_analyzer = IndentationAnalyzer()
        self.bracket_tracker = BracketTracker()
        self.comment_extractor = CommentExtractor()
        
        # Chunking strategies
        self.strategies = {
            'function_based': self._chunk_by_functions,
            'class_based': self._chunk_by_classes,
            'block_based': self._chunk_by_blocks,
            'indentation_based': self._chunk_by_indentation,
            'line_based': self._chunk_by_lines,
            'hybrid': self._chunk_hybrid
        }
    
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from generic code file
        
        Args:
            content: Source code content
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Get lexer for syntax highlighting
            lexer = self._get_lexer(file_context.path)
            
            if lexer:
                # Use token-based analysis
                return self._chunk_with_lexer(content, file_context, lexer)
            else:
                # Fall back to pattern-based analysis
                return self._chunk_without_lexer(content, file_context)
                
        except Exception as e:
            logger.error(f"Error chunking code file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _get_lexer(self, file_path: Path):
        """Get Pygments lexer for file"""
        try:
            # Try by filename
            return get_lexer_for_filename(str(file_path))
        except ClassNotFound:
            try:
                # Try to guess from content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(1000)
                return guess_lexer(sample)
            except:
                return None
    
    def _chunk_with_lexer(self, content: str, file_context: FileContext,
                         lexer) -> List[Chunk]:
        """Chunk using Pygments lexer"""
        chunks = []
        
        # Tokenize content
        token_analyzer = TokenAnalyzer(lexer)
        tokens = token_analyzer.tokenize(content)
        
        # Extract structure
        structure = token_analyzer.extract_structure(tokens)
        
        # Analyze content
        lines = content.split('\n')
        self.indentation_analyzer.analyze_indentation(lines)
        
        # Detect code blocks
        blocks = self._detect_code_blocks(lines)
        
        # Choose chunking strategy based on structure
        if structure['classes'] and len(structure['classes']) > 2:
            strategy = 'class_based'
        elif structure['functions'] and len(structure['functions']) > 3:
            strategy = 'function_based'
        elif blocks and len(blocks) > 5:
            strategy = 'block_based'
        else:
            strategy = 'hybrid'
        
        # Apply chosen strategy
        chunks = self.strategies[strategy](content, file_context, structure)
        
        # Add metadata from lexer
        for chunk in chunks:
            if 'annotations' not in chunk.metadata:
                chunk.metadata['annotations'] = {}
            chunk.metadata['annotations']['language'] = lexer.name
            chunk.metadata['annotations']['lexer_tokens'] = len(tokens)
        
        return chunks
    
    def _chunk_without_lexer(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Chunk without lexer using pattern detection"""
        lines = content.split('\n')
        
        # Analyze indentation
        self.indentation_analyzer.analyze_indentation(lines)
        
        # Extract comments
        comments = self.comment_extractor.extract_comments(content)
        
        # Detect code patterns
        patterns = self._detect_patterns(lines)
        
        # Build structure
        structure = {
            'functions': [p[1] for p in patterns if p[0] == CodePattern.FUNCTION],
            'classes': [p[1] for p in patterns if p[0] == CodePattern.CLASS],
            'imports': [p[1] for p in patterns if p[0] == CodePattern.IMPORT],
            'comments': comments
        }
        
        # Choose strategy
        if structure['classes']:
            return self._chunk_by_classes(content, file_context, structure)
        elif structure['functions']:
            return self._chunk_by_functions(content, file_context, structure)
        else:
            return self._chunk_hybrid(content, file_context, structure)
    
    def _detect_patterns(self, lines: List[str]) -> List[Tuple[CodePattern, str, int]]:
        """Detect code patterns in lines"""
        patterns = []
        
        for i, line in enumerate(lines):
            result = self.pattern_detector.detect_pattern(line)
            if result:
                pattern_type, name = result
                patterns.append((pattern_type, name, i))
        
        return patterns
    
    def _detect_code_blocks(self, lines: List[str]) -> List[GenericCodeBlock]:
        """Detect code blocks using indentation and brackets"""
        blocks = []
        block_stack = []
        current_block = None
        
        self.bracket_tracker.reset()
        
        for i, line in enumerate(lines):
            indent_level = self.indentation_analyzer.get_indent_level(line)
            
            # Check for pattern
            pattern_result = self.pattern_detector.detect_pattern(line)
            
            if pattern_result:
                pattern_type, name = pattern_result
                
                # Start new block
                new_block = GenericCodeBlock(
                    pattern=pattern_type,
                    name=name,
                    content="",
                    start_line=i,
                    end_line=i,
                    indent_level=indent_level,
                    tokens=[],
                    metadata={}
                )
                
                # Handle nesting
                if current_block and indent_level > current_block.indent_level:
                    new_block.parent = current_block
                    current_block.children.append(new_block)
                
                blocks.append(new_block)
                block_stack.append(new_block)
                current_block = new_block
            
            # Track brackets for block boundaries
            bracket_info = self.bracket_tracker.process_line(line)
            
            # Update current block
            if current_block:
                current_block.end_line = i
                
                # Check if block ends (indentation decrease or bracket close)
                if indent_level < current_block.indent_level or \
                   (bracket_info['depth'] == 0 and bracket_info['has_closing']):
                    if block_stack:
                        block_stack.pop()
                        current_block = block_stack[-1] if block_stack else None
        
        return blocks
    
    def _chunk_by_functions(self, content: str, file_context: FileContext,
                          structure: Dict[str, Any]) -> List[Chunk]:
        """Chunk by function definitions"""
        chunks = []
        lines = content.split('\n')
        
        # Find function boundaries
        function_blocks = self._find_function_blocks(lines, structure.get('functions', []))
        
        # Create header chunk with imports
        header_chunk = self._create_header_chunk(lines, file_context)
        if header_chunk:
            chunks.append(header_chunk)
        
        # Create chunks for each function
        for block in function_blocks:
            function_content = '\n'.join(lines[block['start']:block['end'] + 1])
            
            if self.count_tokens(function_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=function_content,
                    chunk_type='code_function',
                    metadata={
                        'function_name': block['name'],
                        'line_range': (block['start'], block['end']),
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path),
                    start_line=block['start'],
                    end_line=block['end']
                ))
            else:
                # Split large function
                split_chunks = self._split_large_function(
                    function_content, block['name'], block['start'], file_context
                )
                chunks.extend(split_chunks)
        
        # Add remaining code as separate chunk
        remaining = self._get_remaining_code(lines, function_blocks)
        if remaining:
            chunks.append(self.create_chunk(
                content=remaining,
                chunk_type='code_global',
                metadata={
                    'contains': 'global_code',
                    'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_by_classes(self, content: str, file_context: FileContext,
                        structure: Dict[str, Any]) -> List[Chunk]:
        """Chunk by class definitions"""
        chunks = []
        lines = content.split('\n')
        
        # Find class boundaries
        class_blocks = self._find_class_blocks(lines, structure.get('classes', []))
        
        # Create header chunk
        header_chunk = self._create_header_chunk(lines, file_context)
        if header_chunk:
            chunks.append(header_chunk)
        
        # Create chunks for each class
        for block in class_blocks:
            class_content = '\n'.join(lines[block['start']:block['end'] + 1])
            
            if self.count_tokens(class_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=class_content,
                    chunk_type='code_class',
                    metadata={
                        'class_name': block['name'],
                        'line_range': (block['start'], block['end']),
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path),
                    start_line=block['start'],
                    end_line=block['end']
                ))
            else:
                # Split large class into methods
                split_chunks = self._split_class_into_methods(
                    class_content, block['name'], block['start'], file_context
                )
                chunks.extend(split_chunks)
        
        return chunks
    
    def _chunk_by_blocks(self, content: str, file_context: FileContext,
                       structure: Dict[str, Any]) -> List[Chunk]:
        """Chunk by code blocks (bracket-delimited sections)"""
        chunks = []
        lines = content.split('\n')
        blocks = self._detect_code_blocks(lines)
        
        current_chunk_lines = []
        current_tokens = 0
        
        for block in blocks:
            block_lines = lines[block.start_line:block.end_line + 1]
            block_content = '\n'.join(block_lines)
            block_tokens = self.count_tokens(block_content)
            
            if block_tokens <= self.max_tokens:
                # Create chunk for block
                chunks.append(self.create_chunk(
                    content=block_content,
                    chunk_type='code_block',
                    metadata={
                        'block_type': block.pattern.value,
                        'block_name': block.name,
                        'indent_level': block.indent_level,
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path),
                    start_line=block.start_line,
                    end_line=block.end_line
                ))
            else:
                # Split large block
                split_chunks = self._split_large_block(block_content, block, file_context)
                chunks.extend(split_chunks)
        
        return chunks
    
    def _chunk_by_indentation(self, content: str, file_context: FileContext,
                             structure: Dict[str, Any]) -> List[Chunk]:
        """Chunk by indentation levels (for Python-like languages)"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_indent = 0
        chunk_start = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                current_chunk.append(line)
                continue
            
            indent_level = self.indentation_analyzer.get_indent_level(line)
            
            # Start new chunk on indent decrease to 0
            if indent_level == 0 and current_chunk and current_indent > 0:
                chunk_content = '\n'.join(current_chunk)
                
                if self.count_tokens(chunk_content) <= self.max_tokens:
                    chunks.append(self.create_chunk(
                        content=chunk_content,
                        chunk_type='code_indented',
                        metadata={
                            'indent_based': True,
                            'max_indent': current_indent,
                            'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                        },
                        file_path=str(file_context.path),
                        start_line=chunk_start,
                        end_line=i - 1
                    ))
                else:
                    # Split if too large
                    split_chunks = self._split_by_size(chunk_content, chunk_start, file_context)
                    chunks.extend(split_chunks)
                
                current_chunk = [line]
                chunk_start = i
                current_indent = indent_level
            else:
                current_chunk.append(line)
                current_indent = max(current_indent, indent_level)
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type='code_indented',
                metadata={
                    'indent_based': True,
                    'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                },
                file_path=str(file_context.path),
                start_line=chunk_start,
                end_line=len(lines) - 1
            ))
        
        return chunks
    
    def _chunk_by_lines(self, content: str, file_context: FileContext,
                       structure: Dict[str, Any]) -> List[Chunk]:
        """Simple line-based chunking"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_start = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='code_lines',
                    metadata={
                        'line_based': True,
                        'line_count': len(current_chunk),
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path),
                    start_line=chunk_start,
                    end_line=i - 1
                ))
                current_chunk = []
                current_tokens = 0
                chunk_start = i
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='code_lines',
                metadata={
                    'line_based': True,
                    'line_count': len(current_chunk),
                    'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                },
                file_path=str(file_context.path),
                start_line=chunk_start,
                end_line=len(lines) - 1
            ))
        
        return chunks
    
    def _chunk_hybrid(self, content: str, file_context: FileContext,
                     structure: Dict[str, Any]) -> List[Chunk]:
        """Hybrid chunking combining multiple strategies"""
        chunks = []
        lines = content.split('\n')
        
        # Extract high-level structure
        imports = []
        functions = []
        classes = []
        global_code = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            pattern_result = self.pattern_detector.detect_pattern(line)
            
            if pattern_result:
                pattern_type, name = pattern_result
                
                if pattern_type == CodePattern.IMPORT:
                    imports.append(line)
                elif pattern_type == CodePattern.FUNCTION:
                    # Find function end
                    func_end = self._find_block_end(lines, i)
                    functions.append({
                        'name': name,
                        'start': i,
                        'end': func_end,
                        'content': '\n'.join(lines[i:func_end + 1])
                    })
                    i = func_end
                elif pattern_type == CodePattern.CLASS:
                    # Find class end
                    class_end = self._find_block_end(lines, i)
                    classes.append({
                        'name': name,
                        'start': i,
                        'end': class_end,
                        'content': '\n'.join(lines[i:class_end + 1])
                    })
                    i = class_end
                else:
                    global_code.append(line)
            else:
                global_code.append(line)
            
            i += 1
        
        # Create chunks
        
        # Imports chunk
        if imports:
            chunks.append(self.create_chunk(
                content='\n'.join(imports),
                chunk_type='code_imports',
                metadata={
                    'import_count': len(imports),
                    'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                },
                file_path=str(file_context.path)
            ))
        
        # Class chunks
        for class_info in classes:
            if self.count_tokens(class_info['content']) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=class_info['content'],
                    chunk_type='code_class',
                    metadata={
                        'class_name': class_info['name'],
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path),
                    start_line=class_info['start'],
                    end_line=class_info['end']
                ))
            else:
                # Split large class
                split_chunks = self._split_by_size(
                    class_info['content'], 
                    class_info['start'], 
                    file_context
                )
                chunks.extend(split_chunks)
        
        # Function chunks
        for func_info in functions:
            if self.count_tokens(func_info['content']) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=func_info['content'],
                    chunk_type='code_function',
                    metadata={
                        'function_name': func_info['name'],
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path),
                    start_line=func_info['start'],
                    end_line=func_info['end']
                ))
            else:
                # Split large function
                split_chunks = self._split_by_size(
                    func_info['content'], 
                    func_info['start'], 
                    file_context
                )
                chunks.extend(split_chunks)
        
        # Global code chunk
        if global_code:
            global_content = '\n'.join(global_code)
            if self.count_tokens(global_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=global_content,
                    chunk_type='code_global',
                    metadata={
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path)
                ))
            else:
                # Split by size
                split_chunks = self._split_by_size(global_content, 0, file_context)
                chunks.extend(split_chunks)
        
        return chunks if chunks else self._chunk_by_lines(content, file_context, structure)
    
    def _find_block_end(self, lines: List[str], start: int) -> int:
        """Find the end of a code block"""
        if start >= len(lines):
            return start
        
        # Get initial indentation
        initial_indent = self.indentation_analyzer.get_indent_level(lines[start])
        
        # Track brackets for languages that use them
        self.bracket_tracker.reset()
        self.bracket_tracker.process_line(lines[start])
        
        for i in range(start + 1, len(lines)):
            line = lines[i]
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check indentation
            current_indent = self.indentation_analyzer.get_indent_level(line)
            
            # Process brackets
            self.bracket_tracker.process_line(line)
            
            # Block ends when:
            # 1. Indentation returns to same or lower level (Python-style)
            # 2. Brackets are balanced and we see a closing bracket (C-style)
            if current_indent <= initial_indent and i > start + 1:
                if self.bracket_tracker.depth == 0:
                    return i - 1
            
            # Also check if brackets are balanced after a closing bracket
            if self.bracket_tracker.is_balanced() and '}' in line:
                return i
        
        return len(lines) - 1
    
    def _find_function_blocks(self, lines: List[str], 
                            function_names: List[str]) -> List[Dict[str, Any]]:
        """Find function block boundaries"""
        blocks = []
        
        for i, line in enumerate(lines):
            pattern_result = self.pattern_detector.detect_pattern(line)
            
            if pattern_result and pattern_result[0] == CodePattern.FUNCTION:
                name = pattern_result[1] or f"function_{i}"
                end = self._find_block_end(lines, i)
                
                blocks.append({
                    'name': name,
                    'start': i,
                    'end': end
                })
        
        return blocks
    
    def _find_class_blocks(self, lines: List[str], 
                         class_names: List[str]) -> List[Dict[str, Any]]:
        """Find class block boundaries"""
        blocks = []
        
        for i, line in enumerate(lines):
            pattern_result = self.pattern_detector.detect_pattern(line)
            
            if pattern_result and pattern_result[0] == CodePattern.CLASS:
                name = pattern_result[1] or f"class_{i}"
                end = self._find_block_end(lines, i)
                
                blocks.append({
                    'name': name,
                    'start': i,
                    'end': end
                })
        
        return blocks
    
    def _create_header_chunk(self, lines: List[str], 
                           file_context: FileContext) -> Optional[Chunk]:
        """Create header chunk with imports and file-level comments"""
        header_lines = []
        
        for line in lines:
            # Stop at first function/class definition
            pattern_result = self.pattern_detector.detect_pattern(line)
            if pattern_result and pattern_result[0] in [CodePattern.FUNCTION, CodePattern.CLASS]:
                break
            
            # Include imports, comments, and global declarations
            if line.strip() and (
                pattern_result and pattern_result[0] == CodePattern.IMPORT or
                line.strip().startswith('#') or
                line.strip().startswith('//') or
                line.strip().startswith('/*') or
                not self._is_code_definition(line)
            ):
                header_lines.append(line)
        
        if not header_lines:
            return None
        
        header_content = '\n'.join(header_lines)
        
        return self.create_chunk(
            content=header_content,
            chunk_type='code_header',
            metadata={
                'contains': 'imports_and_declarations',
                'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
            },
            file_path=str(file_context.path)
        )
    
    def _is_code_definition(self, line: str) -> bool:
        """Check if line is a code definition (function, class, etc.)"""
        pattern_result = self.pattern_detector.detect_pattern(line)
        return pattern_result and pattern_result[0] in [
            CodePattern.FUNCTION, CodePattern.CLASS, 
            CodePattern.INTERFACE, CodePattern.STRUCT
        ]
    
    def _get_remaining_code(self, lines: List[str], 
                          blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Get code not included in function/class blocks"""
        if not blocks:
            return '\n'.join(lines)
        
        remaining_lines = []
        covered_ranges = [(b['start'], b['end']) for b in blocks]
        
        for i, line in enumerate(lines):
            # Check if line is in any block
            in_block = any(start <= i <= end for start, end in covered_ranges)
            
            if not in_block:
                remaining_lines.append(line)
        
        return '\n'.join(remaining_lines) if remaining_lines else None
    
    def _split_large_function(self, content: str, name: str, start_line: int,
                            file_context: FileContext) -> List[Chunk]:
        """Split a large function into smaller chunks"""
        chunks = []
        lines = content.split('\n')
        
        # Keep function signature in each chunk
        signature_lines = []
        for i, line in enumerate(lines):
            signature_lines.append(line)
            if '{' in line or ':' in line:  # End of signature
                break
        
        signature = '\n'.join(signature_lines)
        remaining_lines = lines[len(signature_lines):]
        
        # Split remaining by size
        current_chunk = signature_lines.copy()
        current_tokens = self.count_tokens(signature)
        part = 1
        
        for line in remaining_lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens * 0.9:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='code_function_part',
                    metadata={
                        'function_name': name,
                        'part': part,
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = [f"// ... continuation of {name} ..."]
                current_tokens = self.count_tokens(current_chunk[0])
                part += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='code_function_part',
                metadata={
                    'function_name': name,
                    'part': part,
                    'is_last': True,
                    'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _split_class_into_methods(self, content: str, name: str, start_line: int,
                                 file_context: FileContext) -> List[Chunk]:
        """Split a large class into method-level chunks"""
        chunks = []
        lines = content.split('\n')
        
        # Extract class declaration
        class_decl_lines = []
        for i, line in enumerate(lines):
            class_decl_lines.append(line)
            if '{' in line:
                break
        
        class_declaration = '\n'.join(class_decl_lines)
        
        # Find methods within class
        method_blocks = []
        i = len(class_decl_lines)
        
        while i < len(lines):
            pattern_result = self.pattern_detector.detect_pattern(lines[i])
            
            if pattern_result and pattern_result[0] == CodePattern.FUNCTION:
                method_name = pattern_result[1] or f"method_{i}"
                method_end = self._find_block_end(lines, i)
                
                method_blocks.append({
                    'name': method_name,
                    'start': i,
                    'end': min(method_end, len(lines) - 1)
                })
                i = method_end + 1
            else:
                i += 1
        
        # Create chunks for methods
        for method in method_blocks:
            method_lines = [f"// Class: {name}", ""] + lines[method['start']:method['end'] + 1]
            method_content = '\n'.join(method_lines)
            
            if self.count_tokens(method_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=method_content,
                    chunk_type='code_method',
                    metadata={
                        'class_name': name,
                        'method_name': method['name'],
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path)
                ))
            else:
                # Split large method
                split_chunks = self._split_by_size(method_content, method['start'], file_context)
                chunks.extend(split_chunks)
        
        return chunks
    
    def _split_large_block(self, content: str, block: GenericCodeBlock,
                          file_context: FileContext) -> List[Chunk]:
        """Split a large code block"""
        return self._split_by_size(content, block.start_line, file_context)
    
    def _split_by_size(self, content: str, start_line: int,
                      file_context: FileContext) -> List[Chunk]:
        """Split content by token size"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        part = 1
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens * 0.9 and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='code_part',
                    metadata={
                        'part': part,
                        'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
                part += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='code_part',
                metadata={
                    'part': part,
                    'is_last': True,
                    'language': file_context.language if hasattr(file_context, 'language') else 'unknown'
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback to simple line-based chunking"""
        logger.warning(f"Using fallback chunking for {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        chunk_size = 50  # Lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            # Ensure within token limit
            while self.count_tokens(chunk_content) > self.max_tokens and chunk_lines:
                chunk_lines = chunk_lines[:-1]
                chunk_content = '\n'.join(chunk_lines)
            
            if chunk_lines:
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='code_fallback',
                    metadata={
                        'is_fallback': True,
                        'line_range': (i, min(i + len(chunk_lines), len(lines))),
                        'language': 'unknown'
                    },
                    file_path=str(file_context.path),
                    start_line=i,
                    end_line=min(i + len(chunk_lines), len(lines))
                ))
        
        return chunks