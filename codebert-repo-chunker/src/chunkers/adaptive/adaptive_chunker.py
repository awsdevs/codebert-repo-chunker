"""
Adaptive chunker for unknown file types
Analyzes content structure and applies intelligent chunking strategies
"""

import re
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import logging

from src.core.base_chunker import BaseChunker, Chunk
from src.core.file_context import FileContext
from src.utils.text_utils import TextAnalyzer
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ContentPattern:
    """Represents a detected content pattern"""
    pattern_type: str
    confidence: float
    indicators: List[str]
    suggested_strategy: str
    metadata: Dict[str, Any]

@dataclass
class StructureAnalysis:
    """Results of structural analysis"""
    line_count: int
    avg_line_length: float
    max_line_length: int
    indent_style: str
    indent_levels: List[int]
    blank_line_ratio: float
    has_consistent_structure: bool
    detected_sections: List[Tuple[int, int]]
    entropy_profile: List[float]

class ContentAnalyzer:
    """Analyzes content to determine structure and patterns"""
    
    # Pattern definitions for content detection
    PATTERN_DEFINITIONS = {
        'code': {
            'indicators': [
                r'\bfunction\s+\w+\s*\(',
                r'\bdef\s+\w+\s*\(',
                r'\bclass\s+\w+',
                r'\bpublic\s+class\s+',
                r'\bprivate\s+\w+\s+\w+\s*\(',
                r'\bif\s*\([^)]+\)\s*{',
                r'\bfor\s*\([^)]+\)\s*{',
                r'\breturn\s+[^;]+;',
                r'=>\s*{',
                r'\bconst\s+\w+\s*=',
                r'\bvar\s+\w+\s*=',
                r'\blet\s+\w+\s*='
            ],
            'threshold': 3,
            'strategy': 'code_block'
        },
        'configuration': {
            'indicators': [
                r'^\s*[\w\-\.]+\s*[:=]\s*.*$',
                r'^\s*<[\w\-]+>.*</[\w\-]+>\s*$',
                r'^\[[\w\-\.]+\]',
                r'^\s*[\w\-]+\s*:\s*\S+',
                r'^\s*export\s+[\w_]+='
            ],
            'threshold': 5,
            'strategy': 'config_section'
        },
        'structured_data': {
            'indicators': [
                r'^[^,]+,[^,]+,[^,]+',
                r'^\w+\t\w+\t\w+',
                r'^\|[^|]+\|[^|]+\|',
                r'^"[^"]+","[^"]+","[^"]+"'
            ],
            'threshold': 10,
            'strategy': 'data_rows'
        },
        'script': {
            'indicators': [
                r'^#!/',
                r'\$\w+',
                r'\${[\w_]+}',
                r'echo\s+["\']',
                r'if\s+\[\[',
                r'export\s+[\w_]+=',
                r'source\s+\S+',
                r'\|\s*grep\s+',
                r'\|\s*awk\s+'
            ],
            'threshold': 3,
            'strategy': 'script_block'
        },
        'markup': {
            'indicators': [
                r'<[\w\-]+\s*[^>]*>',
                r'</[\w\-]+>',
                r'<!--.*-->',
                r'<!\[CDATA\[',
                r'&\w+;',
                r'^\s*#\s+\w+',
                r'^\s*##\s+\w+',
                r'\[[\w\s]+\]\([^)]+\)'
            ],
            'threshold': 5,
            'strategy': 'markup_section'
        },
        'log': {
            'indicators': [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}:\d{2}:\d{2}',
                r'\b(?:ERROR|WARN|INFO|DEBUG|TRACE)\b',
                r'\[[\w\.\-]+\]',
                r'^\d+\s+\w+\s+\d+',
                r'at\s+[\w\.]+\([\w\.]+:\d+\)'
            ],
            'threshold': 5,
            'strategy': 'log_entry'
        }
    }
    
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Perform comprehensive content analysis"""
        lines = content.split('\n')
        
        # Basic metrics
        metrics = self._calculate_metrics(lines)
        
        # Pattern detection
        patterns = self._detect_patterns(content, lines)
        
        # Structure analysis
        structure = self._analyze_structure(lines)
        
        # Entropy analysis
        entropy_profile = self._calculate_entropy_profile(lines)
        
        # Determine best strategy
        strategy = self._determine_strategy(patterns, structure, metrics)
        
        return {
            'metrics': metrics,
            'patterns': patterns,
            'structure': structure,
            'entropy_profile': entropy_profile,
            'strategy': strategy,
            'confidence': self._calculate_confidence(patterns, structure)
        }
    
    def _calculate_metrics(self, lines: List[str]) -> Dict[str, Any]:
        """Calculate basic content metrics"""
        if not lines:
            return {
                'line_count': 0,
                'char_count': 0,
                'word_count': 0,
                'avg_line_length': 0,
                'max_line_length': 0,
                'unique_chars': 0
            }
        
        line_lengths = [len(line) for line in lines]
        char_count = sum(line_lengths)
        word_count = sum(len(line.split()) for line in lines)
        
        return {
            'line_count': len(lines),
            'char_count': char_count,
            'word_count': word_count,
            'avg_line_length': sum(line_lengths) / max(len(lines), 1),
            'max_line_length': max(line_lengths) if line_lengths else 0,
            'unique_chars': len(set(''.join(lines)))
        }
    
    def _detect_patterns(self, content: str, lines: List[str]) -> List[ContentPattern]:
        """Detect content patterns"""
        detected_patterns = []
        
        for pattern_type, config in self.PATTERN_DEFINITIONS.items():
            matches = []
            for pattern in config['indicators']:
                try:
                    found = re.findall(pattern, content, re.MULTILINE)
                    matches.extend(found[:10])  # Limit matches per pattern
                except re.error:
                    logger.warning(f"Invalid regex pattern: {pattern}")
                    continue
            
            if len(matches) >= config['threshold']:
                confidence = min(len(matches) / (config['threshold'] * 2), 1.0)
                detected_patterns.append(ContentPattern(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    indicators=matches[:5],
                    suggested_strategy=config['strategy'],
                    metadata={'match_count': len(matches)}
                ))
        
        return sorted(detected_patterns, key=lambda p: p.confidence, reverse=True)
    
    def _analyze_structure(self, lines: List[str]) -> StructureAnalysis:
        """Analyze structural properties of content"""
        if not lines:
            return StructureAnalysis(
                line_count=0,
                avg_line_length=0,
                max_line_length=0,
                indent_style='none',
                indent_levels=[],
                blank_line_ratio=0,
                has_consistent_structure=False,
                detected_sections=[],
                entropy_profile=[]
            )
        
        # Analyze indentation
        indent_info = self._analyze_indentation(lines)
        
        # Find blank lines
        blank_lines = sum(1 for line in lines if not line.strip())
        blank_ratio = blank_lines / max(len(lines), 1)
        
        # Detect sections
        sections = self._detect_sections(lines)
        
        # Calculate line lengths
        line_lengths = [len(line) for line in lines]
        
        return StructureAnalysis(
            line_count=len(lines),
            avg_line_length=sum(line_lengths) / max(len(lines), 1),
            max_line_length=max(line_lengths) if line_lengths else 0,
            indent_style=indent_info['style'],
            indent_levels=indent_info['levels'],
            blank_line_ratio=blank_ratio,
            has_consistent_structure=indent_info['consistent'],
            detected_sections=sections,
            entropy_profile=[]
        )
    
    def _analyze_indentation(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze indentation patterns"""
        indent_chars = []
        indent_counts = []
        
        for line in lines:
            if line and line[0] in ' \t':
                match = re.match(r'^([ \t]+)', line)
                if match:
                    indent = match.group(1)
                    indent_chars.append('\t' if '\t' in indent else ' ')
                    indent_counts.append(len(indent))
        
        if not indent_chars:
            return {'style': 'none', 'levels': [], 'consistent': False}
        
        # Determine predominant style
        char_counts = Counter(indent_chars)
        main_char = char_counts.most_common(1)[0][0]
        
        # Determine indent width
        if main_char == ' ':
            space_counts = [c for c, ch in zip(indent_counts, indent_chars) if ch == ' ']
            if space_counts:
                # Find GCD of space counts to determine indent width
                from math import gcd
                from functools import reduce
                width = reduce(gcd, space_counts)
                style = f'spaces_{width}'
            else:
                style = 'spaces'
        else:
            style = 'tabs'
        
        # Check consistency
        consistent = len(char_counts) == 1
        
        return {
            'style': style,
            'levels': sorted(set(indent_counts)),
            'consistent': consistent
        }
    
    def _detect_sections(self, lines: List[str]) -> List[Tuple[int, int]]:
        """Detect logical sections in content"""
        sections = []
        current_start = 0
        
        for i in range(len(lines)):
            # Section boundaries
            is_boundary = False
            
            # Multiple blank lines
            if i > 0 and not lines[i].strip() and not lines[i-1].strip():
                is_boundary = True
            
            # Separator lines
            if re.match(r'^[-=_*]{3,}$', lines[i].strip()):
                is_boundary = True
            
            # Headers
            if re.match(r'^#{1,6}\s+', lines[i]):
                is_boundary = True
            
            if is_boundary and i > current_start:
                sections.append((current_start, i))
                current_start = i + 1
        
        # Add final section
        if current_start < len(lines):
            sections.append((current_start, len(lines)))
        
        return sections
    
    def _calculate_entropy_profile(self, lines: List[str]) -> List[float]:
        """Calculate Shannon entropy for each line"""
        entropies = []
        
        for line in lines:
            if not line:
                entropies.append(0.0)
            else:
                entropy = self._calculate_entropy(line)
                entropies.append(entropy)
        
        return entropies
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Character frequency
        freq_map = Counter(text)
        length = len(text)
        
        entropy = 0.0
        for count in freq_map.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _determine_strategy(self, patterns: List[ContentPattern], 
                           structure: StructureAnalysis,
                           metrics: Dict[str, Any]) -> str:
        """Determine best chunking strategy"""
        if patterns and patterns[0].confidence > 0.7:
            return patterns[0].suggested_strategy
        
        # Fallback strategies based on structure
        if structure.has_consistent_structure:
            if structure.indent_style != 'none':
                return 'indented_blocks'
            elif structure.detected_sections:
                return 'section_based'
        
        if structure.blank_line_ratio > 0.2:
            return 'paragraph_based'
        
        if metrics['avg_line_length'] > 100:
            return 'long_lines'
        
        return 'intelligent_lines'
    
    def _calculate_confidence(self, patterns: List[ContentPattern],
                             structure: StructureAnalysis) -> float:
        """Calculate overall confidence in analysis"""
        confidence = 0.0
        
        if patterns:
            confidence = patterns[0].confidence * 0.6
        
        if structure.has_consistent_structure:
            confidence += 0.2
        
        if structure.detected_sections:
            confidence += 0.2
        
        return min(confidence, 1.0)

class AdaptiveChunker(BaseChunker):
    """Main adaptive chunker that handles unknown file types"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, max_tokens)
        self.content_analyzer = ContentAnalyzer()
        self.chunk_strategies = {
            'code_block': self._chunk_code_blocks,
            'config_section': self._chunk_config_sections,
            'data_rows': self._chunk_data_rows,
            'script_block': self._chunk_script_blocks,
            'markup_section': self._chunk_markup_sections,
            'log_entry': self._chunk_log_entries,
            'indented_blocks': self._chunk_by_indentation,
            'section_based': self._chunk_by_sections,
            'paragraph_based': self._chunk_by_paragraphs,
            'long_lines': self._chunk_long_lines,
            'intelligent_lines': self._chunk_intelligent_lines
        }
        
        # Cache for repeated patterns
        self.pattern_cache = {}
        self.cache_size_limit = 100
    
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Create chunks from unknown content type"""
        try:
            # Check cache
            content_hash = self._get_content_hash(content)
            if content_hash in self.pattern_cache:
                analysis = self.pattern_cache[content_hash]
                logger.debug(f"Using cached analysis for {file_context.path}")
            else:
                # Analyze content
                logger.info(f"Analyzing unknown file: {file_context.path}")
                analysis = self.content_analyzer.analyze_content(content)
                
                # Cache result
                self._cache_analysis(content_hash, analysis)
            
            # Log analysis results
            logger.info(
                f"Analysis for {file_context.path}: "
                f"strategy={analysis['strategy']}, "
                f"confidence={analysis['confidence']:.2f}"
            )
            
            # Apply chunking strategy
            strategy = analysis['strategy']
            if strategy in self.chunk_strategies:
                chunks = self.chunk_strategies[strategy](content, file_context, analysis)
            else:
                logger.warning(f"Unknown strategy {strategy}, using fallback")
                chunks = self._chunk_fallback(content, file_context)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({
                    'detection_method': 'adaptive',
                    'strategy': strategy,
                    'confidence': analysis['confidence'],
                    'patterns': [p.pattern_type for p in analysis.get('patterns', [])]
                })
            
            logger.info(f"Created {len(chunks)} chunks for {file_context.path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in adaptive chunking for {file_context.path}: {e}")
            return self._chunk_fallback(content, file_context)
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash of content for caching"""
        # Use first 10KB for hash to avoid processing huge files
        sample = content[:10240]
        return hashlib.md5(sample.encode('utf-8')).hexdigest()
    
    def _cache_analysis(self, content_hash: str, analysis: Dict[str, Any]):
        """Cache analysis results"""
        # Limit cache size
        if len(self.pattern_cache) >= self.cache_size_limit:
            # Remove oldest entries (FIFO)
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[content_hash] = analysis
    
    def _chunk_code_blocks(self, content: str, file_context: FileContext,
                          analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk content that appears to be code"""
        chunks = []
        lines = content.split('\n')
        
        current_block = []
        current_indent = 0
        block_start = 0
        
        for i, line in enumerate(lines):
            # Detect indent level
            indent = len(line) - len(line.lstrip()) if line.strip() else current_indent
            
            # Start new chunk conditions
            should_split = False
            
            # Function/class declarations
            if re.search(r'^[a-zA-Z_]\w*\s*\([^)]*\)\s*[:{]?\s*$', line.strip()):
                should_split = True
            
            # Major indent change
            if indent == 0 and current_block and len(current_block) > 5:
                should_split = True
            
            # Size limit
            if len(current_block) > 50:
                should_split = True
            
            # Token limit check
            if current_block:
                temp_content = '\n'.join(current_block + [line])
                if self.count_tokens(temp_content) > self.max_tokens:
                    should_split = True
            
            if should_split and current_block:
                chunk_content = '\n'.join(current_block)
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='adaptive_code',
                    metadata={
                        'block_type': 'code',
                        'start_line': block_start,
                        'end_line': i
                    },
                    file_path=str(file_context.path),
                    start_line=block_start,
                    end_line=i
                ))
                current_block = []
                block_start = i
            
            current_block.append(line)
            current_indent = indent
        
        # Add remaining block
        if current_block:
            chunks.append(self.create_chunk(
                content='\n'.join(current_block),
                chunk_type='adaptive_code',
                metadata={'block_type': 'code'},
                file_path=str(file_context.path),
                start_line=block_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _chunk_config_sections(self, content: str, file_context: FileContext,
                               analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk configuration-like content"""
        chunks = []
        lines = content.split('\n')
        
        current_section = []
        section_name = 'main'
        section_start = 0
        
        for i, line in enumerate(lines):
            # Detect section markers
            section_match = None
            
            # INI-style sections [section]
            ini_match = re.match(r'^\[([^\]]+)\]', line)
            if ini_match:
                section_match = ini_match.group(1)
            
            # YAML-style sections
            yaml_match = re.match(r'^(\w+):$', line)
            if yaml_match and not current_section:
                section_match = yaml_match.group(1)
            
            # Header-style sections
            header_match = re.match(r'^###?\s+(.+)', line)
            if header_match:
                section_match = header_match.group(1)
            
            if section_match:
                # Save current section
                if current_section:
                    chunks.append(self.create_chunk(
                        content='\n'.join(current_section),
                        chunk_type='adaptive_config',
                        metadata={
                            'section': section_name,
                            'block_type': 'config'
                        },
                        file_path=str(file_context.path),
                        start_line=section_start,
                        end_line=i
                    ))
                
                section_name = section_match
                current_section = [line]
                section_start = i
            else:
                current_section.append(line)
                
                # Check size limits
                if len(current_section) > 50:
                    content_str = '\n'.join(current_section)
                    if self.count_tokens(content_str) > self.max_tokens:
                        chunks.append(self.create_chunk(
                            content=content_str,
                            chunk_type='adaptive_config',
                            metadata={
                                'section': section_name,
                                'block_type': 'config'
                            },
                            file_path=str(file_context.path),
                            start_line=section_start,
                            end_line=i
                        ))
                        current_section = []
                        section_start = i + 1
        
        # Add remaining section
        if current_section:
            chunks.append(self.create_chunk(
                content='\n'.join(current_section),
                chunk_type='adaptive_config',
                metadata={
                    'section': section_name,
                    'block_type': 'config'
                },
                file_path=str(file_context.path),
                start_line=section_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _chunk_data_rows(self, content: str, file_context: FileContext,
                        analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk structured data content"""
        chunks = []
        lines = content.split('\n')
        
        # Detect delimiter
        delimiter = self._detect_delimiter(lines[:10])
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            
            # Skip empty batches
            non_empty = [line for line in batch if line.strip()]
            if not non_empty:
                continue
            
            chunk_content = '\n'.join(batch)
            
            # Check token limit
            if self.count_tokens(chunk_content) > self.max_tokens:
                # Split batch into smaller chunks
                sub_batch = []
                for line in batch:
                    sub_batch.append(line)
                    if self.count_tokens('\n'.join(sub_batch)) > self.max_tokens * 0.9:
                        chunks.append(self.create_chunk(
                            content='\n'.join(sub_batch[:-1]),
                            chunk_type='adaptive_data',
                            metadata={
                                'block_type': 'data',
                                'delimiter': delimiter,
                                'row_count': len(sub_batch) - 1
                            },
                            file_path=str(file_context.path),
                            start_line=i,
                            end_line=i + len(sub_batch) - 1
                        ))
                        sub_batch = [line]
                
                if sub_batch:
                    chunks.append(self.create_chunk(
                        content='\n'.join(sub_batch),
                        chunk_type='adaptive_data',
                        metadata={
                            'block_type': 'data',
                            'delimiter': delimiter,
                            'row_count': len(sub_batch)
                        },
                        file_path=str(file_context.path)
                    ))
            else:
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='adaptive_data',
                    metadata={
                        'block_type': 'data',
                        'delimiter': delimiter,
                        'row_count': len(non_empty)
                    },
                    file_path=str(file_context.path),
                    start_line=i,
                    end_line=min(i + batch_size, len(lines))
                ))
        
        return chunks
    
    def _detect_delimiter(self, sample_lines: List[str]) -> str:
        """Detect delimiter in structured data"""
        delimiters = [',', '\t', '|', ';', ':']
        delimiter_counts = Counter()
        
        for line in sample_lines:
            if not line.strip():
                continue
            for delim in delimiters:
                count = line.count(delim)
                if count > 0:
                    delimiter_counts[delim] += count
        
        if delimiter_counts:
            return delimiter_counts.most_common(1)[0][0]
        return ','
    
    def _chunk_script_blocks(self, content: str, file_context: FileContext,
                            analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk script-like content"""
        chunks = []
        lines = content.split('\n')
        
        current_block = []
        in_function = False
        function_name = None
        block_start = 0
        
        for i, line in enumerate(lines):
            # Detect function boundaries
            func_match = re.match(r'^(?:function\s+)?(\w+)\s*\(\)\s*{', line)
            if func_match:
                # Save current block
                if current_block:
                    chunks.append(self.create_chunk(
                        content='\n'.join(current_block),
                        chunk_type='adaptive_script',
                        metadata={
                            'block_type': 'script',
                            'function': function_name
                        },
                        file_path=str(file_context.path),
                        start_line=block_start,
                        end_line=i
                    ))
                
                in_function = True
                function_name = func_match.group(1)
                current_block = [line]
                block_start = i
            elif line.strip() == '}' and in_function:
                # End of function
                current_block.append(line)
                chunks.append(self.create_chunk(
                    content='\n'.join(current_block),
                    chunk_type='adaptive_script',
                    metadata={
                        'block_type': 'script',
                        'function': function_name
                    },
                    file_path=str(file_context.path),
                    start_line=block_start,
                    end_line=i + 1
                ))
                current_block = []
                in_function = False
                function_name = None
                block_start = i + 1
            else:
                current_block.append(line)
                
                # Check size limits
                if len(current_block) > 50:
                    content_str = '\n'.join(current_block)
                    if self.count_tokens(content_str) > self.max_tokens:
                        chunks.append(self.create_chunk(
                            content=content_str,
                            chunk_type='adaptive_script',
                            metadata={
                                'block_type': 'script',
                                'function': function_name
                            },
                            file_path=str(file_context.path),
                            start_line=block_start,
                            end_line=i
                        ))
                        current_block = []
                        block_start = i + 1
        
        # Add remaining block
        if current_block:
            chunks.append(self.create_chunk(
                content='\n'.join(current_block),
                chunk_type='adaptive_script',
                metadata={
                    'block_type': 'script',
                    'function': function_name
                },
                file_path=str(file_context.path),
                start_line=block_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _chunk_markup_sections(self, content: str, file_context: FileContext,
                              analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk markup/markdown content"""
        chunks = []
        lines = content.split('\n')
        
        current_section = []
        section_level = 0
        section_title = ''
        section_start = 0
        
        for i, line in enumerate(lines):
            # Detect headers
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                # Save current section
                if current_section:
                    chunks.append(self.create_chunk(
                        content='\n'.join(current_section),
                        chunk_type='adaptive_markup',
                        metadata={
                            'block_type': 'markup',
                            'section_level': section_level,
                            'section_title': section_title
                        },
                        file_path=str(file_context.path),
                        start_line=section_start,
                        end_line=i
                    ))
                
                section_level = len(header_match.group(1))
                section_title = header_match.group(2)
                current_section = [line]
                section_start = i
            else:
                current_section.append(line)
                
                # Check size limits
                if len(current_section) > 50:
                    content_str = '\n'.join(current_section)
                    if self.count_tokens(content_str) > self.max_tokens:
                        chunks.append(self.create_chunk(
                            content=content_str,
                            chunk_type='adaptive_markup',
                            metadata={
                                'block_type': 'markup',
                                'section_level': section_level,
                                'section_title': section_title
                            },
                            file_path=str(file_context.path),
                            start_line=section_start,
                            end_line=i
                        ))
                        current_section = []
                        section_start = i + 1
        
        # Add remaining section
        if current_section:
            chunks.append(self.create_chunk(
                content='\n'.join(current_section),
                chunk_type='adaptive_markup',
                metadata={
                    'block_type': 'markup',
                    'section_level': section_level,
                    'section_title': section_title
                },
                file_path=str(file_context.path),
                start_line=section_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _chunk_log_entries(self, content: str, file_context: FileContext,
                          analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk log file content"""
        chunks = []
        lines = content.split('\n')
        
        current_entry = []
        entry_start = 0
        entry_count = 0
        
        for i, line in enumerate(lines):
            # Detect log entry start (timestamp or log level)
            is_new_entry = (
                re.match(r'^\d{4}-\d{2}-\d{2}', line) or
                re.match(r'^\[\d{4}-\d{2}-\d{2}', line) or
                re.match(r'^\w+\s+\d+,?\s+\d{4}', line) or
                re.match(r'^(?:ERROR|WARN|INFO|DEBUG|TRACE)', line)
            )
            
            if is_new_entry and current_entry:
                entry_count += 1
                
                # Create chunk every N entries or when size limit reached
                if entry_count >= 50 or self.count_tokens('\n'.join(current_entry)) > self.max_tokens * 0.8:
                    chunks.append(self.create_chunk(
                        content='\n'.join(current_entry),
                        chunk_type='adaptive_log',
                        metadata={
                            'block_type': 'log',
                            'entry_count': entry_count
                        },
                        file_path=str(file_context.path),
                        start_line=entry_start,
                        end_line=i
                    ))
                    current_entry = []
                    entry_count = 0
                    entry_start = i
            
            current_entry.append(line)
        
        # Add remaining entries
        if current_entry:
            chunks.append(self.create_chunk(
                content='\n'.join(current_entry),
                chunk_type='adaptive_log',
                metadata={
                    'block_type': 'log',
                    'entry_count': entry_count + 1
                },
                file_path=str(file_context.path),
                start_line=entry_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _chunk_by_indentation(self, content: str, file_context: FileContext,
                             analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk by indentation levels"""
        chunks = []
        lines = content.split('\n')
        
        current_block = []
        current_indent = 0
        block_start = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                current_block.append(line)
                continue
            
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            
            # New top-level block
            if indent == 0 and current_block and current_indent > 0:
                # Save current block
                chunks.append(self.create_chunk(
                    content='\n'.join(current_block),
                    chunk_type='adaptive_indented',
                    metadata={
                        'block_type': 'indented',
                        'indent_level': current_indent
                    },
                    file_path=str(file_context.path),
                    start_line=block_start,
                    end_line=i
                ))
                current_block = []
                block_start = i
            
            current_block.append(line)
            current_indent = indent
            
            # Check size limits
            if len(current_block) > 50:
                content_str = '\n'.join(current_block)
                if self.count_tokens(content_str) > self.max_tokens:
                    chunks.append(self.create_chunk(
                        content=content_str,
                        chunk_type='adaptive_indented',
                        metadata={
                            'block_type': 'indented',
                            'indent_level': current_indent
                        },
                        file_path=str(file_context.path),
                        start_line=block_start,
                        end_line=i
                    ))
                    current_block = []
                    block_start = i + 1
        
        # Add remaining block
        if current_block:
            chunks.append(self.create_chunk(
                content='\n'.join(current_block),
                chunk_type='adaptive_indented',
                metadata={
                    'block_type': 'indented',
                    'indent_level': current_indent
                },
                file_path=str(file_context.path),
                start_line=block_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _chunk_by_sections(self, content: str, file_context: FileContext,
                          analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk by detected sections"""
        chunks = []
        lines = content.split('\n')
        sections = analysis['structure'].detected_sections if 'structure' in analysis else []
        
        if not sections:
            # Fallback if no sections detected
            return self._chunk_intelligent_lines(content, file_context, analysis)
        
        for start, end in sections:
            section_lines = lines[start:end]
            section_content = '\n'.join(section_lines)
            
            # Check if section fits in one chunk
            if self.count_tokens(section_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=section_content,
                    chunk_type='adaptive_section',
                    metadata={
                        'block_type': 'section',
                        'section_index': len(chunks)
                    },
                    file_path=str(file_context.path),
                    start_line=start,
                    end_line=end
                ))
            else:
                # Split large section
                sub_chunks = self._split_large_section(section_lines, start, file_context)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_section(self, lines: List[str], start_offset: int,
                            file_context: FileContext) -> List[Chunk]:
        """Split a large section into smaller chunks"""
        chunks = []
        current_chunk = []
        chunk_start = start_offset
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            if self.count_tokens('\n'.join(current_chunk)) > self.max_tokens * 0.9:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk[:-1]),
                    chunk_type='adaptive_section_part',
                    metadata={
                        'block_type': 'section_part',
                        'part_index': len(chunks)
                    },
                    file_path=str(file_context.path),
                    start_line=chunk_start,
                    end_line=start_offset + i
                ))
                current_chunk = [line]
                chunk_start = start_offset + i
        
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='adaptive_section_part',
                metadata={
                    'block_type': 'section_part',
                    'part_index': len(chunks)
                },
                file_path=str(file_context.path),
                start_line=chunk_start,
                end_line=start_offset + len(lines)
            ))
        
        return chunks
    
    def _chunk_by_paragraphs(self, content: str, file_context: FileContext,
                            analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk by paragraphs (blank line separated)"""
        chunks = []
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = []
        chunk_start_para = 0
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Check if adding paragraph exceeds token limit
            test_content = '\n\n'.join(current_chunk + [paragraph])
            
            if current_chunk and self.count_tokens(test_content) > self.max_tokens:
                # Save current chunk
                chunks.append(self.create_chunk(
                    content='\n\n'.join(current_chunk),
                    chunk_type='adaptive_paragraph',
                    metadata={
                        'block_type': 'paragraph',
                        'paragraph_count': len(current_chunk)
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = [paragraph]
                chunk_start_para = i
            else:
                current_chunk.append(paragraph)
        
        # Add remaining paragraphs
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n\n'.join(current_chunk),
                chunk_type='adaptive_paragraph',
                metadata={
                    'block_type': 'paragraph',
                    'paragraph_count': len(current_chunk)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_long_lines(self, content: str, file_context: FileContext,
                         analysis: Dict[str, Any]) -> List[Chunk]:
        """Chunk content with very long lines"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            # Check if adding line exceeds limits
            if current_chunk and (current_size + line_size > 5000 or 
                                 self.count_tokens('\n'.join(current_chunk + [line])) > self.max_tokens):
                # Save current chunk
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='adaptive_long',
                    metadata={
                        'block_type': 'long_lines',
                        'line_count': len(current_chunk)
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='adaptive_long',
                metadata={
                    'block_type': 'long_lines',
                    'line_count': len(current_chunk)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_intelligent_lines(self, content: str, file_context: FileContext,
                                analysis: Dict[str, Any]) -> List[Chunk]:
        """Intelligent line-based chunking with boundary detection"""
        chunks = []
        lines = content.split('\n')
        
        # Detect natural boundaries
        boundaries = self._detect_boundaries(lines)
        
        current_chunk = []
        current_tokens = 0
        chunk_start = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            # Check if we should start new chunk
            should_split = (
                i in boundaries or
                current_tokens + line_tokens > self.max_tokens or
                (len(current_chunk) > 30 and not line.strip())
            )
            
            if should_split and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='adaptive_intelligent',
                    metadata={
                        'block_type': 'intelligent',
                        'method': 'boundary_detection'
                    },
                    file_path=str(file_context.path),
                    start_line=chunk_start,
                    end_line=i
                ))
                current_chunk = []
                current_tokens = 0
                chunk_start = i
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='adaptive_intelligent',
                metadata={
                    'block_type': 'intelligent',
                    'method': 'boundary_detection'
                },
                file_path=str(file_context.path),
                start_line=chunk_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _detect_boundaries(self, lines: List[str]) -> Set[int]:
        """Detect natural boundaries in content"""
        boundaries = set()
        
        for i in range(len(lines)):
            # Multiple blank lines
            if i > 1 and not lines[i].strip() and not lines[i-1].strip():
                boundaries.add(i)
            
            # Separator lines
            if re.match(r'^[-=_*#]{3,}$', lines[i].strip()):
                boundaries.add(i)
            
            # Major structural changes
            if i > 0:
                prev_line = lines[i-1]
                curr_line = lines[i]
                
                # Indent level change to zero
                if prev_line.startswith((' ', '\t')) and curr_line and not curr_line[0].isspace():
                    boundaries.add(i)
                
                # Function/class declarations
                if re.search(r'^(def |class |function |public |private )', curr_line):
                    boundaries.add(i)
        
        return boundaries
    
    def _chunk_fallback(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking strategy"""
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
                    chunk_type='adaptive_fallback',
                    metadata={
                        'block_type': 'fallback',
                        'method': 'fixed_lines'
                    },
                    file_path=str(file_context.path),
                    start_line=i,
                    end_line=min(i + len(chunk_lines), len(lines))
                ))
        
        return chunks