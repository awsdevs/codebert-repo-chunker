"""
Text processing and analysis utilities for code and documentation
Provides text manipulation, analysis, and natural language processing
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set, Iterator
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum
import logging
import hashlib
from functools import lru_cache
import difflib
import textwrap
from itertools import groupby

# Optional NLP dependencies
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextComplexity(Enum):
    """Text complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class TextType(Enum):
    """Types of text content"""
    CODE = "code"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    PROSE = "prose"
    MIXED = "mixed"

@dataclass
class TextStatistics:
    """Statistics about text content"""
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    avg_line_length: float = 0.0
    
    unique_words: int = 0
    vocabulary_richness: float = 0.0  # unique_words / word_count
    
    uppercase_ratio: float = 0.0
    lowercase_ratio: float = 0.0
    digit_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    whitespace_ratio: float = 0.0
    
    complexity_score: float = 0.0
    readability_score: float = 0.0
    
    most_common_words: List[Tuple[str, int]] = field(default_factory=list)
    most_common_chars: List[Tuple[str, int]] = field(default_factory=list)
    
    language_hint: Optional[str] = None
    encoding_hint: Optional[str] = None

@dataclass
class TextSegment:
    """Represents a segment of text with metadata"""
    content: str
    start_line: int
    end_line: int
    segment_type: TextType
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TextAnalyzer:
    """Comprehensive text analysis utility"""
    
    # Common programming language keywords for detection
    LANGUAGE_KEYWORDS = {
        'python': {'def', 'class', 'import', 'from', 'if', 'elif', 'else', 'for', 'while', 'return', 'yield', 'lambda', 'async', 'await'},
        'java': {'public', 'private', 'class', 'interface', 'extends', 'implements', 'static', 'void', 'final', 'package', 'import'},
        'javascript': {'function', 'var', 'let', 'const', 'class', 'export', 'import', 'async', 'await', 'return', 'console'},
        'c++': {'class', 'namespace', 'template', 'virtual', 'public', 'private', 'protected', 'include', 'using', 'std'},
        'go': {'func', 'package', 'import', 'type', 'struct', 'interface', 'defer', 'goroutine', 'channel'},
        'rust': {'fn', 'let', 'mut', 'impl', 'trait', 'pub', 'mod', 'use', 'match', 'enum'},
        'ruby': {'def', 'class', 'module', 'require', 'include', 'attr_accessor', 'attr_reader', 'end'},
    }
    
    # Common stop words (if NLTK not available)
    STOP_WORDS = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
        'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'to', 'of',
        'in', 'for', 'with', 'by', 'from', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'this', 'that', 'these',
        'those', 'and', 'but', 'or', 'so', 'than', 'too', 'very', 'just', 'not'
    }
    
    def __init__(self, 
                 enable_nlp: bool = False,
                 nlp_model: str = 'en_core_web_sm'):
        """
        Initialize text analyzer
        
        Args:
            enable_nlp: Enable advanced NLP features
            nlp_model: SpaCy model to use
        """
        self.enable_nlp = enable_nlp
        
        # Initialize NLP components if available
        if enable_nlp and NLTK_AVAILABLE:
            self._init_nltk()
        
        if enable_nlp and SPACY_AVAILABLE:
            self._init_spacy(nlp_model)
        
        # Initialize text processors
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        
        # Cache for expensive computations
        self._cache = {}
    
    def _init_nltk(self):
        """Initialize NLTK resources"""
        try:
            # Download required NLTK data
            required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for data in required_data:
                try:
                    nltk.data.find(f'tokenizers/{data}')
                except LookupError:
                    nltk.download(data, quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK: {e}")
            self.stop_words = self.STOP_WORDS
    
    def _init_spacy(self, model_name: str):
        """Initialize SpaCy model"""
        try:
            self.nlp = spacy.load(model_name)
        except Exception as e:
            logger.warning(f"Failed to load SpaCy model {model_name}: {e}")
            self.nlp = None
    
    def analyze(self, text: str) -> TextStatistics:
        """
        Perform comprehensive text analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            TextStatistics object with analysis results
        """
        stats = TextStatistics()
        
        if not text:
            return stats
        
        # Basic counts
        stats.char_count = len(text)
        stats.line_count = text.count('\n') + 1
        
        # Word analysis
        words = self.extract_words(text)
        stats.word_count = len(words)
        stats.unique_words = len(set(words))
        
        if stats.word_count > 0:
            stats.vocabulary_richness = stats.unique_words / stats.word_count
            stats.avg_word_length = sum(len(w) for w in words) / stats.word_count
        
        # Sentence analysis
        sentences = self.extract_sentences(text)
        stats.sentence_count = len(sentences)
        
        if stats.sentence_count > 0:
            stats.avg_sentence_length = stats.word_count / stats.sentence_count
        
        # Paragraph analysis
        paragraphs = self.extract_paragraphs(text)
        stats.paragraph_count = len(paragraphs)
        
        # Line analysis
        lines = text.splitlines()
        if lines:
            stats.avg_line_length = sum(len(line) for line in lines) / len(lines)
        
        # Character type ratios
        if stats.char_count > 0:
            stats.uppercase_ratio = sum(1 for c in text if c.isupper()) / stats.char_count
            stats.lowercase_ratio = sum(1 for c in text if c.islower()) / stats.char_count
            stats.digit_ratio = sum(1 for c in text if c.isdigit()) / stats.char_count
            stats.punctuation_ratio = sum(1 for c in text if c in string.punctuation) / stats.char_count
            stats.whitespace_ratio = sum(1 for c in text if c.isspace()) / stats.char_count
        
        # Most common analysis
        if words:
            word_counts = Counter(w.lower() for w in words)
            stats.most_common_words = word_counts.most_common(10)
        
        char_counts = Counter(c for c in text if not c.isspace())
        stats.most_common_chars = char_counts.most_common(10)
        
        # Complexity and readability
        stats.complexity_score = self.calculate_complexity(text)
        stats.readability_score = self.calculate_readability(text)
        
        # Language hint
        stats.language_hint = self.detect_language(text)
        
        return stats
    
    def extract_words(self, text: str) -> List[str]:
        """Extract words from text"""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Fallback to regex
        return re.findall(r'\b\w+\b', text)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback to simple splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def calculate_complexity(self, text: str) -> float:
        """
        Calculate text complexity score (0-100)
        
        Based on:
        - Average sentence length
        - Average word length
        - Vocabulary richness
        - Nesting depth (for code)
        """
        stats = self.analyze(text)
        
        complexity = 0.0
        
        # Sentence length factor (longer = more complex)
        if stats.avg_sentence_length > 0:
            complexity += min(stats.avg_sentence_length / 30 * 25, 25)
        
        # Word length factor
        if stats.avg_word_length > 0:
            complexity += min(stats.avg_word_length / 10 * 25, 25)
        
        # Vocabulary richness factor
        complexity += stats.vocabulary_richness * 25
        
        # Code complexity factors
        if self.detect_code_content(text):
            # Nesting depth
            max_nesting = self.calculate_max_nesting(text)
            complexity += min(max_nesting / 5 * 25, 25)
        else:
            # Punctuation complexity for prose
            complexity += stats.punctuation_ratio * 100 * 0.25
        
        return min(complexity, 100)
    
    def calculate_readability(self, text: str) -> float:
        """
        Calculate readability score (0-100)
        Using simplified Flesch Reading Ease formula
        """
        stats = self.analyze(text)
        
        if stats.word_count == 0 or stats.sentence_count == 0:
            return 0.0
        
        # Flesch Reading Ease formula (simplified)
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        # We approximate syllables as word_length / 3
        
        avg_syllables_per_word = stats.avg_word_length / 3
        
        score = 206.835 - (1.015 * stats.avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-100
        return max(0, min(100, score))
    
    def calculate_max_nesting(self, text: str) -> int:
        """Calculate maximum nesting depth in code"""
        max_depth = 0
        current_depth = 0
        
        for char in text:
            if char in '{[(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in '}])':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect programming language from text"""
        # Count language-specific keywords
        scores = {}
        
        words = set(self.extract_words(text.lower()))
        
        for language, keywords in self.LANGUAGE_KEYWORDS.items():
            score = len(words & keywords)
            if score > 0:
                scores[language] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return None
    
    def detect_code_content(self, text: str) -> bool:
        """Detect if text contains code"""
        code_indicators = [
            r'^\s*(?:def|class|function|public|private)\s+',  # Function/class definitions
            r'(?:import|include|require|using)\s+',  # Import statements
            r'(?:if|for|while|switch)\s*\(',  # Control structures
            r'[{};]$',  # Code block markers
            r'=>|->|::|<<|>>',  # Special operators
            r'^\s*(?://|#|/\*)',  # Comments
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        return False
    
    def segment_text(self,
                    text: str,
                    segment_size: int = 1000,
                    overlap: int = 100,
                    preserve_sentences: bool = True) -> List[TextSegment]:
        """
        Segment text into chunks
        
        Args:
            text: Text to segment
            segment_size: Target segment size in characters
            overlap: Overlap between segments
            preserve_sentences: Try to preserve sentence boundaries
            
        Returns:
            List of text segments
        """
        segments = []
        
        if preserve_sentences:
            sentences = self.extract_sentences(text)
            current_segment = []
            current_size = 0
            start_line = 1
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                if current_size + sentence_size > segment_size and current_segment:
                    # Create segment
                    content = ' '.join(current_segment)
                    end_line = start_line + content.count('\n')
                    
                    segments.append(TextSegment(
                        content=content,
                        start_line=start_line,
                        end_line=end_line,
                        segment_type=self.detect_text_type(content)
                    ))
                    
                    # Prepare next segment with overlap
                    if overlap > 0 and current_segment:
                        overlap_sentences = []
                        overlap_size = 0
                        
                        for sent in reversed(current_segment):
                            overlap_size += len(sent)
                            overlap_sentences.insert(0, sent)
                            if overlap_size >= overlap:
                                break
                        
                        current_segment = overlap_sentences
                        current_size = overlap_size
                        start_line = end_line + 1
                    else:
                        current_segment = []
                        current_size = 0
                        start_line = end_line + 1
                
                current_segment.append(sentence)
                current_size += sentence_size
            
            # Add remaining segment
            if current_segment:
                content = ' '.join(current_segment)
                segments.append(TextSegment(
                    content=content,
                    start_line=start_line,
                    end_line=start_line + content.count('\n'),
                    segment_type=self.detect_text_type(content)
                ))
        
        else:
            # Simple character-based segmentation
            for i in range(0, len(text), segment_size - overlap):
                segment_text = text[i:i + segment_size]
                
                segments.append(TextSegment(
                    content=segment_text,
                    start_line=text[:i].count('\n') + 1,
                    end_line=text[:i + len(segment_text)].count('\n') + 1,
                    segment_type=self.detect_text_type(segment_text)
                ))
        
        return segments
    
    def detect_text_type(self, text: str) -> TextType:
        """Detect the type of text content"""
        # Check for code
        if self.detect_code_content(text):
            # Check if it's a comment or docstring
            if text.strip().startswith(('"""', "'''", '/*', '//', '#')):
                if '"""' in text or "'''" in text:
                    return TextType.DOCSTRING
                return TextType.COMMENT
            return TextType.CODE
        
        # Check for configuration
        if any(pattern in text for pattern in ['=', ':', 'true', 'false', 'null']):
            if text.count('=') > 3 or text.count(':') > 3:
                return TextType.CONFIGURATION
        
        # Check for documentation markers
        doc_markers = ['##', '###', '====', '----', '@param', '@return', '@throws']
        if any(marker in text for marker in doc_markers):
            return TextType.DOCUMENTATION
        
        # Default to prose
        return TextType.PROSE
    
    def clean_text(self,
                  text: str,
                  remove_comments: bool = False,
                  remove_empty_lines: bool = False,
                  normalize_whitespace: bool = True,
                  strip_lines: bool = True) -> str:
        """
        Clean text with various options
        
        Args:
            text: Text to clean
            remove_comments: Remove code comments
            remove_empty_lines: Remove empty lines
            normalize_whitespace: Normalize whitespace
            strip_lines: Strip whitespace from lines
            
        Returns:
            Cleaned text
        """
        lines = text.splitlines()
        
        # Process lines
        cleaned_lines = []
        for line in lines:
            # Remove comments if requested
            if remove_comments:
                # Remove single-line comments
                line = re.sub(r'//.*$', '', line)
                line = re.sub(r'#.*$', '', line)
            
            # Strip lines if requested
            if strip_lines:
                line = line.rstrip()
            
            # Skip empty lines if requested
            if remove_empty_lines and not line.strip():
                continue
            
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        # Normalize whitespace if requested
        if normalize_whitespace:
            # Replace multiple spaces with single space
            result = re.sub(r' +', ' ', result)
            # Replace multiple newlines with double newline
            result = re.sub(r'\n\n+', '\n\n', result)
        
        return result
    
    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from markdown or documentation
        
        Returns:
            List of (language, code) tuples
        """
        code_blocks = []
        
        # Markdown code blocks
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for language, code in matches:
            language = language or 'unknown'
            code_blocks.append((language, code.strip()))
        
        # Indented code blocks (4 spaces or 1 tab)
        lines = text.splitlines()
        current_block = []
        in_code_block = False
        
        for line in lines:
            if line.startswith(('    ', '\t')) and line.strip():
                in_code_block = True
                current_block.append(line[4:] if line.startswith('    ') else line[1:])
            elif in_code_block and line.strip():
                # End of code block
                if current_block:
                    code_blocks.append(('unknown', '\n'.join(current_block)))
                current_block = []
                in_code_block = False
        
        # Add final block if any
        if current_block:
            code_blocks.append(('unknown', '\n'.join(current_block)))
        
        return code_blocks
    
    def tokenize(self,
                text: str,
                lowercase: bool = True,
                remove_stopwords: bool = True,
                stem: bool = False,
                lemmatize: bool = False) -> List[str]:
        """
        Tokenize text with various options
        
        Args:
            text: Text to tokenize
            lowercase: Convert to lowercase
            remove_stopwords: Remove stop words
            stem: Apply stemming
            lemmatize: Apply lemmatization
            
        Returns:
            List of tokens
        """
        # Extract words
        tokens = self.extract_words(text)
        
        # Lowercase if requested
        if lowercase:
            tokens = [t.lower() for t in tokens]
        
        # Remove stopwords if requested
        if remove_stopwords:
            stop_words = self.stop_words if hasattr(self, 'stop_words') else self.STOP_WORDS
            tokens = [t for t in tokens if t not in stop_words]
        
        # Apply stemming if requested
        if stem and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Apply lemmatization if requested
        if lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def calculate_similarity(self,
                           text1: str,
                           text2: str,
                           method: str = 'jaccard') -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method (jaccard, cosine, levenshtein)
            
        Returns:
            Similarity score (0-1)
        """
        if method == 'jaccard':
            # Jaccard similarity based on tokens
            tokens1 = set(self.tokenize(text1))
            tokens2 = set(self.tokenize(text2))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            
            return len(intersection) / len(union)
        
        elif method == 'cosine':
            # Cosine similarity based on token frequency
            tokens1 = self.tokenize(text1)
            tokens2 = self.tokenize(text2)
            
            # Create frequency vectors
            all_tokens = set(tokens1 + tokens2)
            
            vec1 = [tokens1.count(t) for t in all_tokens]
            vec2 = [tokens2.count(t) for t in all_tokens]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a ** 2 for a in vec1) ** 0.5
            norm2 = sum(b ** 2 for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif method == 'levenshtein':
            # Normalized Levenshtein distance
            distance = self._levenshtein_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            
            if max_len == 0:
                return 1.0
            
            return 1 - (distance / max_len)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            
            previous_row = current_row
        
        return previous_row[-1]
    
    def wrap_text(self,
                 text: str,
                 width: int = 80,
                 indent: str = '',
                 subsequent_indent: Optional[str] = None) -> str:
        """
        Wrap text to specified width
        
        Args:
            text: Text to wrap
            width: Line width
            indent: Initial line indent
            subsequent_indent: Subsequent lines indent
            
        Returns:
            Wrapped text
        """
        if subsequent_indent is None:
            subsequent_indent = indent
        
        wrapped = textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False
        )
        
        return wrapped
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'https?://(?:[-\w.])+(?::\d+)?(?:[/]\S*)?'
        return re.findall(url_pattern, text)
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        # Extract integers and floats
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        numbers = []
        for match in matches:
            try:
                if '.' in match:
                    numbers.append(float(match))
                else:
                    numbers.append(float(match))
            except ValueError:
                pass
        
        return numbers
    
    def highlight_text(self,
                      text: str,
                      terms: List[str],
                      highlight_start: str = '**',
                      highlight_end: str = '**',
                      case_sensitive: bool = False) -> str:
        """
        Highlight terms in text
        
        Args:
            text: Text to highlight in
            terms: Terms to highlight
            highlight_start: Start marker for highlight
            highlight_end: End marker for highlight
            case_sensitive: Case sensitive matching
            
        Returns:
            Text with highlighted terms
        """
        result = text
        
        for term in terms:
            if case_sensitive:
                pattern = re.escape(term)
            else:
                pattern = re.escape(term)
                pattern = f'(?i){pattern}'
            
            replacement = f'{highlight_start}{term}{highlight_end}'
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def get_text_hash(self, text: str, algorithm: str = 'sha256') -> str:
        """Get hash of text"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text"""
        # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
        return unicodedata.normalize('NFC', text)
    
    def remove_accents(self, text: str) -> str:
        """Remove accents from text"""
        # Normalize to NFD and filter out combining characters
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

# Convenience functions
text_analyzer = TextAnalyzer()

def analyze_text(text: str) -> TextStatistics:
    """Analyze text and return statistics"""
    return text_analyzer.analyze(text)

def clean_text(text: str, **kwargs) -> str:
    """Clean text with options"""
    return text_analyzer.clean_text(text, **kwargs)

def tokenize(text: str, **kwargs) -> List[str]:
    """Tokenize text"""
    return text_analyzer.tokenize(text, **kwargs)

def calculate_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
    """Calculate text similarity"""
    return text_analyzer.calculate_similarity(text1, text2, method)

def segment_text(text: str, segment_size: int = 1000, **kwargs) -> List[TextSegment]:
    """Segment text into chunks"""
    return text_analyzer.segment_text(text, segment_size, **kwargs)