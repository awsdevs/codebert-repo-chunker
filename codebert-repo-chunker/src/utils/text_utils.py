
import re
import string
import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import unicodedata

# Optional NLTK support
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TextStatistics:
    """Statistics for a text chunk"""
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_line_length: float = 0.0
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    
    # Character type ratios
    uppercase_ratio: float = 0.0
    lowercase_ratio: float = 0.0
    digit_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    whitespace_ratio: float = 0.0
    
    # Vocabulary
    unique_tokens: int = 0
    vocabulary_richness: float = 0.0  # Type-Token Ratio
    most_common_words: List[Tuple[str, int]] = field(default_factory=list)
    most_common_chars: List[Tuple[str, int]] = field(default_factory=list)
    
    # Complexity scores
    complexity_score: float = 0.0  # 0-100
    readability_score: float = 0.0 # 0-100 (Flesch-Kincaid derived)
    
    # Language hint
    language_hint: str = "unknown"

class TextAnalyzer:
    """Analyzes text content for statistics and metrics"""
    
    def __init__(self, enable_nlp: bool = True):
        self.enable_nlp = enable_nlp and NLTK_AVAILABLE
        if self.enable_nlp:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data: {e}")
                    self.enable_nlp = False
    
    def analyze(self, text: str) -> TextStatistics:
        """Analyze text and return statistics"""
        stats = TextStatistics()
        
        if not text:
            return stats
            
        # Basic counts
        stats.char_count = len(text)
        
        words = self.extract_words(text)
        stats.word_count = len(words)
        
        if stats.word_count > 0:
            stats.unique_tokens = len(set(w.lower() for w in words))
            stats.vocabulary_richness = stats.unique_tokens / stats.word_count
            stats.avg_word_length = sum(len(w) for w in words) / stats.word_count
        
        sentences = self.extract_sentences(text)
        stats.sentence_count = len(sentences)
        
        if stats.sentence_count > 0:
            stats.avg_sentence_length = stats.word_count / stats.sentence_count
        
        # Paragraph analysis
        paragraphs = self.extract_paragraphs(text)
        stats.paragraph_count = len(paragraphs)
        
        # Line analysis
        lines = text.splitlines()
        stats.line_count = len(lines)
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
            try:
                word_counts = Counter(w.lower() for w in words)
                stats.most_common_words = word_counts.most_common(10)
            except Exception:
                pass
        
        char_counts = Counter(c for c in text if not c.isspace())
        stats.most_common_chars = char_counts.most_common(10)
        
        # Complexity and readability - PASS STATS TO AVOID RECURSION
        stats.complexity_score = self.calculate_complexity(text, stats)
        stats.readability_score = self.calculate_readability(text)
        
        # Language hint
        stats.language_hint = self.detect_language(text)
        
        return stats
    
    def extract_words(self, text: str) -> List[str]:
        """Extract words from text"""
        if self.enable_nlp:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Fallback to regex
        return re.findall(r'\b\w+\b', text)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if self.enable_nlp:
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
    
    def calculate_complexity(self, text: str, stats: Optional[TextStatistics] = None) -> float:
        """
        Calculate text complexity score (0-100)
        
        Based on:
        - Average sentence length
        - Average word length
        - Vocabulary richness
        - Nesting depth (for code)
        """
        # Use provided stats if available to avoid recursion
        if stats:
            avg_sent_len = stats.avg_sentence_length
            avg_word_len = stats.avg_word_length
            vocab_richness = stats.vocabulary_richness
        else:
             words = self.extract_words(text)
             sentences = self.extract_sentences(text)
             avg_sent_len = len(words) / max(1, len(sentences))
             avg_word_len = sum(len(w) for w in words) / max(1, len(words))
             vocab_size = len(set(w.lower() for w in words))
             vocab_richness = vocab_size / max(1, len(words))
        
        complexity = 0.0
        
        # Sentence length factor (longer = more complex)
        if avg_sent_len > 0:
            complexity += min(avg_sent_len / 30 * 25, 25)
        
        # Word length factor
        if avg_word_len > 0:
            complexity += min(avg_word_len / 10 * 25, 25)
        
        # Vocabulary richness factor
        complexity += vocab_richness * 25
        
        # Code complexity factors
        if self.detect_code_content(text):
            # Nesting depth
            max_nesting = self.calculate_max_nesting(text)
            complexity += min(max_nesting / 5 * 25, 25)
        else:
            # Punctuation complexity for prose
            punc_ratio = 0.0
            if len(text) > 0:
                punc_ratio = sum(1 for c in text if c in string.punctuation) / len(text)
            complexity += min(punc_ratio * 100, 25)
            
        return min(100.0, complexity)
    
    def calculate_readability(self, text: str) -> float:
        """
        Calculate readability score (0-100, higher is easier)
        Approximate Flesch Reading Ease
        """
        # Simple approximation without full syllable counting
        if not text:
            return 100.0
            
        words = self.extract_words(text)
        sentences = self.extract_sentences(text)
        
        if not words or not sentences:
            return 100.0
            
        # Avg words per sentence
        asl = len(words) / len(sentences)
        
        # Avg many-char words (proxy for syllables)
        complex_words = sum(1 for w in words if len(w) > 6)
        pcw = (complex_words / len(words)) * 100
        
        # Simplified formula
        score = 100 - (asl * 0.5) - (pcw * 0.5)
        return max(0.0, min(100.0, score))
    
    def detect_code_content(self, text: str) -> bool:
        """Detect if text looks like code"""
        # Heuristics
        code_indicators = [
            '{', '}', 'if (', 'for (', 'def ', 'class ', 'import ', 
            'return ', ';', '=>', 'function', 'var ', 'const '
        ]
        
        count = sum(1 for ind in code_indicators if ind in text)
        return count >= 2
    
    def calculate_max_nesting(self, text: str) -> int:
        """Calculate maximum nesting level (indentation or braces)"""
        max_depth = 0
        current_depth = 0
        
        # Check braces
        for char in text:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        # Check indentation (python style)
        lines = text.splitlines()
        max_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent)
            
        # Approximation: 4 spaces per level
        indent_depth = max_indent // 4
        
        return max(max_depth, indent_depth)

    def detect_language(self, text: str) -> str:
        """Simple language detection based on keywords"""
        scores = {
            'python': 0,
            'javascript': 0,
            'java': 0,
            'html': 0,
            'css': 0,
            'sql': 0
        }
        
        keywords = {
            'python': ['def ', 'import ', 'from ', 'class ', 'if __name__', 'print('],
            'javascript': ['function', 'const ', 'var ', 'let ', '=>', 'import ', 'console.log'],
            'java': ['public class', 'private ', 'protected ', 'void ', 'String ', 'System.out'],
            'html': ['<html>', '<div>', '<body>', '<script>', '<style>'],
            'css': ['body {', 'div {', 'margin:', 'padding:', 'color:'],
            'sql': ['SELECT ', 'FROM ', 'WHERE ', 'INSERT INTO', 'UPDATE ', 'DELETE ']
        }
        
        for lang, kws in keywords.items():
            for kw in kws:
                if kw in text:
                    scores[lang] += 1
        
        # Get max score
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 0:
            return best_lang
            
        return "unknown"
