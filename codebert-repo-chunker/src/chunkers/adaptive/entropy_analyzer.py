"""
Entropy-based analysis for intelligent content chunking
Uses information theory to detect patterns, boundaries, and structure in unknown files
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter, deque
from dataclasses import dataclass, field
from src.utils.logger import get_logger
from scipy import signal
from scipy.stats import entropy as scipy_entropy
import warnings

# Suppress scipy warnings in production
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = get_logger(__name__)

@dataclass
class EntropyMetrics:
    """Container for entropy analysis metrics"""
    shannon_entropy: float
    normalized_entropy: float
    relative_entropy: Optional[float]
    joint_entropy: Optional[float]
    conditional_entropy: Optional[float]
    mutual_information: Optional[float]
    
@dataclass
class EntropyProfile:
    """Complete entropy profile of content"""
    global_entropy: float
    local_entropies: List[float]
    entropy_gradient: List[float]
    peaks: List[int]
    valleys: List[int]
    boundaries: List[int]
    segments: List[Tuple[int, int, float]]  # (start, end, avg_entropy)
    statistics: Dict[str, float]

@dataclass 
class WindowMetrics:
    """Metrics for a sliding window analysis"""
    position: int
    size: int
    entropy: float
    char_distribution: Dict[str, float]
    token_diversity: float
    pattern_score: float

class EntropyAnalyzer:
    """Advanced entropy analysis for content understanding"""
    
    def __init__(self, 
                 window_size: int = 50,
                 boundary_threshold: float = 0.3,
                 smoothing_factor: float = 0.1):
        """
        Initialize entropy analyzer
        
        Args:
            window_size: Size of sliding window for local entropy
            boundary_threshold: Threshold for detecting boundaries
            smoothing_factor: Factor for smoothing entropy curves
        """
        self.window_size = window_size
        self.boundary_threshold = boundary_threshold
        self.smoothing_factor = smoothing_factor
        
        # Cache for repeated calculations
        self.cache = {}
        self.cache_size_limit = 1000
        
        # Pre-computed logarithms for efficiency
        self.log_cache = {}
        
    def analyze_content(self, content: str) -> EntropyProfile:
        """
        Perform comprehensive entropy analysis on content
        
        Args:
            content: Text content to analyze
            
        Returns:
            Complete entropy profile
        """
        if not content:
            return self._empty_profile()
        
        # Calculate global entropy
        global_entropy = self.calculate_shannon_entropy(content)
        
        # Calculate local entropies using sliding window
        local_entropies = self.calculate_local_entropies(content)
        
        # Calculate entropy gradient
        entropy_gradient = self.calculate_gradient(local_entropies)
        
        # Detect peaks and valleys
        peaks = self.detect_peaks(local_entropies)
        valleys = self.detect_valleys(local_entropies)
        
        # Detect boundaries
        boundaries = self.detect_boundaries(entropy_gradient, local_entropies)
        
        # Segment content based on entropy
        segments = self.segment_by_entropy(local_entropies, boundaries)
        
        # Calculate statistics
        statistics = self.calculate_statistics(local_entropies, entropy_gradient)
        
        return EntropyProfile(
            global_entropy=global_entropy,
            local_entropies=local_entropies,
            entropy_gradient=entropy_gradient,
            peaks=peaks,
            valleys=valleys,
            boundaries=boundaries,
            segments=segments,
            statistics=statistics
        )
    
    def calculate_shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text
        
        Shannon entropy H(X) = -Σ p(x) * log2(p(x))
        
        Args:
            text: Input text
            
        Returns:
            Shannon entropy value
        """
        if not text:
            return 0.0
        
        # Check cache
        cache_key = hash(text[:1000]) if len(text) > 1000 else hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Calculate character frequency
        freq_map = Counter(text)
        length = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in freq_map.values():
            probability = count / length
            if probability > 0:
                # Use cached logarithm if available
                if probability not in self.log_cache:
                    self.log_cache[probability] = math.log2(probability)
                entropy -= probability * self.log_cache[probability]
        
        # Cache result
        self._update_cache(cache_key, entropy)
        
        return entropy
    
    def calculate_normalized_entropy(self, text: str) -> float:
        """
        Calculate normalized entropy (0 to 1)
        
        Args:
            text: Input text
            
        Returns:
            Normalized entropy value
        """
        if not text:
            return 0.0
        
        entropy = self.calculate_shannon_entropy(text)
        
        # Maximum possible entropy for alphabet size
        alphabet_size = len(set(text))
        if alphabet_size <= 1:
            return 0.0
        
        max_entropy = math.log2(alphabet_size)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def calculate_relative_entropy(self, text1: str, text2: str) -> float:
        """
        Calculate relative entropy (KL divergence) between two texts
        
        KL(P||Q) = Σ p(x) * log2(p(x) / q(x))
        
        Args:
            text1: First text (P)
            text2: Second text (Q)
            
        Returns:
            Relative entropy value
        """
        if not text1 or not text2:
            return 0.0
        
        # Get character distributions
        dist1 = self._get_distribution(text1)
        dist2 = self._get_distribution(text2)
        
        # Calculate KL divergence
        kl_divergence = 0.0
        for char, p in dist1.items():
            q = dist2.get(char, 1e-10)  # Small value to avoid division by zero
            if p > 0 and q > 0:
                kl_divergence += p * math.log2(p / q)
        
        return kl_divergence
    
    def calculate_joint_entropy(self, text1: str, text2: str) -> float:
        """
        Calculate joint entropy of two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Joint entropy value
        """
        if not text1 or not text2:
            return 0.0
        
        # Combine texts for joint distribution
        combined = text1 + text2
        
        # Calculate joint entropy
        return self.calculate_shannon_entropy(combined)
    
    def calculate_conditional_entropy(self, text: str, context_size: int = 1) -> float:
        """
        Calculate conditional entropy H(X|Y)
        
        Args:
            text: Input text
            context_size: Size of context window
            
        Returns:
            Conditional entropy value
        """
        if len(text) <= context_size:
            return 0.0
        
        # Build context-based probability model
        context_map = {}
        
        for i in range(len(text) - context_size):
            context = text[i:i + context_size]
            next_char = text[i + context_size]
            
            if context not in context_map:
                context_map[context] = []
            context_map[context].append(next_char)
        
        # Calculate conditional entropy
        conditional_entropy = 0.0
        total_contexts = sum(len(chars) for chars in context_map.values())
        
        for context, chars in context_map.items():
            context_prob = len(chars) / total_contexts
            char_entropy = self.calculate_shannon_entropy(''.join(chars))
            conditional_entropy += context_prob * char_entropy
        
        return conditional_entropy
    
    def calculate_mutual_information(self, text1: str, text2: str) -> float:
        """
        Calculate mutual information between two texts
        
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Mutual information value
        """
        if not text1 or not text2:
            return 0.0
        
        h_x = self.calculate_shannon_entropy(text1)
        h_y = self.calculate_shannon_entropy(text2)
        h_xy = self.calculate_joint_entropy(text1, text2)
        
        return h_x + h_y - h_xy
    
    def calculate_local_entropies(self, text: str) -> List[float]:
        """
        Calculate entropy for sliding windows
        
        Args:
            text: Input text
            
        Returns:
            List of local entropy values
        """
        if not text:
            return []
        
        lines = text.split('\n')
        entropies = []
        
        # Use sliding window over lines
        window = deque(maxlen=self.window_size)
        
        for i, line in enumerate(lines):
            window.append(line)
            
            if len(window) >= self.window_size // 2:  # Start calculating after half window
                window_text = '\n'.join(window)
                entropy = self.calculate_normalized_entropy(window_text)
                entropies.append(entropy)
        
        # Smooth entropy curve
        if len(entropies) > 5:
            entropies = self._smooth_curve(entropies)
        
        return entropies
    
    def calculate_gradient(self, values: List[float]) -> List[float]:
        """
        Calculate gradient of entropy values
        
        Args:
            values: List of entropy values
            
        Returns:
            List of gradient values
        """
        if len(values) < 2:
            return [0.0] * len(values)
        
        gradient = []
        
        # Forward difference for first element
        gradient.append(values[1] - values[0])
        
        # Central difference for middle elements
        for i in range(1, len(values) - 1):
            grad = (values[i + 1] - values[i - 1]) / 2
            gradient.append(grad)
        
        # Backward difference for last element
        gradient.append(values[-1] - values[-2])
        
        return gradient
    
    def detect_peaks(self, values: List[float], prominence: float = 0.1) -> List[int]:
        """
        Detect peaks in entropy profile
        
        Args:
            values: List of entropy values
            prominence: Minimum prominence of peaks
            
        Returns:
            Indices of detected peaks
        """
        if len(values) < 3:
            return []
        
        try:
            # Use scipy for robust peak detection
            peaks, properties = signal.find_peaks(values, prominence=prominence)
            return peaks.tolist()
        except Exception as e:
            logger.warning(f"Peak detection failed, using fallback: {e}")
            return self._detect_peaks_fallback(values, prominence)
    
    def _detect_peaks_fallback(self, values: List[float], prominence: float) -> List[int]:
        """Fallback peak detection without scipy"""
        peaks = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                # Check prominence
                left_min = min(values[max(0, i - 5):i])
                right_min = min(values[i + 1:min(len(values), i + 6)])
                
                if values[i] - max(left_min, right_min) >= prominence:
                    peaks.append(i)
        
        return peaks
    
    def detect_valleys(self, values: List[float], prominence: float = 0.1) -> List[int]:
        """
        Detect valleys in entropy profile
        
        Args:
            values: List of entropy values
            prominence: Minimum prominence of valleys
            
        Returns:
            Indices of detected valleys
        """
        if len(values) < 3:
            return []
        
        # Invert values to find valleys as peaks
        inverted = [-v for v in values]
        
        try:
            valleys, properties = signal.find_peaks(inverted, prominence=prominence)
            return valleys.tolist()
        except Exception as e:
            logger.warning(f"Valley detection failed, using fallback: {e}")
            return self._detect_valleys_fallback(values, prominence)
    
    def _detect_valleys_fallback(self, values: List[float], prominence: float) -> List[int]:
        """Fallback valley detection without scipy"""
        valleys = []
        
        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                # Check prominence
                left_max = max(values[max(0, i - 5):i])
                right_max = max(values[i + 1:min(len(values), i + 6)])
                
                if min(left_max, right_max) - values[i] >= prominence:
                    valleys.append(i)
        
        return valleys
    
    def detect_boundaries(self, gradient: List[float], 
                         entropies: List[float]) -> List[int]:
        """
        Detect content boundaries based on entropy changes
        
        Args:
            gradient: Entropy gradient values
            entropies: Local entropy values
            
        Returns:
            Indices of detected boundaries
        """
        boundaries = set()
        
        if len(gradient) < 3:
            return []
        
        # Method 1: Large gradient changes
        abs_gradient = [abs(g) for g in gradient]
        if abs_gradient:
            threshold = np.percentile(abs_gradient, 75)
            
            for i, grad_val in enumerate(abs_gradient):
                if grad_val > threshold:
                    boundaries.add(i)
        
        # Method 2: Entropy level transitions
        if entropies:
            mean_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)
            
            for i in range(1, len(entropies)):
                # Crossing mean by more than 1 std
                if abs(entropies[i] - mean_entropy) > std_entropy:
                    if abs(entropies[i - 1] - mean_entropy) <= std_entropy:
                        boundaries.add(i)
        
        # Method 3: Pattern changes (second derivative)
        if len(gradient) > 2:
            second_derivative = self.calculate_gradient(gradient)
            
            for i, val in enumerate(second_derivative):
                if abs(val) > self.boundary_threshold:
                    boundaries.add(i)
        
        return sorted(list(boundaries))
    
    def segment_by_entropy(self, entropies: List[float], 
                          boundaries: List[int]) -> List[Tuple[int, int, float]]:
        """
        Segment content based on entropy boundaries
        
        Args:
            entropies: Local entropy values
            boundaries: Boundary indices
            
        Returns:
            List of segments (start, end, avg_entropy)
        """
        if not entropies:
            return []
        
        segments = []
        
        # Add start and end boundaries
        all_boundaries = [0] + boundaries + [len(entropies)]
        all_boundaries = sorted(list(set(all_boundaries)))
        
        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]
            
            if start < end and end <= len(entropies):
                segment_entropies = entropies[start:end]
                avg_entropy = np.mean(segment_entropies) if segment_entropies else 0.0
                
                segments.append((start, end, avg_entropy))
        
        return segments
    
    def calculate_statistics(self, entropies: List[float], 
                           gradient: List[float]) -> Dict[str, float]:
        """
        Calculate statistical metrics
        
        Args:
            entropies: Local entropy values
            gradient: Gradient values
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        if entropies:
            stats['mean_entropy'] = np.mean(entropies)
            stats['std_entropy'] = np.std(entropies)
            stats['min_entropy'] = np.min(entropies)
            stats['max_entropy'] = np.max(entropies)
            stats['entropy_range'] = stats['max_entropy'] - stats['min_entropy']
            stats['entropy_variance'] = np.var(entropies)
            
            # Quantiles
            stats['q25_entropy'] = np.percentile(entropies, 25)
            stats['q50_entropy'] = np.percentile(entropies, 50)
            stats['q75_entropy'] = np.percentile(entropies, 75)
        
        if gradient:
            stats['mean_gradient'] = np.mean(np.abs(gradient))
            stats['max_gradient'] = np.max(np.abs(gradient))
            stats['gradient_variance'] = np.var(gradient)
        
        return stats
    
    def analyze_sliding_window(self, text: str, 
                              window_size: Optional[int] = None) -> List[WindowMetrics]:
        """
        Perform sliding window analysis with detailed metrics
        
        Args:
            text: Input text
            window_size: Custom window size
            
        Returns:
            List of window metrics
        """
        if not text:
            return []
        
        window_size = window_size or self.window_size
        lines = text.split('\n')
        metrics = []
        
        for i in range(len(lines) - window_size + 1):
            window_lines = lines[i:i + window_size]
            window_text = '\n'.join(window_lines)
            
            # Calculate metrics
            entropy = self.calculate_normalized_entropy(window_text)
            char_dist = self._get_distribution(window_text)
            token_diversity = self._calculate_token_diversity(window_text)
            pattern_score = self._calculate_pattern_score(window_text)
            
            metrics.append(WindowMetrics(
                position=i,
                size=window_size,
                entropy=entropy,
                char_distribution=char_dist,
                token_diversity=token_diversity,
                pattern_score=pattern_score
            ))
        
        return metrics
    
    def detect_compression(self, text: str) -> Tuple[bool, float]:
        """
        Detect if content might be compressed or encrypted
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_compressed, confidence)
        """
        if not text:
            return False, 0.0
        
        # High entropy suggests compression/encryption
        entropy = self.calculate_normalized_entropy(text)
        
        # Check for common patterns
        char_dist = self._get_distribution(text)
        uniformity = self._calculate_uniformity(char_dist)
        
        # Binary content check
        binary_chars = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
        binary_ratio = binary_chars / len(text)
        
        # Decision logic
        is_compressed = (
            entropy > 0.9 or
            uniformity > 0.8 or
            binary_ratio > 0.3
        )
        
        # Confidence calculation
        confidence = min(
            (entropy * 0.4 + uniformity * 0.3 + binary_ratio * 0.3),
            1.0
        )
        
        return is_compressed, confidence
    
    def find_repetitive_patterns(self, text: str, 
                                min_length: int = 3) -> Dict[str, int]:
        """
        Find repetitive patterns in text
        
        Args:
            text: Input text
            min_length: Minimum pattern length
            
        Returns:
            Dictionary of patterns and their counts
        """
        patterns = {}
        
        if len(text) < min_length:
            return patterns
        
        # Use suffix array approach for efficiency
        for length in range(min_length, min(len(text) // 2, 50)):
            for i in range(len(text) - length + 1):
                pattern = text[i:i + length]
                
                # Skip low-entropy patterns
                if self.calculate_normalized_entropy(pattern) < 0.2:
                    continue
                
                count = text.count(pattern)
                if count >= 2:
                    patterns[pattern] = count
        
        # Filter overlapping patterns
        filtered_patterns = {}
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        for pattern, count in sorted_patterns[:100]:  # Limit to top 100
            # Check if not substring of already added pattern
            is_substring = any(pattern in p for p in filtered_patterns)
            if not is_substring:
                filtered_patterns[pattern] = count
        
        return filtered_patterns
    
    def _get_distribution(self, text: str) -> Dict[str, float]:
        """Get character probability distribution"""
        if not text:
            return {}
        
        freq = Counter(text)
        total = len(text)
        
        return {char: count / total for char, count in freq.items()}
    
    def _calculate_uniformity(self, distribution: Dict[str, float]) -> float:
        """Calculate uniformity of distribution"""
        if not distribution:
            return 0.0
        
        values = list(distribution.values())
        if len(values) <= 1:
            return 0.0
        
        # Calculate variance
        mean = np.mean(values)
        variance = np.var(values)
        
        # Lower variance means more uniform
        uniformity = 1.0 / (1.0 + variance * 100)
        
        return min(uniformity, 1.0)
    
    def _calculate_token_diversity(self, text: str) -> float:
        """Calculate token diversity metric"""
        if not text:
            return 0.0
        
        tokens = text.split()
        if not tokens:
            return 0.0
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        return unique_tokens / total_tokens
    
    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate pattern complexity score"""
        if not text:
            return 0.0
        
        # Check for various patterns
        patterns = {
            'code': len(re.findall(r'[{}()\[\];]', text)),
            'natural': len(re.findall(r'\b[A-Z][a-z]+\b', text)),
            'numeric': len(re.findall(r'\d+', text)),
            'special': len(re.findall(r'[!@#$%^&*]', text))
        }
        
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Weighted score
        score = (
            patterns['code'] * 0.3 +
            patterns['natural'] * 0.3 +
            patterns['numeric'] * 0.2 +
            patterns['special'] * 0.2
        ) / total_chars
        
        return min(score, 1.0)
    
    def _smooth_curve(self, values: List[float]) -> List[float]:
        """Apply smoothing to reduce noise"""
        if len(values) < 3:
            return values
        
        # Simple moving average
        window = 3
        smoothed = []
        
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            
            window_values = values[start:end]
            smoothed.append(np.mean(window_values))
        
        return smoothed
    
    def _update_cache(self, key: Any, value: Any):
        """Update cache with size management"""
        if len(self.cache) >= self.cache_size_limit:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def _empty_profile(self) -> EntropyProfile:
        """Return empty entropy profile"""
        return EntropyProfile(
            global_entropy=0.0,
            local_entropies=[],
            entropy_gradient=[],
            peaks=[],
            valleys=[],
            boundaries=[],
            segments=[],
            statistics={}
        )

class EntropyChunker:
    """Chunker that uses entropy analysis for intelligent splitting"""
    
    def __init__(self, analyzer: EntropyAnalyzer, max_tokens: int = 450):
        """
        Initialize entropy-based chunker
        
        Args:
            analyzer: Entropy analyzer instance
            max_tokens: Maximum tokens per chunk
        """
        self.analyzer = analyzer
        self.max_tokens = max_tokens
    
    def chunk_by_entropy(self, content: str, 
                         min_chunk_size: int = 10) -> List[Tuple[int, int]]:
        """
        Create chunks based on entropy analysis
        
        Args:
            content: Content to chunk
            min_chunk_size: Minimum lines per chunk
            
        Returns:
            List of (start_line, end_line) tuples
        """
        lines = content.split('\n')
        
        if len(lines) < min_chunk_size:
            return [(0, len(lines))]
        
        # Analyze entropy profile
        profile = self.analyzer.analyze_content(content)
        
        # Use boundaries for initial segmentation
        chunks = []
        boundaries = [0] + profile.boundaries + [len(lines)]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Ensure minimum chunk size
            if end - start >= min_chunk_size:
                chunks.append((start, end))
            elif chunks:
                # Merge with previous chunk
                chunks[-1] = (chunks[-1][0], end)
            else:
                chunks.append((start, end))
        
        return chunks
    
    def adaptive_chunk(self, content: str, 
                      profile: Optional[EntropyProfile] = None) -> List[Dict[str, Any]]:
        """
        Adaptively chunk content using entropy profile
        
        Args:
            content: Content to chunk
            profile: Pre-computed entropy profile
            
        Returns:
            List of chunk dictionaries
        """
        if not profile:
            profile = self.analyzer.analyze_content(content)
        
        lines = content.split('\n')
        chunks = []
        
        # Use entropy segments
        for start, end, avg_entropy in profile.segments:
            segment_lines = lines[start:end]
            
            # Determine chunk type based on entropy
            if avg_entropy < 0.3:
                chunk_type = 'low_entropy'  # Likely repetitive/structured
            elif avg_entropy > 0.7:
                chunk_type = 'high_entropy'  # Likely compressed/random
            else:
                chunk_type = 'medium_entropy'  # Normal content
            
            chunks.append({
                'content': '\n'.join(segment_lines),
                'type': chunk_type,
                'start_line': start,
                'end_line': end,
                'avg_entropy': avg_entropy,
                'metadata': {
                    'entropy_stats': {
                        'min': min(profile.local_entropies[start:end]) if start < end else 0,
                        'max': max(profile.local_entropies[start:end]) if start < end else 0,
                        'mean': avg_entropy
                    }
                }
            })
        
        return chunks