"""
Similarity calculation utilities for code comparison
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import Levenshtein
from difflib import SequenceMatcher
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import re
import logging

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Calculate various similarity metrics between code snippets"""
    
    def __init__(self):
        """Initialize similarity calculator"""
        self.tfidf_vectorizer = None
    
    def calculate_all_similarities(self, code1: str, code2: str) -> Dict[str, float]:
        """
        Calculate all similarity metrics between two code snippets
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            Dictionary of similarity scores
        """
        return {
            'exact_match': self.exact_match(code1, code2),
            'levenshtein': self.levenshtein_similarity(code1, code2),
            'token': self.token_similarity(code1, code2),
            'structural': self.structural_similarity(code1, code2),
            'semantic': self.semantic_similarity(code1, code2),
            'jaccard': self.jaccard_similarity(code1, code2),
            'cosine_tfidf': self.cosine_tfidf_similarity(code1, code2)
        }
    
    def exact_match(self, code1: str, code2: str) -> float:
        """Check if two code snippets are exactly the same"""
        return 1.0 if code1 == code2 else 0.0
    
    def levenshtein_similarity(self, code1: str, code2: str) -> float:
        """Calculate Levenshtein distance-based similarity"""
        if not code1 or not code2:
            return 0.0
        
        distance = Levenshtein.distance(code1, code2)
        max_len = max(len(code1), len(code2))
        
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    def token_similarity(self, code1: str, code2: str) -> float:
        """Calculate token-based similarity"""
        tokens1 = self._tokenize(code1)
        tokens2 = self._tokenize(code2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate overlap
        common = set(tokens1) & set(tokens2)
        total = set(tokens1) | set(tokens2)
        
        return len(common) / len(total) if total else 0.0
    
    def structural_similarity(self, code1: str, code2: str) -> float:
        """Calculate structural similarity (for Python code)"""
        try:
            # Parse AST
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            
            # Extract structure
            structure1 = self._extract_structure(tree1)
            structure2 = self._extract_structure(tree2)
            
            # Compare structures
            if not structure1 or not structure2:
                return 0.0
            
            common = set(structure1) & set(structure2)
            total = set(structure1) | set(structure2)
            
            return len(common) / len(total) if total else 0.0
            
        except:
            # Fallback for non-Python or invalid code
            return self.token_similarity(code1, code2)
    
    def semantic_similarity(self, code1: str, code2: str) -> float:
        """Calculate semantic similarity using sequence matching"""
        # Normalize code
        norm1 = self._normalize_code(code1)
        norm2 = self._normalize_code(code2)
        
        # Use sequence matcher
        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()
    
    def jaccard_similarity(self, code1: str, code2: str) -> float:
        """Calculate Jaccard similarity"""
        tokens1 = set(self._tokenize(code1))
        tokens2 = set(self._tokenize(code2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0
    
    def cosine_tfidf_similarity(self, code1: str, code2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors"""
        try:
            # Initialize vectorizer if needed
            if not self.tfidf_vectorizer:
                self.tfidf_vectorizer = TfidfVectorizer(
                    tokenizer=self._tokenize,
                    lowercase=False
                )
                # Fit on both documents
                self.tfidf_vectorizer.fit([code1, code2])
            
            # Transform to vectors
            vectors = self.tfidf_vectorizer.transform([code1, code2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            
            return float(similarity)
            
        except:
            return 0.0
    
    def embedding_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray,
                           metric: str = 'cosine') -> float:
        """
        Calculate similarity between embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        elif metric == 'euclidean':
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)
        
        elif metric == 'manhattan':
            # Manhattan distance converted to similarity
            distance = np.sum(np.abs(embedding1 - embedding2))
            return 1.0 / (1.0 + distance)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _tokenize(self, code: str) -> List[str]:
        """Tokenize code into meaningful tokens"""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        
        # Filter out single characters except operators
        filtered = []
        for token in tokens:
            if len(token) > 1 or token in '+-*/=<>!&|':
                filtered.append(token)
        
        return filtered
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison"""
        # Remove whitespace variations
        code = re.sub(r'\s+', ' ', code)
        
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize variable names (simple approach)
        code = re.sub(r'\b[a-z_][a-z0-9_]*\b', 'VAR', code)
        
        return code.strip()
    
    def _extract_structure(self, tree: ast.AST) -> List[str]:
        """Extract structural elements from AST"""
        structure = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure.append(f"func:{node.name}")
            elif isinstance(node, ast.ClassDef):
                structure.append(f"class:{node.name}")
            elif isinstance(node, ast.For):
                structure.append("for_loop")
            elif isinstance(node, ast.While):
                structure.append("while_loop")
            elif isinstance(node, ast.If):
                structure.append("if_statement")
            elif isinstance(node, ast.Try):
                structure.append("try_except")
        
        return structure
    
    def find_similar_chunks(self, query: str, chunks: List[str],
                          threshold: float = 0.7,
                          metric: str = 'semantic') -> List[Tuple[int, float]]:
        """
        Find similar chunks to a query
        
        Args:
            query: Query code snippet
            chunks: List of code chunks to search
            threshold: Similarity threshold
            metric: Similarity metric to use
            
        Returns:
            List of (index, similarity) tuples
        """
        similarities = []
        
        for i, chunk in enumerate(chunks):
            if metric == 'levenshtein':
                sim = self.levenshtein_similarity(query, chunk)
            elif metric == 'token':
                sim = self.token_similarity(query, chunk)
            elif metric == 'structural':
                sim = self.structural_similarity(query, chunk)
            elif metric == 'semantic':
                sim = self.semantic_similarity(query, chunk)
            elif metric == 'jaccard':
                sim = self.jaccard_similarity(query, chunk)
            else:
                sim = self.semantic_similarity(query, chunk)
            
            if sim >= threshold:
                similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities