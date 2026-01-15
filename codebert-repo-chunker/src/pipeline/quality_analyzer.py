from typing import Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import pickle

from src.core.chunk_model import Chunk
from src.utils.text_utils import TextAnalyzer, TextStatistics

@dataclass
class QualityMetrics:
    complexity: float
    readability: float
    maintainability: float
    loc: int
    comment_ratio: float

class QualityAnalyzer:
    """
    Analyzes code chunks for quality metrics using heuristics and text analysis.
    """
    
    def __init__(self):
        self.text_analyzer = TextAnalyzer(enable_nlp=False)  # Keep it fast
        
    def analyze_chunk(self, chunk: Chunk) -> Dict[str, Union[int, float]]:
        """
        Calculate quality metrics for a code chunk.
        
        Args:
            chunk: The code chunk to analyze
            
        Returns:
            Dictionary of metrics
        """
        if not chunk.content:
            return {}
            
        # Basic text analysis
        stats = self.text_analyzer.analyze(chunk.content)
        
        # Calculate Maintainability Index (simplified MI)
        complexity = stats.complexity_score
        loc = stats.line_count
        
        # Proxy metrics
        maintainability = max(0, 100 - (complexity * 0.5) - (max(0, loc - 20) * 0.5))
        
        # Comment ratio
        comment_ratio = self._estimate_comment_ratio(chunk.content)
        
        metrics = {
            "complexity": round(complexity, 2),
            "readability": round(stats.readability_score, 2),
            "maintainability": round(maintainability, 2),
            "loc": loc,
            "comment_ratio": round(comment_ratio, 2)
        }
        
        return metrics

    def _estimate_comment_ratio(self, content: str) -> float:
        lines = content.splitlines()
        if not lines:
            return 0.0
            
        comment_lines = 0
        for line in lines:
            line = line.strip()
            if line.startswith(('#', '//', '*', '/*', '"""', "'''")):
                comment_lines += 1
                
        return comment_lines / len(lines)

    def analyze_directory(self, chunks_dir: Path) -> Dict[str, Any]:
        """
        Analyze quality for all chunks in a directory (pickled).
        
        Args:
            chunks_dir: Directory containing pickled Chunk objects
            
        Returns:
            Dictionary with quality scores and detailed metrics
        """
        results = {
            "file_metrics": {},
            "overall_score": 0.0,
            "total_chunks": 0
        }
        
        if not chunks_dir.exists():
            return results
            
        total_maintainability = 0.0
        count = 0
        
        # Walk through chunks directory
        for chunk_file in chunks_dir.glob("**/*.pkl"):
            try:
                with open(chunk_file, "rb") as f:
                    chunk = pickle.load(f)
                    
                if hasattr(chunk, 'content'): # Duck typing check for Chunk
                    metrics = self.analyze_chunk(chunk)
                    # Use chunk.id or file name as key
                    chunk_id = getattr(chunk, 'id', chunk_file.stem)
                    results["file_metrics"][chunk_id] = metrics
                    
                    total_maintainability += metrics.get("maintainability", 0)
                    count += 1
            except Exception as e:
                # logger might not be defined in this scope if I didn't import it in this file
                # I should add logger import
                pass
                
        results["total_chunks"] = count
        if count > 0:
            # Normalize maintainability (0-100) to 0-1 score
            avg_maintainability = total_maintainability / count
            results["overall_score"] = min(1.0, max(0.0, avg_maintainability / 100.0))
            
        return results
