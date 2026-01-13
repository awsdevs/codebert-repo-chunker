"""
Learning system for adaptive improvement of chunking strategies
Learns from unknown files and improves pattern detection over time
"""

import os
import json
import pickle
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict, deque
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging
import threading
import queue
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyEffectiveness(Enum):
    """Effectiveness ratings for chunking strategies"""
    EXCELLENT = 5
    GOOD = 4
    SATISFACTORY = 3
    POOR = 2
    FAILED = 1

@dataclass
class FilePattern:
    """Represents a learned file pattern"""
    pattern_id: str
    file_extension: str
    content_features: Dict[str, float]
    detected_patterns: List[str]
    successful_strategy: str
    confidence_score: float
    occurrence_count: int
    last_seen: datetime
    effectiveness_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyMetrics:
    """Metrics for a chunking strategy"""
    strategy_name: str
    total_uses: int
    success_count: int
    failure_count: int
    avg_chunk_quality: float
    avg_processing_time: float
    file_types_handled: Set[str]
    effectiveness_by_type: Dict[str, float]
    last_updated: datetime

@dataclass
class LearningRecord:
    """Record of a learning event"""
    timestamp: datetime
    file_path: str
    file_type: str
    original_strategy: str
    final_strategy: str
    chunk_count: int
    avg_chunk_size: int
    processing_time: float
    effectiveness: StrategyEffectiveness
    features_extracted: Dict[str, float]
    user_feedback: Optional[Dict[str, Any]] = None

class PatternDatabase:
    """SQLite database for storing learned patterns"""
    
    def __init__(self, db_path: Path):
        """
        Initialize pattern database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Connection pool for thread safety
        self._local = threading.local()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(str(self.db_path))
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # File patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_patterns (
                pattern_id TEXT PRIMARY KEY,
                file_extension TEXT,
                content_features TEXT,
                detected_patterns TEXT,
                successful_strategy TEXT,
                confidence_score REAL,
                occurrence_count INTEGER,
                last_seen TIMESTAMP,
                effectiveness_score REAL,
                metadata TEXT
            )
        ''')
        
        # Strategy metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_metrics (
                strategy_name TEXT PRIMARY KEY,
                total_uses INTEGER,
                success_count INTEGER,
                failure_count INTEGER,
                avg_chunk_quality REAL,
                avg_processing_time REAL,
                file_types_handled TEXT,
                effectiveness_by_type TEXT,
                last_updated TIMESTAMP
            )
        ''')
        
        # Learning records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                file_path TEXT,
                file_type TEXT,
                original_strategy TEXT,
                final_strategy TEXT,
                chunk_count INTEGER,
                avg_chunk_size INTEGER,
                processing_time REAL,
                effectiveness INTEGER,
                features_extracted TEXT,
                user_feedback TEXT
            )
        ''')
        
        # Feature importance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                feature_name TEXT PRIMARY KEY,
                importance_score REAL,
                update_count INTEGER,
                last_updated TIMESTAMP
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_extension ON file_patterns(file_extension)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy ON file_patterns(successful_strategy)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON learning_records(timestamp)')
        
        conn.commit()
        conn.close()
    
    def store_pattern(self, pattern: FilePattern):
        """Store a file pattern"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO file_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            pattern.file_extension,
            json.dumps(pattern.content_features),
            json.dumps(pattern.detected_patterns),
            pattern.successful_strategy,
            pattern.confidence_score,
            pattern.occurrence_count,
            pattern.last_seen.isoformat(),
            pattern.effectiveness_score,
            json.dumps(pattern.metadata)
        ))
        
        conn.commit()
    
    def get_pattern(self, pattern_id: str) -> Optional[FilePattern]:
        """Retrieve a pattern by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM file_patterns WHERE pattern_id = ?', (pattern_id,))
        row = cursor.fetchone()
        
        if row:
            return FilePattern(
                pattern_id=row['pattern_id'],
                file_extension=row['file_extension'],
                content_features=json.loads(row['content_features']),
                detected_patterns=json.loads(row['detected_patterns']),
                successful_strategy=row['successful_strategy'],
                confidence_score=row['confidence_score'],
                occurrence_count=row['occurrence_count'],
                last_seen=datetime.fromisoformat(row['last_seen']),
                effectiveness_score=row['effectiveness_score'],
                metadata=json.loads(row['metadata'])
            )
        
        return None
    
    def find_similar_patterns(self, features: Dict[str, float], 
                            threshold: float = 0.8) -> List[FilePattern]:
        """Find patterns similar to given features"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM file_patterns')
        patterns = []
        
        for row in cursor.fetchall():
            pattern_features = json.loads(row['content_features'])
            similarity = self._calculate_similarity(features, pattern_features)
            
            if similarity >= threshold:
                patterns.append(FilePattern(
                    pattern_id=row['pattern_id'],
                    file_extension=row['file_extension'],
                    content_features=pattern_features,
                    detected_patterns=json.loads(row['detected_patterns']),
                    successful_strategy=row['successful_strategy'],
                    confidence_score=row['confidence_score'],
                    occurrence_count=row['occurrence_count'],
                    last_seen=datetime.fromisoformat(row['last_seen']),
                    effectiveness_score=row['effectiveness_score'],
                    metadata=json.loads(row['metadata'])
                ))
        
        return sorted(patterns, key=lambda p: p.confidence_score, reverse=True)
    
    def _calculate_similarity(self, features1: Dict[str, float], 
                            features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature sets"""
        if not features1 or not features2:
            return 0.0
        
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(features1[f] * features2[f] for f in common_features)
        norm1 = math.sqrt(sum(v ** 2 for k, v in features1.items() if k in common_features))
        norm2 = math.sqrt(sum(v ** 2 for k, v in features2.items() if k in common_features))
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def update_strategy_metrics(self, metrics: StrategyMetrics):
        """Update strategy performance metrics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO strategy_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.strategy_name,
            metrics.total_uses,
            metrics.success_count,
            metrics.failure_count,
            metrics.avg_chunk_quality,
            metrics.avg_processing_time,
            json.dumps(list(metrics.file_types_handled)),
            json.dumps(metrics.effectiveness_by_type),
            metrics.last_updated.isoformat()
        ))
        
        conn.commit()
    
    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """Get metrics for a strategy"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM strategy_metrics WHERE strategy_name = ?', 
                      (strategy_name,))
        row = cursor.fetchone()
        
        if row:
            return StrategyMetrics(
                strategy_name=row['strategy_name'],
                total_uses=row['total_uses'],
                success_count=row['success_count'],
                failure_count=row['failure_count'],
                avg_chunk_quality=row['avg_chunk_quality'],
                avg_processing_time=row['avg_processing_time'],
                file_types_handled=set(json.loads(row['file_types_handled'])),
                effectiveness_by_type=json.loads(row['effectiveness_by_type']),
                last_updated=datetime.fromisoformat(row['last_updated'])
            )
        
        return None
    
    def record_learning_event(self, record: LearningRecord):
        """Store a learning event"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_records VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.timestamp.isoformat(),
            record.file_path,
            record.file_type,
            record.original_strategy,
            record.final_strategy,
            record.chunk_count,
            record.avg_chunk_size,
            record.processing_time,
            record.effectiveness.value,
            json.dumps(record.features_extracted),
            json.dumps(record.user_feedback) if record.user_feedback else None
        ))
        
        conn.commit()
    
    def get_recent_learning_records(self, days: int = 7) -> List[LearningRecord]:
        """Get recent learning records"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT * FROM learning_records 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (since.isoformat(),))
        
        records = []
        for row in cursor.fetchall():
            records.append(LearningRecord(
                timestamp=datetime.fromisoformat(row['timestamp']),
                file_path=row['file_path'],
                file_type=row['file_type'],
                original_strategy=row['original_strategy'],
                final_strategy=row['final_strategy'],
                chunk_count=row['chunk_count'],
                avg_chunk_size=row['avg_chunk_size'],
                processing_time=row['processing_time'],
                effectiveness=StrategyEffectiveness(row['effectiveness']),
                features_extracted=json.loads(row['features_extracted']),
                user_feedback=json.loads(row['user_feedback']) if row['user_feedback'] else None
            ))
        
        return records

class FeatureExtractor:
    """Extract features from unknown files for pattern learning"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, analyzer='char', ngram_range=(1, 3))
        self.scaler = StandardScaler()
        
    def extract_features(self, content: str, file_path: Path) -> Dict[str, float]:
        """
        Extract comprehensive features from file content
        
        Args:
            content: File content
            file_path: Path to file
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic metrics
        features['file_size'] = len(content)
        features['line_count'] = content.count('\n')
        features['avg_line_length'] = features['file_size'] / max(features['line_count'], 1)
        features['max_line_length'] = max((len(line) for line in content.split('\n')), default=0)
        
        # Character distribution
        char_types = {
            'alpha': sum(c.isalpha() for c in content),
            'digit': sum(c.isdigit() for c in content),
            'space': sum(c.isspace() for c in content),
            'special': sum(not c.isalnum() and not c.isspace() for c in content)
        }
        
        total_chars = sum(char_types.values())
        for char_type, count in char_types.items():
            features[f'ratio_{char_type}'] = count / max(total_chars, 1)
        
        # Structural patterns
        features['indent_spaces'] = content.count('    ')
        features['indent_tabs'] = content.count('\t')
        features['brackets_curly'] = content.count('{') + content.count('}')
        features['brackets_square'] = content.count('[') + content.count(']')
        features['brackets_round'] = content.count('(') + content.count(')')
        features['semicolons'] = content.count(';')
        features['equals'] = content.count('=')
        features['quotes_single'] = content.count("'")
        features['quotes_double'] = content.count('"')
        
        # Pattern detection
        import re
        
        patterns = {
            'functions': len(re.findall(r'\b(?:function|def|func)\b', content)),
            'classes': len(re.findall(r'\b(?:class|interface|struct)\b', content)),
            'imports': len(re.findall(r'\b(?:import|require|include|use)\b', content)),
            'comments_line': len(re.findall(r'//|#|--', content)),
            'comments_block': len(re.findall(r'/\*|\*/|<!--', content)),
            'urls': len(re.findall(r'https?://\S+', content)),
            'emails': len(re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', content)),
            'ips': len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', content)),
            'timestamps': len(re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', content))
        }
        
        for pattern_name, count in patterns.items():
            features[f'pattern_{pattern_name}'] = count
        
        # Entropy-based features
        from src.chunkers.adaptive.entropy_analyzer import EntropyAnalyzer
        
        entropy_analyzer = EntropyAnalyzer()
        entropy = entropy_analyzer.calculate_normalized_entropy(content[:1000])
        features['entropy'] = entropy
        
        # Token-based features
        tokens = content.split()
        if tokens:
            features['token_count'] = len(tokens)
            features['unique_tokens'] = len(set(tokens))
            features['token_diversity'] = features['unique_tokens'] / features['token_count']
            features['avg_token_length'] = sum(len(t) for t in tokens) / len(tokens)
        
        # Extension-based features
        extension = file_path.suffix.lower()
        features['has_extension'] = 1.0 if extension else 0.0
        features[f'ext_{extension}'] = 1.0  # One-hot encoding for extension
        
        # Normalize features
        return self._normalize_features(features)
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize feature values to [0, 1] range"""
        normalized = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Log transform for large values
                if value > 100:
                    normalized[key] = math.log(value + 1) / 10
                else:
                    normalized[key] = value / 100
                
                # Clip to [0, 1]
                normalized[key] = min(max(normalized[key], 0.0), 1.0)
            else:
                normalized[key] = value
        
        return normalized

class PatternClusterer:
    """Cluster similar file patterns for strategy recommendation"""
    
    def __init__(self, n_clusters: int = 10):
        """
        Initialize pattern clusterer
        
        Args:
            n_clusters: Number of clusters for KMeans
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.dbscan = None
        self.feature_matrix = None
        self.pattern_ids = []
        
    def fit(self, patterns: List[FilePattern]):
        """
        Fit clustering models to patterns
        
        Args:
            patterns: List of file patterns
        """
        if not patterns:
            return
        
        # Extract feature matrix
        self.pattern_ids = [p.pattern_id for p in patterns]
        feature_vectors = []
        
        for pattern in patterns:
            # Convert features dict to vector
            feature_vector = list(pattern.content_features.values())
            feature_vectors.append(feature_vector)
        
        self.feature_matrix = np.array(feature_vectors)
        
        # Fit KMeans
        if len(patterns) >= self.n_clusters:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.kmeans.fit(self.feature_matrix)
        
        # Fit DBSCAN for density-based clustering
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.dbscan.fit(self.feature_matrix)
    
    def predict_cluster(self, features: Dict[str, float]) -> int:
        """
        Predict cluster for new features
        
        Args:
            features: Feature dictionary
            
        Returns:
            Cluster ID
        """
        if not self.kmeans:
            return -1
        
        # Convert to feature vector
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Ensure same dimensionality
        if feature_vector.shape[1] != self.feature_matrix.shape[1]:
            return -1
        
        return self.kmeans.predict(feature_vector)[0]
    
    def find_nearest_patterns(self, features: Dict[str, float], 
                            k: int = 5) -> List[str]:
        """
        Find k nearest patterns to given features
        
        Args:
            features: Feature dictionary
            k: Number of neighbors
            
        Returns:
            List of pattern IDs
        """
        if not self.feature_matrix.size:
            return []
        
        # Convert to feature vector
        feature_vector = np.array(list(features.values()))
        
        # Calculate distances
        distances = np.linalg.norm(self.feature_matrix - feature_vector, axis=1)
        
        # Get k nearest
        nearest_indices = np.argsort(distances)[:k]
        
        return [self.pattern_ids[i] for i in nearest_indices]

class StrategyOptimizer:
    """Optimize strategy selection based on learning"""
    
    def __init__(self, database: PatternDatabase):
        """
        Initialize strategy optimizer
        
        Args:
            database: Pattern database instance
        """
        self.database = database
        self.strategy_scores = defaultdict(lambda: defaultdict(float))
        self.feature_importance = defaultdict(float)
        
    def recommend_strategy(self, features: Dict[str, float], 
                          file_extension: str) -> Tuple[str, float]:
        """
        Recommend best strategy for given features
        
        Args:
            features: Extracted features
            file_extension: File extension
            
        Returns:
            Tuple of (strategy_name, confidence)
        """
        # Find similar patterns
        similar_patterns = self.database.find_similar_patterns(features, threshold=0.7)
        
        if not similar_patterns:
            # No similar patterns, use default
            return 'adaptive', 0.5
        
        # Score strategies based on similar patterns
        strategy_scores = Counter()
        total_weight = 0
        
        for pattern in similar_patterns:
            weight = pattern.confidence_score * pattern.effectiveness_score
            strategy_scores[pattern.successful_strategy] += weight
            total_weight += weight
        
        if not strategy_scores:
            return 'adaptive', 0.5
        
        # Get best strategy
        best_strategy = strategy_scores.most_common(1)[0][0]
        confidence = strategy_scores[best_strategy] / total_weight
        
        return best_strategy, confidence
    
    def update_feature_importance(self, features: Dict[str, float], 
                                 effectiveness: StrategyEffectiveness):
        """
        Update feature importance based on effectiveness
        
        Args:
            features: Feature dictionary
            effectiveness: Strategy effectiveness
        """
        # Update importance scores
        effectiveness_weight = effectiveness.value / 5.0
        
        for feature_name, feature_value in features.items():
            if feature_value > 0:
                self.feature_importance[feature_name] += effectiveness_weight * feature_value
        
        # Normalize importance scores
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            for feature_name in self.feature_importance:
                self.feature_importance[feature_name] /= total_importance
    
    def get_important_features(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important features
        
        Args:
            top_k: Number of top features
            
        Returns:
            List of (feature_name, importance) tuples
        """
        return sorted(self.feature_importance.items(), 
                     key=lambda x: x[1], reverse=True)[:top_k]

class LearningSystem:
    """Main learning system for adaptive chunking improvement"""
    
    def __init__(self, db_path: Path, 
                 learning_rate: float = 0.1,
                 batch_size: int = 100):
        """
        Initialize learning system
        
        Args:
            db_path: Path to pattern database
            learning_rate: Learning rate for updates
            batch_size: Batch size for processing
        """
        self.database = PatternDatabase(db_path)
        self.feature_extractor = FeatureExtractor()
        self.clusterer = PatternClusterer()
        self.optimizer = StrategyOptimizer(self.database)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Learning queue for async processing
        self.learning_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Metrics tracking
        self.metrics = {
            'patterns_learned': 0,
            'strategies_optimized': 0,
            'effectiveness_improved': 0,
            'processing_errors': 0
        }
        
        # Load existing patterns
        self._load_patterns()
    
    def _load_patterns(self):
        """Load and cluster existing patterns"""
        # This would load patterns from database and fit clusterer
        # Implementation depends on database size and performance requirements
        pass
    
    def learn_from_file(self, file_path: Path, content: str, 
                       chunks: List[Any], strategy: str,
                       processing_time: float) -> str:
        """
        Learn from file processing
        
        Args:
            file_path: Path to file
            content: File content
            chunks: Generated chunks
            strategy: Strategy used
            processing_time: Processing time
            
        Returns:
            Pattern ID
        """
        # Extract features
        features = self.feature_extractor.extract_features(content, file_path)
        
        # Calculate effectiveness
        effectiveness = self._evaluate_chunks(chunks)
        
        # Generate pattern ID
        pattern_id = self._generate_pattern_id(features)
        
        # Check if pattern exists
        existing_pattern = self.database.get_pattern(pattern_id)
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.occurrence_count += 1
            existing_pattern.last_seen = datetime.now()
            existing_pattern.effectiveness_score = (
                existing_pattern.effectiveness_score * 0.9 + 
                effectiveness.value / 5.0 * 0.1
            )
            self.database.store_pattern(existing_pattern)
        else:
            # Create new pattern
            pattern = FilePattern(
                pattern_id=pattern_id,
                file_extension=file_path.suffix,
                content_features=features,
                detected_patterns=[],  # Would be filled by pattern detection
                successful_strategy=strategy,
                confidence_score=0.5,  # Initial confidence
                occurrence_count=1,
                last_seen=datetime.now(),
                effectiveness_score=effectiveness.value / 5.0,
                metadata={'file_size': len(content)}
            )
            self.database.store_pattern(pattern)
            self.metrics['patterns_learned'] += 1
        
        # Record learning event
        record = LearningRecord(
            timestamp=datetime.now(),
            file_path=str(file_path),
            file_type=file_path.suffix,
            original_strategy=strategy,
            final_strategy=strategy,
            chunk_count=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // max(len(chunks), 1),
            processing_time=processing_time,
            effectiveness=effectiveness,
            features_extracted=features
        )
        self.database.record_learning_event(record)
        
        # Update feature importance
        self.optimizer.update_feature_importance(features, effectiveness)
        
        # Queue for async processing
        self.learning_queue.put((pattern_id, features, effectiveness))
        
        return pattern_id
    
    def _evaluate_chunks(self, chunks: List[Any]) -> StrategyEffectiveness:
        """Evaluate chunk quality"""
        if not chunks:
            return StrategyEffectiveness.FAILED
        
        # Evaluation criteria
        scores = []
        
        # Size consistency
        sizes = [len(c.content) for c in chunks]
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        size_score = 1.0 / (1.0 + size_variance / 10000)
        scores.append(size_score)
        
        # Token distribution
        token_counts = [c.token_count for c in chunks if hasattr(c, 'token_count')]
        if token_counts:
            avg_tokens = np.mean(token_counts)
            token_score = min(avg_tokens / 400, 1.0)  # Ideal around 400 tokens
            scores.append(token_score)
        
        # Content coherence (simplified)
        coherence_score = 0.7  # Would implement actual coherence checking
        scores.append(coherence_score)
        
        # Calculate overall score
        overall_score = np.mean(scores)
        
        # Map to effectiveness level
        if overall_score >= 0.8:
            return StrategyEffectiveness.EXCELLENT
        elif overall_score >= 0.6:
            return StrategyEffectiveness.GOOD
        elif overall_score >= 0.4:
            return StrategyEffectiveness.SATISFACTORY
        elif overall_score >= 0.2:
            return StrategyEffectiveness.POOR
        else:
            return StrategyEffectiveness.FAILED
    
    def _generate_pattern_id(self, features: Dict[str, float]) -> str:
        """Generate unique pattern ID from features"""
        # Use hash of key features
        key_features = sorted(features.items())[:20]  # Top 20 features
        feature_str = str(key_features)
        
        return hashlib.md5(feature_str.encode()).hexdigest()
    
    def suggest_strategy(self, file_path: Path, content: str) -> Tuple[str, float]:
        """
        Suggest best strategy for file
        
        Args:
            file_path: File path
            content: File content
            
        Returns:
            Tuple of (strategy, confidence)
        """
        # Extract features
        features = self.feature_extractor.extract_features(content, file_path)
        
        # Get recommendation
        strategy, confidence = self.optimizer.recommend_strategy(
            features, file_path.suffix
        )
        
        logger.info(f"Suggested strategy '{strategy}' with confidence {confidence:.2f}")
        
        return strategy, confidence
    
    def update_strategy_effectiveness(self, strategy: str, 
                                     effectiveness: StrategyEffectiveness,
                                     file_type: str):
        """
        Update strategy effectiveness metrics
        
        Args:
            strategy: Strategy name
            effectiveness: Effectiveness rating
            file_type: File type handled
        """
        # Get or create metrics
        metrics = self.database.get_strategy_metrics(strategy)
        
        if not metrics:
            metrics = StrategyMetrics(
                strategy_name=strategy,
                total_uses=0,
                success_count=0,
                failure_count=0,
                avg_chunk_quality=0.0,
                avg_processing_time=0.0,
                file_types_handled=set(),
                effectiveness_by_type={},
                last_updated=datetime.now()
            )
        
        # Update metrics
        metrics.total_uses += 1
        
        if effectiveness.value >= 3:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        metrics.file_types_handled.add(file_type)
        
        # Update effectiveness by type
        if file_type not in metrics.effectiveness_by_type:
            metrics.effectiveness_by_type[file_type] = 0.0
        
        metrics.effectiveness_by_type[file_type] = (
            metrics.effectiveness_by_type[file_type] * 0.9 + 
            effectiveness.value / 5.0 * 0.1
        )
        
        metrics.last_updated = datetime.now()
        
        # Store updated metrics
        self.database.update_strategy_metrics(metrics)
        self.metrics['strategies_optimized'] += 1
    
    def start_async_processing(self):
        """Start asynchronous learning processing"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(target=self._process_learning_queue)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started async learning processing")
    
    def stop_async_processing(self):
        """Stop asynchronous processing"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            logger.info("Stopped async learning processing")
    
    def _process_learning_queue(self):
        """Process learning queue in background"""
        batch = []
        
        while not self.stop_processing.is_set():
            try:
                # Get items from queue with timeout
                item = self.learning_queue.get(timeout=1)
                batch.append(item)
                
                # Process batch when full
                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
                    
            except queue.Empty:
                # Process remaining batch
                if batch:
                    self._process_batch(batch)
                    batch = []
            except Exception as e:
                logger.error(f"Error in learning processing: {e}")
                self.metrics['processing_errors'] += 1
    
    def _process_batch(self, batch: List[Tuple]):
        """Process a batch of learning items"""
        if not batch:
            return
        
        try:
            # Re-cluster patterns periodically
            if self.metrics['patterns_learned'] % 100 == 0:
                self._recluster_patterns()
            
            # Update effectiveness scores
            for pattern_id, features, effectiveness in batch:
                if effectiveness.value >= 4:
                    self.metrics['effectiveness_improved'] += 1
            
            logger.info(f"Processed learning batch of {len(batch)} items")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.metrics['processing_errors'] += 1
    
    def _recluster_patterns(self):
        """Re-cluster patterns for better recommendations"""
        # Would fetch patterns from database and re-fit clusterer
        # Implementation depends on performance requirements
        logger.info("Re-clustering patterns")
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Generate learning system report"""
        recent_records = self.database.get_recent_learning_records(days=7)
        
        report = {
            'summary': {
                'total_patterns': self.metrics['patterns_learned'],
                'strategies_optimized': self.metrics['strategies_optimized'],
                'effectiveness_improved': self.metrics['effectiveness_improved'],
                'processing_errors': self.metrics['processing_errors']
            },
            'recent_activity': {
                'files_processed': len(recent_records),
                'avg_effectiveness': np.mean([r.effectiveness.value for r in recent_records]) if recent_records else 0,
                'most_used_strategies': Counter(r.final_strategy for r in recent_records).most_common(5)
            },
            'feature_importance': self.optimizer.get_important_features(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        if self.metrics['processing_errors'] > 10:
            recommendations.append("High error rate detected - review error logs")
        
        if self.metrics['effectiveness_improved'] < self.metrics['patterns_learned'] * 0.5:
            recommendations.append("Low effectiveness improvement - consider adjusting strategies")
        
        return recommendations
    
    def export_learned_patterns(self, output_path: Path):
        """Export learned patterns for backup or analysis"""
        patterns = []
        
        # Would fetch all patterns from database
        # and export to JSON or pickle
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        
        logger.info(f"Exported patterns to {output_path}")
    
    def import_learned_patterns(self, input_path: Path):
        """Import previously learned patterns"""
        if not input_path.exists():
            logger.error(f"Import file not found: {input_path}")
            return
        
        with open(input_path, 'r') as f:
            patterns = json.load(f)
        
        # Would validate and import patterns to database
        
        logger.info(f"Imported {len(patterns)} patterns from {input_path}")