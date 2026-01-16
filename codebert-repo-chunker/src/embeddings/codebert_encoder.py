"""
CodeBERT encoder for generating embeddings from code chunks
Provides efficient encoding with batching, caching, and optimization
"""

import torch
import torch.nn as nn
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import (
    RobertaTokenizer, 
    RobertaModel,
    RobertaConfig,
    AutoTokenizer,
    AutoModel
)
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import hashlib
import pickle
from src.utils.logger import get_logger
from collections import defaultdict
from enum import Enum
import time
from tqdm import tqdm
import faiss
import onnxruntime as ort
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

class ModelType(Enum):
    """Available model types for code embedding"""
    CODEBERT_BASE = "microsoft/codebert-base"
    GRAPHCODEBERT = "microsoft/graphcodebert-base"
    UNIXCODER = "microsoft/unixcoder-base"
    CODET5_BASE = "Salesforce/codet5-base"
    CODEGEN = "Salesforce/codegen-350M-mono"
    STARENCODER = "bigcode/starencoder"
    CODEBERT_MLNET = "microsoft/codebert-base-mlm"
    CODESAGE = "codesage/codesage-small"
    CUSTOM = "custom"

class PoolingStrategy(Enum):
    """Pooling strategies for sequence embeddings"""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    WEIGHTED_MEAN = "weighted_mean"
    ATTENTION = "attention"

@dataclass
class EncoderConfig:
    """Configuration for CodeBERT encoder"""
    model_type: ModelType = ModelType.CODEBERT_BASE
    model_name: str = "microsoft/codebert-base"
    max_length: int = 512
    batch_size: int = 32
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    normalize_embeddings: bool = True
    use_cuda: bool = True
    use_fp16: bool = False
    cache_embeddings: bool = True
    cache_dir: Path = Path(".cache/embeddings")
    onnx_optimization: bool = False
    quantization: bool = False
    num_workers: int = 4
    device: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.device is None:
            if self.use_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                self.use_cuda = False
        
        # Create cache directory
        if self.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class EmbeddingResult:
    """Result of encoding operation"""
    embeddings: np.ndarray
    tokens: List[List[str]]
    attention_masks: Optional[np.ndarray] = None
    token_type_ids: Optional[np.ndarray] = None
    pooled_output: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CodeBERTEncoder:
    """
    CodeBERT encoder for generating code embeddings
    Supports multiple models and optimization strategies
    """
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        """
        Initialize CodeBERT encoder
        
        Args:
            config: Encoder configuration
        """
        self.config = config or EncoderConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize model and tokenizer
        self._init_model()
        
        # Initialize cache
        self.cache = {}
        self.cache_file = self.config.cache_dir / f"{self.config.model_type.value}_cache.pkl"
        self._load_cache()
        
        # Initialize ONNX runtime if enabled
        self.ort_session = None
        if self.config.onnx_optimization:
            self._init_onnx()
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _init_model(self):
        """Initialize model and tokenizer"""
        logger.info(f"Initializing {self.config.model_name} model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.config.model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable evaluation mode
            self.model.eval()
            
            # Apply optimizations
            if self.config.use_fp16 and self.config.use_cuda:
                self.model = self.model.half()
            
            # Compile model with torch.compile (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
            
            logger.info(f"Model initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Fallback to sentence-transformers
            logger.info("Falling back to sentence-transformers...")
            self._init_sentence_transformer()
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer as fallback"""
        self.model = SentenceTransformer('microsoft/codebert-base')
        self.tokenizer = self.model.tokenizer
        self.is_sentence_transformer = True
    
    def _init_onnx(self):
        """Initialize ONNX runtime for optimized inference"""
        try:
            onnx_path = self.config.cache_dir / f"{self.config.model_type.value}.onnx"
            
            if not onnx_path.exists():
                self._export_to_onnx(onnx_path)
            
            # Create ONNX runtime session
            providers = ['CUDAExecutionProvider'] if self.config.use_cuda else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            logger.info("ONNX runtime initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize ONNX runtime: {e}")
            self.config.onnx_optimization = False
    
    def _export_to_onnx(self, onnx_path: Path):
        """Export model to ONNX format"""
        logger.info("Exporting model to ONNX format...")
        
        # Create dummy input
        dummy_input = self.tokenizer(
            "def hello(): pass",
            return_tensors="pt",
            max_length=self.config.max_length,
            padding=True,
            truncation=True
        )
        
        # Move to device
        dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
        
        # Export
        torch.onnx.export(
            self.model,
            tuple(dummy_input.values()),
            str(onnx_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
        
        logger.info(f"Model exported to {onnx_path}")
    
    def encode(self, 
              texts: Union[str, List[str]], 
              batch_size: Optional[int] = None,
              show_progress: bool = False,
              return_tokens: bool = False) -> EmbeddingResult:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            return_tokens: Return tokenized text
            
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        batch_size = batch_size or self.config.batch_size
        
        # Check cache
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.config.cache_embeddings:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    cached_embeddings.append((i, self.cache[cache_key]))
                    self.stats['cache_hits'] += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats['cache_misses'] += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Encode uncached texts
        if uncached_texts:
            new_embeddings = self._encode_batch(
                uncached_texts, 
                batch_size, 
                show_progress,
                return_tokens
            )
            
            # Add to cache
            if self.config.cache_embeddings:
                for text, embedding in zip(uncached_texts, new_embeddings.embeddings):
                    cache_key = self._get_cache_key(text)
                    self.cache[cache_key] = embedding
        else:
            new_embeddings = EmbeddingResult(
                embeddings=np.array([]),
                tokens=[]
            )
        
        # Combine cached and new embeddings
        all_embeddings = np.zeros((len(texts), self.model.config.hidden_size))
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for i, idx in enumerate(uncached_indices):
            all_embeddings[idx] = new_embeddings.embeddings[i]
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            all_embeddings = self._normalize_embeddings(all_embeddings)
        
        # Update statistics
        self.stats['total_encoded'] += len(texts)
        
        result = EmbeddingResult(
            embeddings=all_embeddings[0] if single_input else all_embeddings,
            tokens=new_embeddings.tokens if return_tokens else [],
            metadata={
                'model': self.config.model_name,
                'pooling': self.config.pooling_strategy.value,
                'normalized': self.config.normalize_embeddings,
                'cache_hits': len(cached_embeddings),
                'cache_misses': len(uncached_texts)
            }
        )
        
        return result
    
    def _encode_batch(self, 
                     texts: List[str], 
                     batch_size: int,
                     show_progress: bool,
                     return_tokens: bool) -> EmbeddingResult:
        """Encode a batch of texts"""
        embeddings = []
        all_tokens = []
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(total=len(texts), desc="Encoding")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get embeddings
                if self.config.onnx_optimization and self.ort_session:
                    batch_embeddings = self._onnx_inference(encoded)
                else:
                    batch_embeddings = self._model_inference(encoded)
                
                # Apply pooling
                pooled = self._apply_pooling(batch_embeddings, encoded['attention_mask'])
                
                embeddings.append(pooled.cpu().numpy())
                
                # Store tokens if requested
                if return_tokens:
                    for input_ids in encoded['input_ids']:
                        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                        all_tokens.append(tokens)
                
                if show_progress:
                    pbar.update(len(batch_texts))
        
        if show_progress:
            pbar.close()
        
        return EmbeddingResult(
            embeddings=np.vstack(embeddings),
            tokens=all_tokens
        )
    
    def _model_inference(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run model inference"""
        outputs = self.model(**encoded)
        return outputs.last_hidden_state
    
    def _onnx_inference(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run ONNX inference"""
        ort_inputs = {
            'input_ids': encoded['input_ids'].cpu().numpy(),
            'attention_mask': encoded['attention_mask'].cpu().numpy()
        }
        
        outputs = self.ort_session.run(None, ort_inputs)
        return torch.tensor(outputs[0]).to(self.device)
    
    def _apply_pooling(self, 
                      hidden_states: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to hidden states"""
        if self.config.pooling_strategy == PoolingStrategy.MEAN:
            # Mean pooling
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.config.pooling_strategy == PoolingStrategy.MAX:
            # Max pooling
            hidden_states[attention_mask == 0] = -1e9
            return torch.max(hidden_states, 1)[0]
        
        elif self.config.pooling_strategy == PoolingStrategy.CLS:
            # CLS token
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_strategy == PoolingStrategy.WEIGHTED_MEAN:
            # Weighted mean (decreasing weights)
            seq_len = hidden_states.size(1)
            weights = torch.arange(seq_len, 0, -1, dtype=torch.float32, device=self.device)
            weights = weights.unsqueeze(0).unsqueeze(-1)
            weights = weights * attention_mask.unsqueeze(-1).float()
            
            weighted_sum = torch.sum(hidden_states * weights, 1)
            weight_sum = torch.sum(weights, 1)
            return weighted_sum / torch.clamp(weight_sum, min=1e-9)
        
        else:
            # Default to mean pooling
            return self._apply_mean_pooling(hidden_states, attention_mask)
    
    def _apply_mean_pooling(self, hidden_states: torch.Tensor, 
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling"""
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-9)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Include model and config in cache key
        key_data = f"{self.config.model_name}_{self.config.pooling_strategy.value}_{text}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save cache to disk"""
        if self.config.cache_embeddings:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                logger.info(f"Saved {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")
    
    def encode_file(self, file_path: Path, chunk_size: int = 512) -> List[EmbeddingResult]:
        """
        Encode a file by chunks
        
        Args:
            file_path: Path to file
            chunk_size: Size of chunks in tokens
            
        Returns:
            List of embedding results
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = self._split_into_chunks(content, chunk_size)
        
        # Encode chunks
        results = []
        for chunk in chunks:
            result = self.encode(chunk)
            result.metadata['file_path'] = str(file_path)
            results.append(result)
        
        return results
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks based on token count"""
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def compute_similarity(self, 
                          embeddings1: np.ndarray,
                          embeddings2: np.ndarray,
                          metric: str = 'cosine') -> np.ndarray:
        """
        Compute similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity matrix
        """
        if metric == 'cosine':
            # Normalize embeddings
            embeddings1 = self._normalize_embeddings(embeddings1)
            embeddings2 = self._normalize_embeddings(embeddings2)
            return np.dot(embeddings1, embeddings2.T)
        
        elif metric == 'euclidean':
            # Euclidean distance
            return -np.sqrt(np.sum((embeddings1[:, np.newaxis] - embeddings2) ** 2, axis=2))
        
        elif metric == 'dot':
            # Dot product
            return np.dot(embeddings1, embeddings2.T)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def build_index(self, embeddings: np.ndarray, index_type: str = 'flat') -> faiss.Index:
        """
        Build FAISS index for similarity search
        
        Args:
            embeddings: Embeddings to index
            index_type: Type of index ('flat', 'lsh', 'hnsw', 'ivf')
            
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        
        if index_type == 'flat':
            # Exact search
            index = faiss.IndexFlatL2(dimension)
        
        elif index_type == 'lsh':
            # LSH for approximate search
            index = faiss.IndexLSH(dimension, dimension * 2)
        
        elif index_type == 'hnsw':
            # HNSW for approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
        
        elif index_type == 'ivf':
            # IVF for approximate search
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            index.train(embeddings)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        return index
    
    def search(self, 
              query_embedding: np.ndarray,
              index: faiss.Index,
              k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding
            index: FAISS index
            k: Number of results
            
        Returns:
            Distances and indices of nearest neighbors
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = index.search(query_embedding.astype(np.float32), k)
        
        return distances, indices
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics"""
        return {
            'total_encoded': self.stats['total_encoded'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_size': len(self.cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1),
            'model': self.config.model_name,
            'device': str(self.device),
            'pooling': self.config.pooling_strategy.value,
            'max_length': self.config.max_length,
            'normalized': self.config.normalize_embeddings
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache"""
        self.save_cache()

class CodeBERTBatchEncoder:
    """
    Batch encoder for processing large collections of code
    """
    
    def __init__(self, encoder: CodeBERTEncoder):
        """
        Initialize batch encoder
        
        Args:
            encoder: CodeBERT encoder instance
        """
        self.encoder = encoder
        self.results = []
    
    def encode_chunks(self, 
                     chunks: List[Any],
                     content_extractor: callable = lambda x: x.content,
                     batch_size: int = 32,
                     show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of chunks
        
        Args:
            chunks: List of chunks to encode
            content_extractor: Function to extract content from chunk
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings
        """
        # Extract contents
        contents = [content_extractor(chunk) for chunk in chunks]
        
        # Encode
        result = self.encoder.encode(
            contents,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Store results
        self.results.append(result)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, result.embeddings):
            chunk.embedding = embedding
        
        return result.embeddings
    
    def build_search_index(self, 
                          embeddings: np.ndarray,
                          index_type: str = 'hnsw') -> faiss.Index:
        """Build search index from embeddings"""
        return self.encoder.build_index(embeddings, index_type)
    
    def search_similar(self,
                      query: str,
                      index: faiss.Index,
                      chunks: List[Any],
                      k: int = 10) -> List[Tuple[Any, float]]:
        """
        Search for similar chunks
        
        Args:
            query: Query text
            index: FAISS index
            chunks: Original chunks
            k: Number of results
            
        Returns:
            List of (chunk, distance) tuples
        """
        # Encode query
        query_result = self.encoder.encode(query)
        
        # Search
        distances, indices = self.encoder.search(query_result.embeddings, index, k)
        
        # Return chunks with distances
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(chunks):
                results.append((chunks[idx], float(dist)))
        
        return results

# Convenience functions
def create_encoder(model_name: str = "microsoft/codebert-base",
                  use_cuda: bool = True,
                  cache: bool = True) -> CodeBERTEncoder:
    """Create a CodeBERT encoder with common settings"""
    config = EncoderConfig(
        model_name=model_name,
        use_cuda=use_cuda,
        cache_embeddings=cache
    )
    return CodeBERTEncoder(config)

def encode_code(code: Union[str, List[str]], 
               model_name: str = "microsoft/codebert-base") -> np.ndarray:
    """Quick function to encode code"""
    encoder = create_encoder(model_name)
    result = encoder.encode(code)
    return result.embeddings

def compute_code_similarity(code1: str, code2: str,
                          model_name: str = "microsoft/codebert-base") -> float:
    """Compute similarity between two code snippets"""
    encoder = create_encoder(model_name)
    
    embeddings = encoder.encode([code1, code2]).embeddings
    similarity = encoder.compute_similarity(
        embeddings[0:1], 
        embeddings[1:2], 
        metric='cosine'
    )
    
    return float(similarity[0, 0])