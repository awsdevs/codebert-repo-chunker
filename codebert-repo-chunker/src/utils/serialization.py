"""
Serialization utilities for chunks and other data structures
Supports multiple formats with compression and validation
"""

import json
import pickle
import msgpack
import yaml
import struct
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, date
from pathlib import Path
import numpy as np
import base64
import zlib
import hashlib
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SerializationFormat(Enum):
    """Supported serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    YAML = "yaml"
    PROTOBUF = "protobuf"
    PARQUET = "parquet"
    BINARY = "binary"

class ChunkSerializer:
    """Serializer for chunk objects with multiple format support"""
    
    def __init__(self, format: SerializationFormat = SerializationFormat.JSON):
        """
        Initialize serializer
        
        Args:
            format: Serialization format to use
        """
        self.format = format
        self.encoders = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.PICKLE: self._serialize_pickle,
            SerializationFormat.MSGPACK: self._serialize_msgpack,
            SerializationFormat.YAML: self._serialize_yaml,
            SerializationFormat.BINARY: self._serialize_binary
        }
        
        self.decoders = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.PICKLE: self._deserialize_pickle,
            SerializationFormat.MSGPACK: self._deserialize_msgpack,
            SerializationFormat.YAML: self._deserialize_yaml,
            SerializationFormat.BINARY: self._deserialize_binary
        }
    
    def serialize(self, obj: Any, compress: bool = False) -> bytes:
        """
        Serialize object to bytes
        
        Args:
            obj: Object to serialize
            compress: Whether to compress the output
            
        Returns:
            Serialized bytes
        """
        # Convert to serializable format
        serializable = self._make_serializable(obj)
        
        # Serialize using selected format
        encoder = self.encoders.get(self.format)
        if not encoder:
            raise ValueError(f"Unsupported format: {self.format}")
        
        data = encoder(serializable)
        
        # Compress if requested
        if compress:
            data = zlib.compress(data)
        
        return data
    
    def deserialize(self, data: bytes, compressed: bool = False) -> Any:
        """
        Deserialize bytes to object
        
        Args:
            data: Serialized bytes
            compressed: Whether the data is compressed
            
        Returns:
            Deserialized object
        """
        # Decompress if needed
        if compressed:
            data = zlib.decompress(data)
        
        # Deserialize using selected format
        decoder = self.decoders.get(self.format)
        if not decoder:
            raise ValueError(f"Unsupported format: {self.format}")
        
        obj = decoder(data)
        
        # Convert back to original types
        return self._restore_types(obj)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to serializable format"""
        if obj is None:
            return None
        
        # Handle dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                '__dataclass__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__,
                'data': self._make_serializable(asdict(obj))
            }
        
        # Handle datetime
        if isinstance(obj, (datetime, date)):
            return {
                '__datetime__': True,
                'value': obj.isoformat()
            }
        
        # Handle Path
        if isinstance(obj, Path):
            return {
                '__path__': True,
                'value': str(obj)
            }
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'shape': obj.shape,
                'data': base64.b64encode(obj.tobytes()).decode('ascii')
            }
        
        # Handle Enum
        if isinstance(obj, Enum):
            return {
                '__enum__': True,
                'class': obj.__class__.__name__,
                'module': obj.__class__.__module__,
                'value': obj.value
            }
        
        # Handle dict
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        
        # Handle list/tuple
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        
        # Handle set
        if isinstance(obj, set):
            return {
                '__set__': True,
                'values': [self._make_serializable(item) for item in obj]
            }
        
        # Basic types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Try to convert to dict
        if hasattr(obj, 'to_dict'):
            return {
                '__custom__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__,
                'data': obj.to_dict()
            }
        
        # Fallback to string representation
        return str(obj)
    
    def _restore_types(self, obj: Any) -> Any:
        """Restore original types from serializable format"""
        if obj is None:
            return None
        
        if isinstance(obj, dict):
            # Check for special type markers
            if '__dataclass__' in obj:
                # Restore dataclass (would need the class definition)
                return self._restore_types(obj['data'])
            
            if '__datetime__' in obj:
                return datetime.fromisoformat(obj['value'])
            
            if '__path__' in obj:
                return Path(obj['value'])
            
            if '__numpy__' in obj:
                data = base64.b64decode(obj['data'].encode('ascii'))
                return np.frombuffer(data, dtype=obj['dtype']).reshape(obj['shape'])
            
            if '__enum__' in obj:
                # Would need to import the enum class
                return obj['value']
            
            if '__set__' in obj:
                return set(self._restore_types(obj['values']))
            
            if '__custom__' in obj:
                # Would need the class definition to properly restore
                return self._restore_types(obj['data'])
            
            # Regular dict
            return {k: self._restore_types(v) for k, v in obj.items()}
        
        if isinstance(obj, list):
            return [self._restore_types(item) for item in obj]
        
        return obj
    
    def _serialize_json(self, obj: Any) -> bytes:
        """Serialize to JSON"""
        return json.dumps(obj, ensure_ascii=False, indent=None).encode('utf-8')
    
    def _deserialize_json(self, data: bytes) -> Any:
        """Deserialize from JSON"""
        return json.loads(data.decode('utf-8'))
    
    def _serialize_pickle(self, obj: Any) -> bytes:
        """Serialize to pickle"""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_pickle(self, data: bytes) -> Any:
        """Deserialize from pickle"""
        return pickle.loads(data)
    
    def _serialize_msgpack(self, obj: Any) -> bytes:
        """Serialize to MessagePack"""
        return msgpack.packb(obj, use_bin_type=True)
    
    def _deserialize_msgpack(self, data: bytes) -> Any:
        """Deserialize from MessagePack"""
        return msgpack.unpackb(data, raw=False)
    
    def _serialize_yaml(self, obj: Any) -> bytes:
        """Serialize to YAML"""
        return yaml.safe_dump(obj).encode('utf-8')
    
    def _deserialize_yaml(self, data: bytes) -> Any:
        """Deserialize from YAML"""
        return yaml.safe_load(data.decode('utf-8'))
    
    def _serialize_binary(self, obj: Any) -> bytes:
        """Serialize to custom binary format"""
        # Custom binary serialization for efficiency
        # This would be implemented based on specific needs
        return pickle.dumps(obj)
    
    def _deserialize_binary(self, data: bytes) -> Any:
        """Deserialize from custom binary format"""
        return pickle.loads(data)