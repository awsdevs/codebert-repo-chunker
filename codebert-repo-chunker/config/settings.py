"""Configuration settings for the chunker"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Central configuration management"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    OUTPUT_DIR = BASE_DIR / "output"
    CONFIG_DIR = BASE_DIR / "config"
    
    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/codebert-base")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "450"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    
    # Processing settings
    SKIP_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.exe', '.dll', '.so'}
    SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'target', 'build', 'dist'}
    
    # Storage settings
    VECTOR_STORE_PATH = OUTPUT_DIR / "vector_store"
    CHUNK_STORE_PATH = OUTPUT_DIR / "chunks"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = OUTPUT_DIR / "logs" / "chunker.log"
    
    @classmethod
    def load_file_mappings(cls) -> Dict[str, Any]:
        """Load file type mappings from YAML"""
        mapping_file = cls.CONFIG_DIR / "file_mappings.yaml"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.OUTPUT_DIR, cls.CHUNK_STORE_PATH, 
                         cls.VECTOR_STORE_PATH, cls.LOG_FILE.parent]:
            dir_path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()