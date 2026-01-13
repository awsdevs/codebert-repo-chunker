# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration from environment variables"""
    
    # Core
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Repository
    BITBUCKET_USERNAME = os.getenv('BITBUCKET_USERNAME')
    BITBUCKET_APP_PASSWORD = os.getenv('BITBUCKET_APP_PASSWORD')
    REPOSITORY_PATH = Path(os.getenv('REPOSITORY_PATH', '.'))
    
    # Storage
    STORAGE_BASE_PATH = Path(os.getenv('STORAGE_BASE_PATH', './storage'))
    CHUNK_STORAGE_BACKEND = os.getenv('CHUNK_STORAGE_BACKEND', 'sqlite')
    VECTOR_BACKEND = os.getenv('VECTOR_BACKEND', 'faiss')
    
    # Models
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'microsoft/codebert-base')
    EMBEDDING_USE_GPU = os.getenv('EMBEDDING_USE_GPU', 'true').lower() == 'true'
    
    # Processing
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = []
        
        if cls.ENVIRONMENT == 'production':
            required.extend([
                'JWT_SECRET_KEY',
                'API_SECRET_KEY',
                'ENCRYPTION_KEY'
            ])
        
        missing = [var for var in required if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        return True