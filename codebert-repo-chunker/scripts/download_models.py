#!/usr/bin/env python
"""
Model Download Manager for CodeBERT Repository Chunker
Downloads and caches required models for offline operation
"""

import os
import sys
import json
import hashlib
import shutil
import tempfile
import click
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import requests
from tqdm import tqdm
import torch
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig
)
import logging
from urllib.parse import urlparse
import zipfile
import tarfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a model to download"""
    name: str
    model_id: str  # HuggingFace model ID
    description: str
    size_mb: int  # Approximate size in MB
    required: bool
    components: List[str]  # ['model', 'tokenizer', 'config']
    revision: Optional[str] = None  # Specific commit/version
    local_path: Optional[Path] = None
    checksum: Optional[str] = None  # For verification

# Model registry - all models we might need
MODEL_REGISTRY = {
    'codebert-base': ModelInfo(
        name='CodeBERT Base',
        model_id='microsoft/codebert-base',
        description='Base CodeBERT model for code understanding',
        size_mb=500,
        required=True,
        components=['model', 'tokenizer', 'config'],
        revision=None  # Use latest
    ),
    'codebert-mlm': ModelInfo(
        name='CodeBERT MLM',
        model_id='microsoft/codebert-base-mlm',
        description='CodeBERT for masked language modeling',
        size_mb=500,
        required=False,
        components=['model', 'tokenizer', 'config']
    ),
    'graphcodebert': ModelInfo(
        name='GraphCodeBERT',
        model_id='microsoft/graphcodebert-base',
        description='Graph-based CodeBERT for better code structure understanding',
        size_mb=500,
        required=False,
        components=['model', 'tokenizer', 'config']
    ),
    'unixcoder': ModelInfo(
        name='UniXcoder',
        model_id='microsoft/unixcoder-base',
        description='Unified cross-modal pre-trained model',
        size_mb=500,
        required=False,
        components=['model', 'tokenizer', 'config']
    ),
    'codet5': ModelInfo(
        name='CodeT5',
        model_id='Salesforce/codet5-base',
        description='T5-based model for code understanding and generation',
        size_mb=850,
        required=False,
        components=['model', 'tokenizer']
    ),
    'codebert-java': ModelInfo(
        name='CodeBERT Java',
        model_id='huggingface/CodeBERTa-small-v1',
        description='CodeBERT specifically fine-tuned for Java',
        size_mb=300,
        required=False,
        components=['model', 'tokenizer']
    ),
    'starcoder': ModelInfo(
        name='StarCoder Base',
        model_id='bigcode/starcoderbase-1b',
        description='StarCoder for code generation (smaller version)',
        size_mb=2000,
        required=False,
        components=['model', 'tokenizer']
    )
}

class ModelDownloader:
    """Manages model downloading and caching"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize model downloader
        
        Args:
            cache_dir: Directory to cache models, defaults to ~/.cache/huggingface
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            os.environ['HF_HOME'] = str(cache_dir)
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        else:
            # Use default HuggingFace cache
            self.cache_dir = Path.home() / '.cache' / 'huggingface'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Local model storage (for our project)
        self.local_models_dir = Path(__file__).parent.parent / 'models'
        self.local_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Track download statistics
        self.stats = {
            'downloaded': [],
            'skipped': [],
            'failed': [],
            'total_size_mb': 0
        }
    
    def check_disk_space(self, required_mb: int) -> bool:
        """Check if enough disk space is available
        
        Args:
            required_mb: Required space in megabytes
            
        Returns:
            True if enough space available
        """
        import shutil
        
        stat = shutil.disk_usage(self.cache_dir)
        available_mb = stat.free / (1024 * 1024)
        
        if available_mb < required_mb * 1.5:  # 50% buffer
            logger.warning(
                f"Low disk space: {available_mb:.0f}MB available, "
                f"{required_mb:.0f}MB required"
            )
            return False
        
        return True
    
    def check_internet_connection(self) -> bool:
        """Check if internet connection is available"""
        try:
            response = requests.get('https://huggingface.co', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info_from_hub(self, model_id: str) -> Dict[str, Any]:
        """Get model information from HuggingFace Hub
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Model metadata from hub
        """
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            model_info = api.model_info(model_id)
            
            return {
                'model_id': model_info.modelId,
                'sha': model_info.sha,
                'last_modified': model_info.lastModified,
                'size_bytes': getattr(model_info, 'size', None),
                'downloads': getattr(model_info, 'downloads', 0),
                'tags': model_info.tags
            }
        except Exception as e:
            logger.warning(f"Could not fetch hub info for {model_id}: {e}")
            return {}
    
    def is_model_cached(self, model_info: ModelInfo) -> bool:
        """Check if model is already cached
        
        Args:
            model_info: Model information
            
        Returns:
            True if model is cached and valid
        """
        try:
            # Check if model exists in transformers cache
            if 'model' in model_info.components:
                config = AutoConfig.from_pretrained(
                    model_info.model_id,
                    local_files_only=True,
                    cache_dir=self.cache_dir
                )
            
            if 'tokenizer' in model_info.components:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info.model_id,
                    local_files_only=True,
                    cache_dir=self.cache_dir
                )
            
            # Check local copy
            local_path = self.local_models_dir / model_info.model_id.replace('/', '_')
            if local_path.exists():
                model_info.local_path = local_path
                
            return True
            
        except Exception:
            return False
    
    def download_with_progress(self, url: str, dest_path: Path, 
                             description: str = "Downloading") -> bool:
        """Download file with progress bar
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            description: Progress bar description
            
        Returns:
            True if successful
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f:
                with tqdm(
                    desc=description,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    def download_model(self, model_info: ModelInfo, force: bool = False) -> bool:
        """Download a model and its components
        
        Args:
            model_info: Model information
            force: Force re-download even if cached
            
        Returns:
            True if successful
        """
        # Check if already cached
        if not force and self.is_model_cached(model_info):
            logger.info(f"✓ {model_info.name} already cached")
            self.stats['skipped'].append(model_info.name)
            return True
        
        # Check disk space
        if not self.check_disk_space(model_info.size_mb):
            logger.error(f"Insufficient disk space for {model_info.name}")
            self.stats['failed'].append(model_info.name)
            return False
        
        logger.info(f"Downloading {model_info.name} ({model_info.size_mb}MB)...")
        
        try:
            # Download model components
            if 'model' in model_info.components:
                logger.info(f"  Downloading model weights...")
                model = AutoModel.from_pretrained(
                    model_info.model_id,
                    cache_dir=self.cache_dir,
                    revision=model_info.revision,
                    resume_download=True,  # Resume if interrupted
                    force_download=force
                )
                
                # Save local copy
                local_path = self.local_models_dir / model_info.model_id.replace('/', '_')
                local_path.mkdir(parents=True, exist_ok=True)
                
                model.save_pretrained(local_path)
                logger.info(f"  ✓ Model saved to {local_path}")
            
            if 'tokenizer' in model_info.components:
                logger.info(f"  Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info.model_id,
                    cache_dir=self.cache_dir,
                    revision=model_info.revision,
                    resume_download=True,
                    force_download=force
                )
                
                # Save local copy
                local_path = self.local_models_dir / model_info.model_id.replace('/', '_')
                tokenizer.save_pretrained(local_path)
                logger.info(f"  ✓ Tokenizer saved")
            
            if 'config' in model_info.components:
                logger.info(f"  Downloading configuration...")
                config = AutoConfig.from_pretrained(
                    model_info.model_id,
                    cache_dir=self.cache_dir,
                    revision=model_info.revision,
                    resume_download=True,
                    force_download=force
                )
                
                # Save local copy
                local_path = self.local_models_dir / model_info.model_id.replace('/', '_')
                config.save_pretrained(local_path)
                logger.info(f"  ✓ Configuration saved")
            
            self.stats['downloaded'].append(model_info.name)
            self.stats['total_size_mb'] += model_info.size_mb
            
            logger.info(f"✓ {model_info.name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_info.name}: {e}")
            self.stats['failed'].append(model_info.name)
            return False
    
    def verify_model(self, model_info: ModelInfo) -> bool:
        """Verify model is correctly downloaded and functional
        
        Args:
            model_info: Model information
            
        Returns:
            True if model is valid
        """
        logger.info(f"Verifying {model_info.name}...")
        
        try:
            # Try loading from local copy
            local_path = self.local_models_dir / model_info.model_id.replace('/', '_')
            
            if local_path.exists():
                # Load and test model
                if 'model' in model_info.components:
                    model = AutoModel.from_pretrained(
                        local_path,
                        local_files_only=True
                    )
                    
                    # Test forward pass with dummy input
                    if 'tokenizer' in model_info.components:
                        tokenizer = AutoTokenizer.from_pretrained(
                            local_path,
                            local_files_only=True
                        )
                        
                        # Test encoding
                        test_text = "def hello_world():\n    print('Hello')"
                        inputs = tokenizer(
                            test_text, 
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )
                        
                        # Test forward pass
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        logger.info(f"  ✓ Model functional (output shape: {outputs.last_hidden_state.shape})")
                
                return True
                
        except Exception as e:
            logger.error(f"Verification failed for {model_info.name}: {e}")
            return False
    
    def download_additional_resources(self):
        """Download additional resources like vocabularies, configs, etc."""
        resources_dir = self.local_models_dir / 'resources'
        resources_dir.mkdir(parents=True, exist_ok=True)
        
        # Download language-specific tokenizers or configs if needed
        additional_resources = {
            'java_keywords.txt': 'https://raw.githubusercontent.com/microsoft/CodeBERT/master/resources/java_keywords.txt',
            'python_keywords.txt': 'https://raw.githubusercontent.com/microsoft/CodeBERT/master/resources/python_keywords.txt',
        }
        
        for filename, url in additional_resources.items():
            dest_path = resources_dir / filename
            
            if dest_path.exists():
                logger.info(f"  ✓ {filename} already exists")
                continue
            
            logger.info(f"  Downloading {filename}...")
            if self.download_with_progress(url, dest_path, f"Downloading {filename}"):
                logger.info(f"  ✓ {filename} downloaded")
    
    def create_offline_config(self):
        """Create configuration file for offline operation"""
        config = {
            'offline_mode': True,
            'model_cache_dir': str(self.cache_dir),
            'local_models_dir': str(self.local_models_dir),
            'models': {}
        }
        
        # Add model information
        for key, model_info in MODEL_REGISTRY.items():
            if model_info.name in self.stats['downloaded'] or \
               model_info.name in self.stats['skipped']:
                local_path = self.local_models_dir / model_info.model_id.replace('/', '_')
                config['models'][key] = {
                    'model_id': model_info.model_id,
                    'local_path': str(local_path),
                    'available': local_path.exists()
                }
        
        # Save configuration
        config_path = Path(__file__).parent.parent / 'config' / 'offline_config.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Offline configuration saved to {config_path}")
        
        return config_path
    
    def cleanup_cache(self, keep_latest: bool = True):
        """Clean up old cached files to save space
        
        Args:
            keep_latest: Keep only the latest version of each model
        """
        logger.info("Cleaning up cache...")
        
        # Use huggingface_hub utilities for safe cleanup
        try:
            from huggingface_hub import scan_cache_dir
            
            cache_info = scan_cache_dir(self.cache_dir)
            
            # Get size before cleanup
            size_before = sum(repo.size_on_disk for repo in cache_info.repos)
            
            # Delete old revisions
            for repo in cache_info.repos:
                if len(repo.revisions) > 1 and keep_latest:
                    # Keep only the latest revision
                    revisions_to_delete = sorted(
                        repo.revisions, 
                        key=lambda x: x.last_modified
                    )[:-1]
                    
                    for revision in revisions_to_delete:
                        revision.delete()
            
            # Get size after cleanup
            cache_info = scan_cache_dir(self.cache_dir)
            size_after = sum(repo.size_on_disk for repo in cache_info.repos)
            
            freed_mb = (size_before - size_after) / (1024 * 1024)
            logger.info(f"Freed {freed_mb:.0f}MB of disk space")
            
        except ImportError:
            logger.warning("huggingface_hub not available, skipping cache cleanup")
    
    def print_summary(self):
        """Print download summary"""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        
        if self.stats['downloaded']:
            print(f"\n✓ Downloaded ({len(self.stats['downloaded'])}):")
            for name in self.stats['downloaded']:
                print(f"  - {name}")
        
        if self.stats['skipped']:
            print(f"\n⊙ Already cached ({len(self.stats['skipped'])}):")
            for name in self.stats['skipped']:
                print(f"  - {name}")
        
        if self.stats['failed']:
            print(f"\n✗ Failed ({len(self.stats['failed'])}):")
            for name in self.stats['failed']:
                print(f"  - {name}")
        
        print(f"\nTotal downloaded: {self.stats['total_size_mb']:.0f}MB")
        print(f"Models directory: {self.local_models_dir}")
        print("="*60)

class ModelValidator:
    """Validate all models are correctly installed"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.validation_results = {}
    
    def validate_all(self) -> Dict[str, bool]:
        """Validate all models in the registry"""
        
        for key, model_info in MODEL_REGISTRY.items():
            logger.info(f"Validating {model_info.name}...")
            
            local_path = self.models_dir / model_info.model_id.replace('/', '_')
            
            if not local_path.exists():
                self.validation_results[key] = False
                logger.warning(f"  ✗ {model_info.name} not found")
                continue
            
            try:
                # Check all required files exist
                required_files = []
                
                if 'model' in model_info.components:
                    required_files.extend([
                        'pytorch_model.bin',
                        'model.safetensors',  # Newer format
                        'config.json'
                    ])
                
                if 'tokenizer' in model_info.components:
                    required_files.extend([
                        'tokenizer_config.json',
                        'vocab.json',
                        'tokenizer.json',
                        'vocab.txt',  # For BERT-based models
                        'merges.txt'  # For BPE tokenizers
                    ])
                
                # Check if at least the essential files exist
                has_model = (local_path / 'pytorch_model.bin').exists() or \
                           (local_path / 'model.safetensors').exists()
                has_config = (local_path / 'config.json').exists()
                has_tokenizer = (local_path / 'tokenizer_config.json').exists() or \
                               (local_path / 'vocab.txt').exists()
                
                if 'model' in model_info.components and not has_model:
                    logger.warning(f"  ✗ Model weights not found")
                    self.validation_results[key] = False
                    continue
                
                if 'tokenizer' in model_info.components and not has_tokenizer:
                    logger.warning(f"  ✗ Tokenizer not found")
                    self.validation_results[key] = False
                    continue
                
                logger.info(f"  ✓ {model_info.name} validated")
                self.validation_results[key] = True
                
            except Exception as e:
                logger.error(f"  ✗ Validation error: {e}")
                self.validation_results[key] = False
        
        return self.validation_results
    
    def print_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("MODEL VALIDATION REPORT")
        print("="*60)
        
        required_models = [k for k, v in MODEL_REGISTRY.items() if v.required]
        optional_models = [k for k, v in MODEL_REGISTRY.items() if not v.required]
        
        print("\nRequired Models:")
        for key in required_models:
            status = "✓" if self.validation_results.get(key, False) else "✗"
            print(f"  {status} {MODEL_REGISTRY[key].name}")
        
        print("\nOptional Models:")
        for key in optional_models:
            status = "✓" if self.validation_results.get(key, False) else "○"
            print(f"  {status} {MODEL_REGISTRY[key].name}")
        
        # Check if all required models are valid
        all_required_valid = all(
            self.validation_results.get(k, False) 
            for k in required_models
        )
        
        if all_required_valid:
            print("\n✅ All required models are properly installed!")
        else:
            print("\n⚠️  Some required models are missing or invalid!")
            print("Run with --force to re-download")
        
        print("="*60)

@click.command()
@click.option('--models', '-m', multiple=True, 
              help='Specific models to download (default: all required)')
@click.option('--all', 'download_all', is_flag=True, 
              help='Download all models including optional ones')
@click.option('--force', '-f', is_flag=True, 
              help='Force re-download even if cached')
@click.option('--cache-dir', type=click.Path(), 
              help='Custom cache directory')
@click.option('--verify', '-v', is_flag=True, 
              help='Verify models after download')
@click.option('--cleanup', is_flag=True, 
              help='Clean up old model versions')
@click.option('--check-only', is_flag=True, 
              help='Only check what models are installed')
@click.option('--offline-config', is_flag=True, 
              help='Generate offline configuration file')
def main(models: Tuple[str], download_all: bool, force: bool, 
         cache_dir: Optional[str], verify: bool, cleanup: bool,
         check_only: bool, offline_config: bool):
    """Download and manage models for CodeBERT Repository Chunker"""
    
    print("="*60)
    print("CodeBERT Model Download Manager")
    print("="*60)
    
    # Initialize downloader
    downloader = ModelDownloader(cache_dir=Path(cache_dir) if cache_dir else None)
    
    # Check-only mode
    if check_only:
        validator = ModelValidator(downloader.local_models_dir)
        validator.validate_all()
        validator.print_report()
        return
    
    # Check internet connection
    if not downloader.check_internet_connection():
        logger.error("No internet connection available!")
        print("\n⚠️  Internet connection required for downloading models")
        print("If models are already downloaded, use --check-only to verify")
        sys.exit(1)
    
    # Determine which models to download
    models_to_download = []
    
    if models:
        # Download specific models
        for model_key in models:
            if model_key in MODEL_REGISTRY:
                models_to_download.append(MODEL_REGISTRY[model_key])
            else:
                logger.warning(f"Unknown model: {model_key}")
                print(f"\nAvailable models:")
                for key, info in MODEL_REGISTRY.items():
                    print(f"  {key}: {info.name}")
    elif download_all:
        # Download all models
        models_to_download = list(MODEL_REGISTRY.values())
    else:
        # Download only required models
        models_to_download = [m for m in MODEL_REGISTRY.values() if m.required]
    
    if not models_to_download:
        logger.error("No models selected for download")
        sys.exit(1)
    
    # Calculate total size
    total_size_mb = sum(m.size_mb for m in models_to_download)
    print(f"\nModels to download: {len(models_to_download)}")
    print(f"Estimated size: {total_size_mb:.0f}MB")
    
    # Check disk space
    if not downloader.check_disk_space(total_size_mb):
        if not click.confirm("\n⚠️  Low disk space! Continue anyway?"):
            sys.exit(1)
    
    # Download models
    print("\nDownloading models...")
    for model_info in models_to_download:
        success = downloader.download_model(model_info, force=force)
        
        if not success and model_info.required:
            logger.error(f"Failed to download required model: {model_info.name}")
            if not click.confirm("Continue without this model?"):
                sys.exit(1)
    
    # Download additional resources
    print("\nDownloading additional resources...")
    downloader.download_additional_resources()
    
    # Verify models if requested
    if verify:
        print("\nVerifying models...")
        validator = ModelValidator(downloader.local_models_dir)
        
        for model_info in models_to_download:
            if downloader.verify_model(model_info):
                print(f"  ✓ {model_info.name} verified")
            else:
                print(f"  ✗ {model_info.name} verification failed")
    
    # Cleanup old versions if requested
    if cleanup:
        downloader.cleanup_cache()
    
    # Create offline configuration
    if offline_config or download_all:
        config_path = downloader.create_offline_config()
        print(f"\nOffline configuration saved to: {config_path}")
    
    # Print summary
    downloader.print_summary()
    
    # Final validation
    print("\nRunning final validation...")
    validator = ModelValidator(downloader.local_models_dir)
    validator.validate_all()
    validator.print_report()
    
    print("\n✅ Model setup complete!")
    print("\nTo test the models, run:")
    print("  python scripts/test_models.py")
    print("\nTo start processing repositories, run:")
    print("  python scripts/process_repo.py /path/to/repo")

if __name__ == '__main__':
    main()