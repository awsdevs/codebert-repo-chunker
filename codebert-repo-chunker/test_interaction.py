
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import sys
import torch # Pre-load torch

logging.basicConfig(level=logging.INFO)

def test_interaction():
    print("1. Importing modules...")
    try:
        from src.classifiers.file_classifier import FileClassifier
        print("   FileClassifier imported.")
    except Exception as e:
        print(f"   FileClassifier import failed: {e}")

    try:
        from src.storage.storage_manager import StorageManager
        print("   StorageManager imported.")
    except Exception as e:
        print(f"   StorageManager import failed: {e}")

    try:
        from src.embeddings.codebert_encoder import CodeBERTEncoder, EncoderConfig
        print("   CodeBERTEncoder imported.")
    except Exception as e:
        print(f"   CodeBERTEncoder import failed: {e}")

    print("\n2. Initializing components...")
    
    print("   Init FileClassifier...")
    fc = FileClassifier()
    print("   FileClassifier initialized.")

    print("   Init CodeBERTEncoder...")
    # This is where it crashed in test_chunk_processor.py
    config = EncoderConfig(use_cuda=False)
    encoder = CodeBERTEncoder(config)
    print("   CodeBERTEncoder initialized.")

    print("   Init StorageManager (Mock config)...")
    # Need to verify if StorageManager init triggers crash
    # But ChunkProcessor only imports it, doesn't init it unless passed.
    # In test_chunk_processor.py, we didn't pass StorageManager.
    # So StorageManager was NOT initialized.
    
    print("\nSuccess! No crash.")

if __name__ == "__main__":
    test_interaction()
