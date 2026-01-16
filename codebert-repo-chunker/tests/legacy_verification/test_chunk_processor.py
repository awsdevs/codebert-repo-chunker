
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import sys
from pathlib import Path
from src.pipeline.chunk_processor import ChunkProcessor, ProcessingConfig

logging.basicConfig(level=logging.INFO)

def test_processor():
    print("Initializing ChunkProcessor...")
    config = ProcessingConfig(enable_embeddings=True)
    processor = ChunkProcessor(config)
    print("ChunkProcessor initialized.")
    
    # Test on a simple file
    test_file = Path("test_sample.py")
    test_file.write_text("def hello():\n    print('hello')")
    
    print("Processing file...")
    chunks = processor.process_file(test_file)
    print(f"Processed {len(chunks)} chunks.")
    
    if chunks:
        print(f"Chunk 0 embedding shape: {chunks[0].embedding.shape if chunks[0].embedding is not None else 'None'}")
        
    test_file.unlink()

if __name__ == "__main__":
    test_processor()
