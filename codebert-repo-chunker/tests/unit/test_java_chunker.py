"""Unit tests for Java chunker"""
import pytest
from transformers import AutoTokenizer
from src.chunkers.code.java_chunker import JavaChunker
from src.core.file_context import FileContext
from pathlib import Path

@pytest.fixture
def tokenizer():
    """Create tokenizer fixture"""
    return AutoTokenizer.from_pretrained("microsoft/codebert-base")

@pytest.fixture
def java_chunker(tokenizer):
    """Create Java chunker fixture"""
    return JavaChunker(tokenizer)

@pytest.fixture
def sample_java_code():
    """Sample Java code for testing"""
    return """
    package com.example.app;
    
    import java.util.List;
    import java.util.ArrayList;
    
    public class UserService {
        private UserRepository repository;
        
        public UserService(UserRepository repository) {
            this.repository = repository;
        }
        
        public List<User> getAllUsers() {
            return repository.findAll();
        }
        
        public User getUserById(Long id) {
            return repository.findById(id);
        }
    }
    """

def test_java_chunker_extracts_package(java_chunker, sample_java_code):
    """Test package extraction"""
    package = java_chunker._extract_package(sample_java_code)
    assert package == "com.example.app"

def test_java_chunker_extracts_imports(java_chunker, sample_java_code):
    """Test import extraction"""
    imports = java_chunker._extract_imports(sample_java_code)
    assert "java.util.List" in imports
    assert "java.util.ArrayList" in imports

def test_java_chunker_creates_chunks(java_chunker, sample_java_code):
    """Test chunk creation"""
    file_context = FileContext(
        path=Path("test.java"),
        extension=".java",
        mime_type="text/x-java",
        file_type="code",
        language="java",
        size_bytes=len(sample_java_code),
        chunking_strategy="java_semantic"
    )
    
    chunks = java_chunker.chunk(sample_java_code, file_context)
    
    assert len(chunks) > 0
    assert any(chunk.chunk_type == "java_class" for chunk in chunks)
    assert any(chunk.chunk_type == "java_method" for chunk in chunks)

def test_java_chunker_handles_empty_content(java_chunker):
    """Test handling of empty content"""
    file_context = FileContext(
        path=Path("empty.java"),
        extension=".java",
        mime_type="text/x-java",
        file_type="code",
        language="java",
        size_bytes=0,
        chunking_strategy="java_semantic"
    )
    
    chunks = java_chunker.chunk("", file_context)
    assert len(chunks) == 0

def test_java_chunker_fallback_on_error(java_chunker):
    """Test fallback chunking on parse error"""
    malformed_code = "public class { invalid java code"
    
    file_context = FileContext(
        path=Path("malformed.java"),
        extension=".java",
        mime_type="text/x-java",
        file_type="code",
        language="java",
        size_bytes=len(malformed_code),
        chunking_strategy="java_semantic"
    )
    
    chunks = java_chunker.chunk(malformed_code, file_context)
    assert len(chunks) > 0
    assert chunks[0].chunk_type == "java_fallback"