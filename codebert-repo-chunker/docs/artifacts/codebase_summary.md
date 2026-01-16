# Codebase Summary & Usecase Analysis

## 🎯 Usecase Overview

**Project Name**: CodeBERT Repo Chunker (likely part of a larger Semantic Code Search system)

**Primary Goal**: To process large software repositories, break them down into semantic units ("chunks"), generate vector embeddings using **CodeBERT**, and enable advanced semantic search and relationship analysis.

**Key Features**:
1.  **Intelligent Ingestion**: Scans repositories, filtering and prioritizing files based on importance and domain (e.g., separating "Application Logic" from "Tests" or "Vendor" code).
2.  **Semantic Chunking**: Instead of naive text splitting, it uses language-aware parsers (AST, regex) to split code into meaningful blocks (Classes, Functions, Methods).
3.  **Multi-Dimensional Classification**: Classifies files by **Category** (Source, Config), **Domain** (UI, Infrastructure), and **Tech Stack** (Django, React, etc.).
4.  **Vector Embedding**: Uses Microsoft's **CodeBERT** (or compatible models) to convert code chunks into high-dimensional vectors for similarity search.
5.  **Graph Building**: Establishes relationships between chunks (Imports, Calls, Inheritance) to build a code knowledge graph.

## ⚠️ Critical Findings

The codebase appears to be an **incomplete refactor or work-in-progress**.
-   **Broken Pipeline**: The `MasterPipeline` imports `RepositoryScanner`, `DependencyResolver`, etc., which **do not exist** in the codebase.
-   **Architecture Mismatch**: The complex orchestrator (`MasterPipeline`) assumes a distributed, full-featured system, while the actual working components (`ChunkProcessor`, `StorageManager`) are more modest.

---

## 📂 File Summaries

### `src/pipeline/`
*   **`master_pipeline.py`**: The intended orchestrator. Defines detailed states (`PipelineStatus`), stages, and configurations. **Currently Broken** due to missing dependencies.
*   **`chunk_processor.py`**: The core worker. Manages the flow of taking files -> classifying -> chunking -> embedding -> storing. Implements parallel processing (Thread/Process pools).
*   **`relationship_builder.py`**: A complex module that analyzes code chunks to find relationships like imports, function calls, and class inheritance using AST and regex.

### `src/core/`
*   **`chunk_model.py`**: Defines the data model (`Chunk`, `ChunkMetadata`, `ChunkRelation`). Supports rich metadata, version history, and embedding vectors.
*   **`base_chunker.py`**: Abstract base class for all chunkers. details token counting, validation, and common text splitting logic.
*   **`file_context.py`**: Dataclass holding file-level context (path, content, language) used during processing.

### `src/chunkers/`
*   **`registry.py`**: A sophisticated `ChunkerRegistry` that auto-selects the right chunker implementation based on file extension, pattern, or content analysis. Supports dynamic registration.

### `src/classifiers/`
*   **`file_classifier.py`**: A rule-based engine that tags files with:
    *   **Category**: Source, Test, Config, etc.
    *   **Domain**: Application, Infrastructure, Presentation.
    *   **Purpose**: Model, Controller, Repository.
    *   **Importance**: A 1-5 score used to prioritize processing.
*   **`content_analyzer.py`** & **`pattern_detector.py`** (Inferring context): Likely helper modules for deep content inspection (e.g., detecting sensitive data or design patterns).

### `src/embeddings/`
*   **`codebert_encoder.py`**: A wrapper around HuggingFace Transformers. Handles:
    *   Loading CodeBERT (and other models).
    *   Tokenization and Pooling (Mean, Max, CLS).
    *   Caching embeddings to disk.
    *   ONNX optimization support.

### `src/storage/`
*   **`storage_manager.py`**: A facade that unifies the three storage backends.
*   **`chunk_storage.py`**: (Inferred) Stores raw text and metadata (SQL/FileSystem).
*   **`metadata_store.py`**: (Inferred) Stores structured metadata.
*   **`vector_store.py`**: (Inferred) Wraps FAISS or similar for vector similarity search.

### `src/utils/`
*   **`file_utils.py`, `text_utils.py`**: General I/O and string manipulation helpers.
*   **`metrics.py`, `monitoring.py`**: Prometheus/Logging integration for observing pipeline health.

---

## 🚀 Recommendation

To make this functional:
1.  **Fix the Pipeline**: Either implement the missing `src/pipeline/*.py` modules or simplify `MasterPipeline` to use the existing `ChunkProcessor` directly.
2.  **Test**: Add integration tests to `tests/integration/test_pipeline.py` to verify the flow.
