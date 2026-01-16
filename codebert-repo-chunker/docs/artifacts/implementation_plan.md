# Implementation Plan: Diff-Based Pipeline Integration

## Goal Description
Implement "smart updates" in the `MasterPipeline` to avoid re-processing files that haven't changed. This will significantly reduce runtime for subsequent runs on large repositories.

## User Review Required
> [!IMPORTANT]
> **Checksum Strategy**: We will compute SHA256 hashes of all files during the scan phase. This adds a small overhead to scanning but saves massive time by skipping chunking/embedding for unchanged files.
> **Deletion Handling**: Files present in the DB but missing/modified on disk will have their OLD chunks deleted immediately before processing new ones.

## Proposed Changes

### [MODIFY] [master_pipeline.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/pipeline/master_pipeline.py)
- **Add Method**: `_compute_file_hash(self, file_path: Path) -> str`
    - simple SHA256 of file content.
- **Update Method**: `run(self, repo_path: Union[str, Path])`
    - logic:
        1. Scan for all files.
        2. Compute current map `{relative_path: checksum}` (Use **Relative Paths** for OS-agnostic persistence).
        3. Retrieve stored map from `storage_manager.get_file_checksums(repo_name)`.
        4. Detect `new`, `modified`, `deleted` files (Language agnostic - works on all scanned files).
        5. **Crash Recovery**: Check identifying marker (e.g. `last_run_status`). If previous run crashed (not "COMPLETED"), log warning and optionally force full re-scan or verify integrity.
        6. Call `storage_manager.delete_file_chunks(deleted_files + modified_files)`.
        7. Pass only `new + modified` files to `chunk_processor.process_batch`.

### [MODIFY] [storage_manager.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/storage_manager.py)
- **Verification**: Ensure `delete_file_chunks` and `get_file_checksums` are exposed and working (Already done in previous phase, will verify).

## Verification Plan

### Automated Tests
- **`test_diff_integration.py`** (New):
    - **Scenario 1**: First Run (all files processed).
    - **Scenario 2**: No Changes (0 files processed).
    - **Scenario 3**: Modify one file (1 file processed, old chunks removed).
    - **Scenario 4**: Delete one file (db cleaned).

### Manual Verification
- Run pipeline twice on the same repo.
    - Run 1: Normal duration.
    - Run 2: Near instant (only scanning overhead).
    - Run 2: Near instant (only scanning overhead).

## Robustness & Fixes (User Feedback)

### [MODIFY] [vector_store.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/vector_store.py)
- **Problem**: `IndexIDMap` wrapper hides `is_trained` and `train()` methods of the underlying IVF index, causing crashes.
- **Fix**:
    - Add `_get_underlying_index(self)` helper.
    - Update `_ensure_trained` and `add` to access the underlying index for training checks.

### [MODIFY] [metadata_store.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/metadata_store.py)
- **Problem**: `store_batch` does not replace existing FTS entries, causing duplicate search results on re-runs.
- **Fix**: Execute `DELETE FROM search_index WHERE chunk_id = ?` before inserting into FTS table in `store_batch`.

### [MODIFY] [quality_analyzer.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/pipeline/quality_analyzer.py)
- **Problem**: Silent exception swallowing (`pass`) makes debugging impossible.
- **Fix**: Replace `pass` with `logger.warning(f"Failed to analyze... {e}")`.

### [MODIFY] [demo_search.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/demo_search.py)
- **Refinement**: Replace generic `Logger` search with a project-specific pattern search (e.g., `ConfigLoader.load_config` usage) to demonstrate structural/pattern search capabilities.

## Code Review Fixes (Phase 5)

### Critical Patches
1. **RelationshipBuilder Integration**
    - [x] **Problem**: `MasterPipeline` ignores relationship building logic.
    - [x] **Fix**: Call `RelationshipBuilder.build_relationships(chunks)` after `ChunkProcessor.process_batch`.
2. **Dependency Consolidation**
    - [x] **Problem**: Three different regex/AST implementations for import extraction.
    - [x] **Fix**: Create `src/utils/import_extractor.py` (Unified AST-based extractor). Refactor `PythonParser`, `RelationshipBuilder`, and `ChunkProcessor` to use it.

### High Priority Fixes
3. **Similarity Optimization**
    - [x] **Problem**: O(n²) loop in `RelationshipBuilder`.
    - [x] **Fix**: Since we have FAISS, use `StorageManager.vector_store` to find nearest neighbors for each chunk, instead of brute-force all-pairs comparison.
4. **Parser Robustness**
    - [x] **Problem**: `except: pass` in parsers hides errors.
    - [x] **Fix**: Add proper error logging in `python_parser.py`, `java_parser.py`, etc.
5. **Quality Analysis Path**
    - [x] **Problem**: `QualityAnalyzer` looks for non-existent pickle files.
    - [x] **Fix**: Update `analyze` to accept list of `Chunk` objects from memory or query DB.
6. **Metadata Consistency**
    - [x] **Problem**: `RelationshipBuilder` accesses missing attributes.
    - [x] **Fix**: Use `getattr(chunk.metadata, ...)` with defaults.
