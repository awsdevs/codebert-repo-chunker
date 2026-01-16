# Pipeline Debugging and Refinement Walkthrough

This document details the successful debugging of the pipeline's chunk creation issue, improvements to dependency graph visualization, unification of logging, and critical fixes to the storage layer.

## Key Achievements

### 1. Resolved "0 Chunks Created" Issue
**Problem:** The pipeline was running successfully but producing 0 chunks.
**Root Cause:** A `AttributeError: 'dict' object has no attribute 'annotations'` was occurring silently (or masked in some logs) within `GenericCodeChunker` and `JSONChunker`.
**Fix:** Updated `src/chunkers/code/generic_code_chunker.py` and other affected chunkers to use dictionary access pattern.
**Verification:**
- Pipeline re-run successfully created **3043 chunks**.
- "Chunks Created" metric in the summary log now accurately reflects this count.

### 2. Dependency Graph Improvements
**Problem:** The generated dependency graphs were described as "a mess" and reportedly "completely blank".
**Enhancements:**
- **Reliability:** Switched to `cdnjs` for `vis-network` compatibility.
- **Error Handling:** Added explicit on-page error messaging.
- **Visualization:** Implemented physics stabilization, node grouping (sources vs dependencies), and improved layout.
- **Encoding:** Enforced `utf-8` encoding.
**Verification:**
- Valid HTML reports (~170KB) are generated in `reports/`.

### 3. Critical Storage Fixes
**Issue #2 (Metadata Mismatch):**
- **Problem:** `Chunk.to_dict()` created nested `location` dict, but retrieval expected flat keys.
- **Fix:** automatic flattening of `location` fields (file_path, start_line, end_line) into the top-level metadata payload during `StorageManager.store_chunk`.
- **Verification:** SQLite `metadata` table now correctly populates the `file_path` column.

**Issue #3 (Missing Imports):**
- **Problem:** `StorageManager` failed at runtime due to missing `ChunkLocation` and `ChunkType` imports.
- **Fix:** Added missing imports from `src.core.chunk_model`.

**Issue #4 (Resource Leak):**
- **Problem:** `chunk_storage.close()` was never called, leaving file handles open.
- **Fix:** Updated `StorageManager.close()` to correctly call `self.save()` and then close both metadata and chunk stores.

### 4. Codebase Standardization
- **Logging:** Replaced inconsistent logging with `src.utils.logger`.
- **Cleanup:** Removed empty `gradle_chunker.py` and duplicate `logging_config.py`.

## Verification Results

### Pipeline Summary
```text
==================================================
Repository:    codebert-repo-chunker
Total Time:    272.50s
Files Scanned: 104
Chunks Created: 3043
==================================================
```

### Generated Artifacts
- Reports: `reports/` (HTML, JSON, MD)
- Data: `data/metadata.db`, `data/content.db`, `data/vectors/`

## Next Steps
- The pipeline is now fully operational and verified.
- Proceed with using `demo_search.py` for semantic search.

## Robustness & Fixes (Recent Updates)
We implemented critical stability improvements based on user feedback:
1.  **Vector Store (IVF Support)**: Added `_get_underlying_index` to properly handle training when using `IndexIDMap`, preventing crashes with approximate indices.
2.  **Metadata Store (FTS Consistency)**: Modified `store_batch` to delete existing FTS entries before insertion, eliminating duplicate search results.
3.  **Quality Analyzer (Observability)**: Replaced silent exception swallowing with `logger.warning` to aid debugging.
4.  **Verification**: Updated `demo_search.py` to verify FTS functionality.

## Search Capability Refinement
**Objective**: Enhance search to support code content (FTS) and complex queries.

### Improvements
1.  **FTS Indexing**: Updated `MetadataStore` to include code `content` in the `rich_text` FTS index. This enables searching for specific function definitions (e.g., `def run`) or variable names.
2.  **Hybrid Search Verification**: Updated `demo_search.py` to verify:
    *   **Vector Search**: Finding concepts (e.g., "dependency resolution").
    *   **Text Search**: Finding code definitions (e.g., `class MasterPipeline`).

### Verification Result
`demo_search.py` successfully finds:
*   `class MasterPipeline` via FTS (Ranked match).
*   `dependency_resolver` via Vector Search (Concept match).

## Configurable Load Strategy
**Objective**: Allow user to choose between Incremental (Diff-Based) and Full System Load.

### Changes
1.  **Config**: Added `force_full_scan` (boolean) to `config.json`.
2.  **Logic**: 
    *   `False` (Default): Use SHA256 diffs to skip unchanged files.
    *   `True`: Treat all files as "Modified", deleting old chunks and re-processing everything.

### Usage
Modify `config.json`:
```json
"pipeline": {
    "force_full_scan": { "value": true, "description": "..." }
}
```
Then run `run_pipeline.py`.
