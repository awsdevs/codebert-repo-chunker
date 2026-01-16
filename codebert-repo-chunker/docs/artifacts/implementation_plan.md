# Implementation Plan: Performance, Robustness, and Integration

## Goal Description
Address critical performance bottlenecks (n=1 inserts), robustness issues (missing indexes, O(n) lookups), and integration gaps identified in the recent code review. The goal is to prepare the system for processing 150+ repositories efficiently and reliably.

## User Review Required
> [!IMPORTANT]
> **VectorStore Logic Changes**: We will introduce a `reverse_map` to `VectorStore` to make deletion O(1). This changes the `save/load` format slightly (though we can rebuild it on load).
> **Search Score Interpretation**: We will clarify that scores are now Cosine Similarity (-1 to 1) rather than L2 distance.

## Proposed Changes

### Phase 3: Performance & Scale (High Priority)

#### [MODIFY] [chunk_storage.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/chunk_storage.py)
- Implement `store_batch(self, chunks)`:
    - Accept list of chunk data.
    - Use `executemany` for high-throughput inserts.
    - Catch and log errors at batch level.

#### [MODIFY] [metadata_store.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/metadata_store.py)
- Implement `store_batch(self, metadata_list)`:
    - Use `executemany` for metadata.
    - Optimize `_build_rich_text` calls within the batch loop.

#### [MODIFY] [vector_store.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/vector_store.py)
- **Optimize Removal (O(n) -> O(1))**:
    - Add `self.reverse_map: Dict[str, int]` (Chunk ID -> FAISS ID).
    - Update `add_with_ids` to populate `reverse_map`.
    - Update `remove` to use `reverse_map` for instant lookup.
    - Update `save`/`load` to persist or rebuild `reverse_map`.
- **Fix IVF Removal**:
    - Ensure `IndexIVF` is wrapped with `IndexIDMap` to support `remove_ids`.

#### [MODIFY] [storage_manager.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/storage_manager.py)
- Implement `store_chunks_batch(self, chunks)`:
    - Orchestrate batch calls to all 3 stores.
    - Ensure atomicity where possible (or at least consistent ordering).

#### [MODIFY] [chunk_processor.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/pipeline/chunk_processor.py)
- Update `process_file` and `process_batch` to accumulate chunks and call `storage_manager.store_chunks_batch`.

---

### Phase 4: Robustness & Data Integrity (Medium Priority)

#### [MODIFY] [metadata_store.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/metadata_store.py)
- **Add Indexes**:
    - `CREATE INDEX idx_meta_filepath ON metadata(file_path)`
    - `CREATE INDEX idx_meta_repository ON metadata(repository)`
- **Enhance FTS**:
    - Update `_build_rich_text` to include `chunk_type`, `class_name`, `language`, `package_name`.

#### [MODIFY] [master_pipeline.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/pipeline/master_pipeline.py)
- **Fix Double Reporting**:
    - Guard `self._generate_reports` in the `finally` block to only run if not already completed.

#### [MODIFY] [storage_manager.py](file:///Users/sai/saiData/codebert_google/codebert_google/codebert-repo-chunker/src/storage/storage_manager.py)
- **Flatten File Checksum**:
    - Ensure `file_checksum` is extracted from `location` dict and stored at top level of metadata for easier access.

---

### Phase 5: Refinement (Low Priority)
- **Quality Analyzer**: Add logger to exception handler.
- **Documentation**: Update docstrings to reflect "Cosine Similarity" instead of Distance.

## Verification Plan

### Automated Tests
- **`test_batch_ops.py`** (New):
    - Verify `store_batch` works for all stores.
    - Verify performance improvement (time 1000 inserts one-by-one vs batch).
- **`test_vector_optimization.py`** (New):
    - Verify `reverse_map` works correctly after add/remove/reload.
    - Verify IVF index deletion (optional, if we switch to IVF).
- **Regression**: Run `test_diff_updates.py` to ensure single-item operations still work.

### Manual Verification
- Run `run_pipeline.py` and check `pipeline_output_final.txt` for:
    - No duplicate report generation logs.
    - Faster execution time (Performance benchmark).
    - **Dependency Graph Validation**: Analyze nodes in `dependency_graph_*.html` to ensure they are valid and the report is correct.
