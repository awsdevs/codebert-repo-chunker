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
        2. Compute current map `{file_path: checksum}`.
        3. Retrieve stored map from `storage_manager.get_file_checksums(repo_name)`.
        4. Identify `new`, `modified`, `deleted` files.
        5. Call `storage_manager.delete_file_chunks(deleted_files + modified_files)`.
        6. Pass only `new + modified` files to `chunk_processor.process_batch`.

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
