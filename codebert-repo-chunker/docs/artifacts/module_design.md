# Design Document: Missing Pipeline Modules

## 1. Goal
To implement the missing components required by `MasterPipeline` to restore full functionality to the `codebert-repo-chunker` system.

## 2. Missing Components
Based on `src/pipeline/master_pipeline.py`, the following modules are missing:
1.  `RepositoryScanner`
2.  `DependencyResolver`
3.  `QualityAnalyzer`
4.  `ReportGenerator`

## 3. Proposed Design

### A. RepositoryScanner
**Responsibility**: Efficiently traverse the filesystem, respecting ignore rules (like `.gitignore`), and identifying processable files.

**Integration**: Used by `MasterPipeline` to generate the stream of files for the `ChunkProcessor`.

**Proposed Interface**:
```python
class RepositoryScanner:
    def __init__(self, config: ScannerConfig):
        self.ignore_patterns: List[str] = self._load_ignore_patterns()
        self.classifier = FileClassifier()

    def scan(self, root_path: Path) -> Iterator[Path]:
        """Yields all valid file paths in the repository."""
        pass

    def scan_files(self, root_path: Path) -> Iterator[Tuple[Path, FileClassification]]:
        """
        Yields paths along with their initial classification.
        This allows the pipeline to skip early (e.g. ignore binary files or low importance).
        """
        pass
```

### B. DependencyResolver
**Responsibility**: Analyze project-level manifests (`requirements.txt`, `package.json`, `pom.xml`, etc.) to detect external libraries and version constraints.

**Integration**: Adds context to the "Knowledge Graph", allowing queries like "Which chunks use the `requests` library?".

**Proposed Interface**:
```python
class DependencyResolver:
    def resolve(self, root_path: Path) -> DependencyGraph:
        """
        Scans root_path for manifest files and builds a graph of
        Project -> External Dependencies.
        """
        pass
        
    def get_dependency_map(self) -> Dict[str, List[Dependency]]:
        """Returns a flattened map of dependencies found."""
        pass
```

### C. QualityAnalyzer
**Responsibility**: specialized analysis for code quality metrics (Complexity, Maintainability Index, Line counts, Comment density).

**Integration**: `ChunkProcessor` calls this after chunking. The metrics are stored in `Chunk.metadata.metrics`.

**Proposed Interface**:
```python
class QualityAnalyzer:
    def analyze_chunk(self, chunk: Chunk) -> Dict[str, Union[int, float]]:
        """Calculates metrics for a single code chunk."""
        # e.g., Cyclomatic Complexity, Halstead metrics
        pass

    def analyze_file(self, content: str, language: str) -> FileMetrics:
        """Calculates file-level metrics."""
        pass
```

### D. ReportGenerator
**Responsibility**: Summarize the pipeline execution statistics into a human-readable format (Markdown/HTML) and machine-readable format (JSON).

**Integration**: Called by `MasterPipeline` at the `PipelineStatus.COMPLETED` stage.

**Proposed Interface**:
```python
class ReportGenerator:
    def generate(self, stats: PipelineStats, output_dir: Path) -> Path:
        """
        Generates:
        1. summary.json (Counts, Times, Errors)
        2. report.md (Visual overview of languages, top relationships, efficiency)
        """
        pass
```

## 4. Integration Logic (MasterPipeline Update)

We will need to update `MasterPipeline` to initialize these instances:

```python
# src/pipeline/master_pipeline.py

class MasterPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.scanner = RepositoryScanner(self.config.scanner)
        self.resolver = DependencyResolver(self.config.resolver)
        self.quality = QualityAnalyzer()
        self.reporter = ReportGenerator()

    async def run(self):
        # 1. Scan
        files = self.scanner.scan_files(self.workspace)
        
        # 2. Resolve Deps (Parallel with Scan usually, or after)
        deps = self.resolver.resolve(self.workspace)
        
        # 3. Process
        for file, classification in files:
            # Chunking -> Quality Analysis -> Embedding -> Storage
            await self.chunk_processor.process(file, classification, quality_analyzer=self.quality)
            
        # 4. Report
        self.reporter.generate(self.stats)
```

## 5. Implementation Order for Discussion
1.  **Scanner**: Critical path. Without this, we can't get files.
2.  **ReportGenerator**: High value for visibility.
3.  **QualityAnalyzer**: Good "nice to have" for metadata.
4.  **DependencyResolver**: Can be complex; maybe start with simple regex for `requirements.txt`/`package.json`.
