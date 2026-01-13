#!/usr/bin/env python
"""
Repository Analysis Tool
Provides comprehensive analysis of repository structure, complexity, and content
"""

import os
import sys
import json
import click
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import re
import hashlib
from dataclasses import dataclass, asdict
import yaml
import chardet
import ast
import statistics

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.classifiers.file_classifier import FileClassifier
from src.classifiers.content_analyzer import UnknownFileAnalyzer
from config.settings import settings
from config.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)

@dataclass
class FileMetrics:
    """Metrics for a single file"""
    path: str
    size_bytes: int
    lines: int
    blank_lines: int
    comment_lines: int
    code_lines: int
    complexity: Optional[int]
    token_estimate: int
    encoding: str
    hash: str
    extension: str
    file_type: str
    language: Optional[str]
    last_modified: str

@dataclass
class RepositoryMetrics:
    """Overall repository metrics"""
    total_files: int
    total_size_bytes: int
    total_lines: int
    total_code_lines: int
    total_comment_lines: int
    total_blank_lines: int
    file_types: Dict[str, int]
    languages: Dict[str, int]
    extensions: Dict[str, int]
    largest_files: List[Dict[str, Any]]
    most_complex_files: List[Dict[str, Any]]
    duplicate_files: List[List[str]]
    encoding_stats: Dict[str, int]
    git_stats: Optional[Dict[str, Any]]

class RepositoryAnalyzer:
    """Main repository analyzer"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.classifier = FileClassifier()
        self.content_analyzer = UnknownFileAnalyzer()
        self.perf_logger = PerformanceLogger()
        
        # Load file mappings
        self.file_mappings = self._load_file_mappings()
        
        # Analysis results
        self.file_metrics: List[FileMetrics] = []
        self.unknown_files: List[Path] = []
        self.binary_files: List[Path] = []
        self.large_files: List[Tuple[Path, int]] = []
        self.errors: List[Dict[str, str]] = []
        
        # Caches
        self.hash_cache: Dict[str, List[str]] = defaultdict(list)
        
    def _load_file_mappings(self) -> Dict:
        """Load file mappings configuration"""
        mapping_file = settings.CONFIG_DIR / "file_mappings.yaml"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def analyze(self) -> RepositoryMetrics:
        """Perform complete repository analysis"""
        logger.info(f"Starting analysis of repository: {self.repo_path}")
        
        # Collect all files
        all_files = self._collect_files()
        logger.info(f"Found {len(all_files)} files to analyze")
        
        # Analyze each file
        for file_path in all_files:
            try:
                metrics = self._analyze_file(file_path)
                if metrics:
                    self.file_metrics.append(metrics)
                    
                    # Track file hash for duplicate detection
                    self.hash_cache[metrics.hash].append(str(file_path))
                    
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                self.errors.append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        # Generate repository metrics
        repo_metrics = self._calculate_repository_metrics()
        
        # Add Git statistics if available
        if self._is_git_repo():
            repo_metrics.git_stats = self._get_git_stats()
        
        return repo_metrics
    
    def _collect_files(self) -> List[Path]:
        """Collect all files from repository"""
        files = []
        
        for file_path in self.repo_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip ignored directories
            if any(skip_dir in file_path.parts for skip_dir in settings.SKIP_DIRS):
                continue
            
            files.append(file_path)
        
        return files
    
    def _analyze_file(self, file_path: Path) -> Optional[FileMetrics]:
        """Analyze a single file"""
        try:
            # Get file stats
            stat = file_path.stat()
            size_bytes = stat.st_size
            
            # Skip very large files
            if size_bytes > 10 * 1024 * 1024:  # 10MB
                self.large_files.append((file_path, size_bytes))
                return None
            
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # Check if binary
            if self._is_binary(file_path, encoding):
                self.binary_files.append(file_path)
                return None
            
            # Read file content
            try:
                with open(file_path, 'r', encoding=encoding['encoding']) as f:
                    content = f.read()
            except:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            # Calculate metrics
            lines_data = self._analyze_lines(content, file_path.suffix)
            complexity = self._calculate_complexity(content, file_path.suffix)
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Classify file
            file_context = self.classifier.classify(file_path)
            
            # Check if unknown
            if file_context.confidence < 0.8:
                self.unknown_files.append(file_path)
            
            # Estimate tokens
            token_estimate = self._estimate_tokens(content)
            
            return FileMetrics(
                path=str(file_path.relative_to(self.repo_path)),
                size_bytes=size_bytes,
                lines=lines_data['total'],
                blank_lines=lines_data['blank'],
                comment_lines=lines_data['comments'],
                code_lines=lines_data['code'],
                complexity=complexity,
                token_estimate=token_estimate,
                encoding=encoding['encoding'],
                hash=file_hash,
                extension=file_path.suffix,
                file_type=file_context.file_type,
                language=file_context.language,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _detect_encoding(self, file_path: Path) -> Dict[str, Any]:
        """Detect file encoding"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
        return result or {'encoding': 'utf-8', 'confidence': 0}
    
    def _is_binary(self, file_path: Path, encoding_info: Dict) -> bool:
        """Check if file is binary"""
        # Check by extension
        if file_path.suffix.lower() in settings.SKIP_EXTENSIONS:
            return True
        
        # Check by encoding confidence
        if encoding_info.get('confidence', 0) < 0.7:
            return True
        
        # Check content
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:
                    return True
        except:
            pass
        
        return False
    
    def _analyze_lines(self, content: str, extension: str) -> Dict[str, int]:
        """Analyze line types in file"""
        lines = content.split('\n')
        total_lines = len(lines)
        blank_lines = 0
        comment_lines = 0
        
        # Comment patterns by extension
        comment_patterns = {
            '.py': (r'^\s*#', r'^\s*"""', r'^\s*\'\'\''),
            '.java': (r'^\s*//', r'^\s*/\*', r'^\s*\*'),
            '.js': (r'^\s*//', r'^\s*/\*', r'^\s*\*'),
            '.sql': (r'^\s*--', r'^\s*/\*'),
            '.xml': (r'^\s*<!--', r'^\s*-->'),
            '.yaml': (r'^\s*#',),
            '.yml': (r'^\s*#',),
            '.sh': (r'^\s*#',),
        }
        
        patterns = comment_patterns.get(extension.lower(), ())
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif patterns and any(re.match(p, line) for p in patterns):
                comment_lines += 1
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return {
            'total': total_lines,
            'blank': blank_lines,
            'comments': comment_lines,
            'code': code_lines
        }
    
    def _calculate_complexity(self, content: str, extension: str) -> Optional[int]:
        """Calculate cyclomatic complexity (simplified)"""
        if extension not in ['.py', '.java', '.js', '.ts']:
            return None
        
        # Simple complexity estimation based on control flow keywords
        complexity_keywords = [
            'if', 'else', 'elif', 'for', 'while', 'case', 'catch',
            'switch', 'try', 'except', 'finally'
        ]
        
        complexity = 1  # Base complexity
        for keyword in complexity_keywords:
            # Use word boundaries to avoid false matches
            pattern = r'\b' + keyword + r'\b'
            complexity += len(re.findall(pattern, content))
        
        return complexity
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for CodeBERT"""
        # Rough estimation: ~1.5 tokens per word
        words = len(content.split())
        return int(words * 1.5)
    
    def _calculate_repository_metrics(self) -> RepositoryMetrics:
        """Calculate overall repository metrics"""
        
        # Aggregate metrics
        total_size = sum(m.size_bytes for m in self.file_metrics)
        total_lines = sum(m.lines for m in self.file_metrics)
        total_code = sum(m.code_lines for m in self.file_metrics)
        total_comments = sum(m.comment_lines for m in self.file_metrics)
        total_blank = sum(m.blank_lines for m in self.file_metrics)
        
        # Count by categories
        file_types = Counter(m.file_type for m in self.file_metrics)
        languages = Counter(m.language for m in self.file_metrics if m.language)
        extensions = Counter(m.extension for m in self.file_metrics)
        encodings = Counter(m.encoding for m in self.file_metrics)
        
        # Find largest files
        largest_files = sorted(
            [{'path': m.path, 'size': m.size_bytes} for m in self.file_metrics],
            key=lambda x: x['size'],
            reverse=True
        )[:10]
        
        # Find most complex files
        complex_files = sorted(
            [{'path': m.path, 'complexity': m.complexity} 
             for m in self.file_metrics if m.complexity],
            key=lambda x: x['complexity'],
            reverse=True
        )[:10]
        
        # Find duplicate files
        duplicates = [
            paths for file_hash, paths in self.hash_cache.items()
            if len(paths) > 1
        ]
        
        return RepositoryMetrics(
            total_files=len(self.file_metrics),
            total_size_bytes=total_size,
            total_lines=total_lines,
            total_code_lines=total_code,
            total_comment_lines=total_comments,
            total_blank_lines=total_blank,
            file_types=dict(file_types),
            languages=dict(languages),
            extensions=dict(extensions),
            largest_files=largest_files,
            most_complex_files=complex_files,
            duplicate_files=duplicates,
            encoding_stats=dict(encodings),
            git_stats=None
        )
    
    def _is_git_repo(self) -> bool:
        """Check if directory is a git repository"""
        git_dir = self.repo_path / '.git'
        return git_dir.exists() and git_dir.is_dir()
    
    def _get_git_stats(self) -> Dict[str, Any]:
        """Get git repository statistics"""
        stats = {}
        
        try:
            # Get commit count
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                stats['total_commits'] = int(result.stdout.strip())
            
            # Get contributors
            result = subprocess.run(
                ['git', 'shortlog', '-sn'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                contributors = result.stdout.strip().split('\n')
                stats['contributors'] = len(contributors)
                stats['top_contributors'] = []
                
                for line in contributors[:5]:
                    match = re.match(r'\s*(\d+)\s+(.+)', line)
                    if match:
                        stats['top_contributors'].append({
                            'name': match.group(2),
                            'commits': int(match.group(1))
                        })
            
            # Get branches
            result = subprocess.run(
                ['git', 'branch', '-r'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                branches = [b.strip() for b in result.stdout.strip().split('\n')]
                stats['branch_count'] = len(branches)
            
            # Get recent activity
            result = subprocess.run(
                ['git', 'log', '--since=30.days', '--oneline'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                recent_commits = result.stdout.strip().split('\n')
                stats['commits_last_30_days'] = len(recent_commits)
            
            # Get file churn (files changed frequently)
            result = subprocess.run(
                ['git', 'log', '--format=', '--name-only', '--since=90.days'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                changed_files = [f for f in result.stdout.strip().split('\n') if f]
                file_changes = Counter(changed_files)
                stats['most_changed_files'] = [
                    {'file': f, 'changes': c}
                    for f, c in file_changes.most_common(10)
                ]
            
        except Exception as e:
            logger.error(f"Error getting git stats: {e}")
        
        return stats

class ChunkingEstimator:
    """Estimate chunking requirements and processing time"""
    
    def __init__(self, metrics: RepositoryMetrics, file_metrics: List[FileMetrics]):
        self.metrics = metrics
        self.file_metrics = file_metrics
    
    def estimate(self) -> Dict[str, Any]:
        """Estimate chunking requirements"""
        
        # Estimate chunks per file type
        chunk_estimates = {}
        
        for file_metric in self.file_metrics:
            ext = file_metric.extension
            
            # Estimate chunks based on token count
            estimated_chunks = max(1, file_metric.token_estimate // 400)
            
            if ext not in chunk_estimates:
                chunk_estimates[ext] = {
                    'file_count': 0,
                    'total_chunks': 0,
                    'avg_chunks_per_file': 0
                }
            
            chunk_estimates[ext]['file_count'] += 1
            chunk_estimates[ext]['total_chunks'] += estimated_chunks
        
        # Calculate averages
        for ext, data in chunk_estimates.items():
            if data['file_count'] > 0:
                data['avg_chunks_per_file'] = data['total_chunks'] / data['file_count']
        
        # Total estimates
        total_chunks = sum(data['total_chunks'] for data in chunk_estimates.values())
        
        # Processing time estimation (rough)
        # Assume: 100 files/minute, 10 chunks/second
        estimated_time_seconds = (
            len(self.file_metrics) * 0.6 +  # File reading
            total_chunks * 0.1  # Chunk processing
        )
        
        # Memory estimation
        # Assume: 1KB per chunk in memory
        estimated_memory_mb = (total_chunks * 1) / 1024
        
        # Storage estimation
        # Chunks + embeddings
        estimated_storage_mb = (
            total_chunks * 2 +  # Raw chunks
            total_chunks * 4    # Embeddings (768 dims * float32)
        ) / 1024
        
        return {
            'chunk_estimates': chunk_estimates,
            'total_estimated_chunks': total_chunks,
            'estimated_processing_time_seconds': estimated_time_seconds,
            'estimated_processing_time_formatted': self._format_time(estimated_time_seconds),
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_storage_mb': estimated_storage_mb,
            'recommendations': self._get_recommendations(total_chunks, estimated_time_seconds)
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"
    
    def _get_recommendations(self, total_chunks: int, time_seconds: float) -> List[str]:
        """Get processing recommendations"""
        recommendations = []
        
        if total_chunks > 10000:
            recommendations.append("Consider processing in batches due to large chunk count")
        
        if time_seconds > 3600:
            recommendations.append("Consider running as a background job due to long processing time")
        
        if self.metrics.total_size_bytes > 1024 * 1024 * 1024:  # 1GB
            recommendations.append("Repository is large, consider filtering unnecessary files")
        
        if len(self.metrics.duplicate_files) > 10:
            recommendations.append(f"Found {len(self.metrics.duplicate_files)} duplicate files - consider deduplication")
        
        return recommendations

class ReportGenerator:
    """Generate analysis reports in various formats"""
    
    def __init__(self, metrics: RepositoryMetrics, file_metrics: List[FileMetrics], 
                 estimates: Dict[str, Any], repo_path: Path):
        self.metrics = metrics
        self.file_metrics = file_metrics
        self.estimates = estimates
        self.repo_path = repo_path
        self.timestamp = datetime.now()
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON report"""
        return {
            'repository': str(self.repo_path),
            'analysis_timestamp': self.timestamp.isoformat(),
            'summary': {
                'total_files': self.metrics.total_files,
                'total_size_mb': self.metrics.total_size_bytes / (1024 * 1024),
                'total_lines': self.metrics.total_lines,
                'code_lines': self.metrics.total_code_lines,
                'comment_ratio': self.metrics.total_comment_lines / max(self.metrics.total_lines, 1),
                'languages': list(self.metrics.languages.keys())
            },
            'metrics': asdict(self.metrics),
            'estimates': self.estimates,
            'file_details': [asdict(fm) for fm in self.file_metrics[:100]]  # Limit details
        }
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"REPOSITORY ANALYSIS REPORT")
        lines.append(f"Repository: {self.repo_path}")
        lines.append(f"Generated: {self.timestamp:%Y-%m-%d %H:%M:%S}")
        lines.append("=" * 80)
        
        # Summary
        lines.append("\n## SUMMARY")
        lines.append(f"Total Files: {self.metrics.total_files:,}")
        lines.append(f"Total Size: {self.metrics.total_size_bytes / (1024*1024):.2f} MB")
        lines.append(f"Total Lines: {self.metrics.total_lines:,}")
        lines.append(f"  - Code: {self.metrics.total_code_lines:,} ({self.metrics.total_code_lines/max(self.metrics.total_lines,1)*100:.1f}%)")
        lines.append(f"  - Comments: {self.metrics.total_comment_lines:,} ({self.metrics.total_comment_lines/max(self.metrics.total_lines,1)*100:.1f}%)")
        lines.append(f"  - Blank: {self.metrics.total_blank_lines:,} ({self.metrics.total_blank_lines/max(self.metrics.total_lines,1)*100:.1f}%)")
        
        # Languages
        lines.append("\n## LANGUAGES")
        for lang, count in sorted(self.metrics.languages.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"  {lang:20} {count:5} files")
        
        # File Types
        lines.append("\n## FILE TYPES")
        for ftype, count in sorted(self.metrics.file_types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {ftype:20} {count:5} files")
        
        # Extensions
        lines.append("\n## TOP EXTENSIONS")
        for ext, count in sorted(self.metrics.extensions.items(), key=lambda x: x[1], reverse=True)[:15]:
            lines.append(f"  {ext:10} {count:5} files")
        
        # Largest Files
        lines.append("\n## LARGEST FILES")
        for f in self.metrics.largest_files[:10]:
            size_mb = f['size'] / (1024 * 1024)
            lines.append(f"  {size_mb:8.2f} MB  {f['path']}")
        
        # Most Complex Files
        if self.metrics.most_complex_files:
            lines.append("\n## MOST COMPLEX FILES")
            for f in self.metrics.most_complex_files[:10]:
                lines.append(f"  Complexity: {f['complexity']:4}  {f['path']}")
        
        # Duplicates
        if self.metrics.duplicate_files:
            lines.append(f"\n## DUPLICATE FILES")
            lines.append(f"Found {len(self.metrics.duplicate_files)} sets of duplicates")
            for i, group in enumerate(self.metrics.duplicate_files[:5], 1):
                lines.append(f"\nDuplicate Set {i}:")
                for path in group:
                    lines.append(f"  - {path}")
        
        # Git Stats
        if self.metrics.git_stats:
            lines.append("\n## GIT STATISTICS")
            lines.append(f"Total Commits: {self.metrics.git_stats.get('total_commits', 'N/A')}")
            lines.append(f"Contributors: {self.metrics.git_stats.get('contributors', 'N/A')}")
            lines.append(f"Branches: {self.metrics.git_stats.get('branch_count', 'N/A')}")
            lines.append(f"Recent Activity: {self.metrics.git_stats.get('commits_last_30_days', 0)} commits in last 30 days")
            
            if 'top_contributors' in self.metrics.git_stats:
                lines.append("\nTop Contributors:")
                for contrib in self.metrics.git_stats['top_contributors']:
                    lines.append(f"  {contrib['name']:30} {contrib['commits']:5} commits")
            
            if 'most_changed_files' in self.metrics.git_stats:
                lines.append("\nMost Changed Files (last 90 days):")
                for f in self.metrics.git_stats['most_changed_files'][:5]:
                    lines.append(f"  {f['changes']:3} changes  {f['file']}")
        
        # Chunking Estimates
        lines.append("\n## CHUNKING ESTIMATES")
        lines.append(f"Estimated Total Chunks: {self.estimates['total_estimated_chunks']:,}")
        lines.append(f"Estimated Processing Time: {self.estimates['estimated_processing_time_formatted']}")
        lines.append(f"Estimated Memory Required: {self.estimates['estimated_memory_mb']:.1f} MB")
        lines.append(f"Estimated Storage Required: {self.estimates['estimated_storage_mb']:.1f} MB")
        
        if self.estimates['recommendations']:
            lines.append("\nRecommendations:")
            for rec in self.estimates['recommendations']:
                lines.append(f"  • {rec}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def generate_csv_report(self) -> str:
        """Generate CSV report of file metrics"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=['path', 'extension', 'language', 'file_type', 
                       'size_bytes', 'lines', 'code_lines', 'comment_lines',
                       'complexity', 'token_estimate', 'encoding']
        )
        
        writer.writeheader()
        for fm in self.file_metrics:
            writer.writerow({
                'path': fm.path,
                'extension': fm.extension,
                'language': fm.language,
                'file_type': fm.file_type,
                'size_bytes': fm.size_bytes,
                'lines': fm.lines,
                'code_lines': fm.code_lines,
                'comment_lines': fm.comment_lines,
                'complexity': fm.complexity,
                'token_estimate': fm.token_estimate,
                'encoding': fm.encoding
            })
        
        return output.getvalue()
    
    def generate_html_report(self) -> str:
        """Generate HTML report with visualizations"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Repository Analysis Report - {self.repo_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #999; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }}
        th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .warning {{ background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .chart {{ width: 100%; height: 300px; background: white; margin: 20px 0; padding: 20px; border-radius: 8px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Repository Analysis Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-value">{self.metrics.total_files:,}</div>
            <div class="metric-label">Total Files</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.metrics.total_size_bytes / (1024*1024):.1f} MB</div>
            <div class="metric-label">Total Size</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.metrics.total_lines:,}</div>
            <div class="metric-label">Total Lines</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(self.metrics.languages)}</div>
            <div class="metric-label">Languages</div>
        </div>
    </div>
    
    <h2>Language Distribution</h2>
    <div class="chart">
        <canvas id="languageChart"></canvas>
    </div>
    
    <h2>File Type Distribution</h2>
    <table>
        <tr>
            <th>File Type</th>
            <th>Count</th>
            <th>Percentage</th>
        </tr>
        {"".join(f'''
        <tr>
            <td>{ftype}</td>
            <td>{count}</td>
            <td>{count/self.metrics.total_files*100:.1f}%</td>
        </tr>
        ''' for ftype, count in sorted(self.metrics.file_types.items(), key=lambda x: x[1], reverse=True))}
    </table>
    
    <h2>Chunking Estimates</h2>
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{self.estimates['total_estimated_chunks']:,}</div>
            <div class="metric-label">Estimated Chunks</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.estimates['estimated_processing_time_formatted']}</div>
            <div class="metric-label">Processing Time</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.estimates['estimated_memory_mb']:.1f} MB</div>
            <div class="metric-label">Memory Required</div>
        </div>
    </div>
    
    {self._generate_recommendations_html()}
    
    <script>
        // Language chart
        const ctx = document.getElementById('languageChart').getContext('2d');
        new Chart(ctx, {{
            type: 'pie',
            data: {{
                labels: {list(self.metrics.languages.keys())},
                datasets: [{{
                    data: {list(self.metrics.languages.values())},
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                    ]
                }}]
            }}
        }});
    </script>
    
    <footer>
        <p style="text-align: center; color: #999; margin-top: 50px;">
            Generated on {self.timestamp:%Y-%m-%d %H:%M:%S}
        </p>
    </footer>
</body>
</html>
        """
        return html
    
    def _generate_recommendations_html(self) -> str:
        """Generate HTML for recommendations"""
        if not self.estimates['recommendations']:
            return ""
        
        recs = "\n".join(f"<li>{rec}</li>" for rec in self.estimates['recommendations'])
        return f"""
        <div class="warning">
            <h3>Recommendations</h3>
            <ul>
                {recs}
            </ul>
        </div>
        """

@click.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory', default='output/reports')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'html', 'csv', 'all']), 
              default='text', help='Report format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--include-files', is_flag=True, help='Include file details in report')
def main(repo_path: str, output: str, format: str, verbose: bool, include_files: bool):
    """Analyze repository structure and estimate chunking requirements"""
    
    try:
        repo = Path(repo_path)
        click.echo(f"Analyzing repository: {repo}")
        
        # Run analysis
        analyzer = RepositoryAnalyzer(repo)
        metrics = analyzer.analyze()
        
        # Estimate chunking
        estimator = ChunkingEstimator(metrics, analyzer.file_metrics)
        estimates = estimator.estimate()
        
        # Generate reports
        generator = ReportGenerator(metrics, analyzer.file_metrics, estimates, repo)
        
        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reports based on format
        if format in ['text', 'all']:
            text_report = generator.generate_text_report()
            report_file = output_dir / f"analysis_{repo.name}_{datetime.now():%Y%m%d_%H%M%S}.txt"
            with open(report_file, 'w') as f:
                f.write(text_report)
            click.echo(f"Text report saved to: {report_file}")
            
            # Also print to console
            if verbose or format == 'text':
                click.echo("\n" + text_report)
        
        if format in ['json', 'all']:
            json_report = generator.generate_json_report()
            report_file = output_dir / f"analysis_{repo.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(report_file, 'w') as f:
                json.dump(json_report, f, indent=2)
            click.echo(f"JSON report saved to: {report_file}")
        
        if format in ['html', 'all']:
            html_report = generator.generate_html_report()
            report_file = output_dir / f"analysis_{repo.name}_{datetime.now():%Y%m%d_%H%M%S}.html"
            with open(report_file, 'w') as f:
                f.write(html_report)
            click.echo(f"HTML report saved to: {report_file}")
        
        if format in ['csv', 'all'] and include_files:
            csv_report = generator.generate_csv_report()
            report_file = output_dir / f"files_{repo.name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
            with open(report_file, 'w') as f:
                f.write(csv_report)
            click.echo(f"CSV file details saved to: {report_file}")
        
        # Report unknown files if any
        if analyzer.unknown_files:
            click.echo(f"\n Found {len(analyzer.unknown_files)} unknown file types")
            if verbose:
                click.echo("Unknown files:")
                for f in analyzer.unknown_files[:10]:
                    click.echo(f"  - {f}")
        
        # Report errors if any
        if analyzer.errors:
            click.echo(f"\n {len(analyzer.errors)} errors occurred during analysis")
            if verbose:
                for error in analyzer.errors[:5]:
                    click.echo(f"  - {error['file']}: {error['error']}")
        
        click.echo("\n✅ Analysis complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
    