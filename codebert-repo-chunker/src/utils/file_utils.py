"""
File utilities for handling various file operations, formats, and encodings
Provides robust file handling with error recovery and performance optimization
"""

import os
import shutil
import tempfile
import hashlib
import mimetypes
import magic
import chardet
import codecs
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Iterator, Tuple, BinaryIO, TextIO
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import yaml
import toml
import configparser
import csv
import zipfile
import tarfile
import gzip
import bz2
import lzma
from contextlib import contextmanager
import logging
import fnmatch
import re
import mmap
import fcntl
import stat
import threading
from functools import lru_cache
import pickle

logger = logging.getLogger(__name__)

class FileType(Enum):
    """File type classifications"""
    SOURCE_CODE = "source_code"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    DATA = "data"
    BINARY = "binary"
    ARCHIVE = "archive"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    UNKNOWN = "unknown"

class EncodingType(Enum):
    """Common file encodings"""
    UTF8 = "utf-8"
    UTF16 = "utf-16"
    UTF32 = "utf-32"
    ASCII = "ascii"
    LATIN1 = "latin-1"
    CP1252 = "cp1252"
    GB2312 = "gb2312"
    SHIFT_JIS = "shift_jis"
    EUC_KR = "euc-kr"

@dataclass
class FileInfo:
    """Comprehensive file information"""
    path: Path
    name: str
    extension: str
    size_bytes: int
    mime_type: Optional[str]
    encoding: Optional[str]
    file_type: FileType
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    is_binary: bool
    is_hidden: bool
    is_symlink: bool
    permissions: str
    owner: Optional[str]
    group: Optional[str]
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class FileUtils:
    """Comprehensive file utility operations"""
    
    # Common source code extensions
    SOURCE_EXTENSIONS = {
        '.py', '.pyw', '.pyx', '.pyi',  # Python
        '.java', '.class', '.jar',  # Java
        '.js', '.mjs', '.jsx', '.ts', '.tsx',  # JavaScript/TypeScript
        '.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.hxx',  # C/C++
        '.cs', '.vb', '.fs',  # .NET
        '.go',  # Go
        '.rs',  # Rust
        '.rb', '.erb',  # Ruby
        '.php', '.phtml',  # PHP
        '.swift',  # Swift
        '.kt', '.kts',  # Kotlin
        '.scala',  # Scala
        '.r', '.R',  # R
        '.m', '.mm',  # Objective-C
        '.lua',  # Lua
        '.pl', '.pm',  # Perl
        '.sh', '.bash', '.zsh', '.fish',  # Shell
        '.ps1', '.psm1',  # PowerShell
        '.sql',  # SQL
        '.dart',  # Dart
        '.jl',  # Julia
        '.ex', '.exs',  # Elixir
        '.clj', '.cljs',  # Clojure
        '.elm',  # Elm
        '.ml', '.mli',  # OCaml
        '.hs',  # Haskell
        '.erl', '.hrl',  # Erlang
        '.nim',  # Nim
        '.v', '.vh',  # Verilog
        '.vhd', '.vhdl',  # VHDL
        '.asm', '.s',  # Assembly
    }
    
    # Configuration file extensions
    CONFIG_EXTENSIONS = {
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
        '.properties', '.env', '.config', '.settings', '.xml', '.plist'
    }
    
    # Documentation extensions
    DOC_EXTENSIONS = {
        '.md', '.rst', '.txt', '.adoc', '.tex', '.rtf',
        '.doc', '.docx', '.odt', '.pdf', '.html', '.htm'
    }
    
    # Data file extensions
    DATA_EXTENSIONS = {
        '.csv', '.tsv', '.parquet', '.feather', '.hdf', '.h5',
        '.npz', '.npy', '.pkl', '.pickle', '.msgpack', '.arrow'
    }
    
    # Archive extensions
    ARCHIVE_EXTENSIONS = {
        '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
        '.tar.gz', '.tar.bz2', '.tar.xz', '.tgz'
    }
    
    # Binary file magic numbers
    BINARY_SIGNATURES = {
        b'\x7fELF': 'elf',  # ELF executable
        b'MZ': 'exe',  # Windows executable
        b'\xca\xfe\xba\xbe': 'class',  # Java class
        b'PK\x03\x04': 'zip',  # ZIP archive
        b'\x1f\x8b': 'gzip',  # GZIP
        b'BZh': 'bzip2',  # BZIP2
        b'\x50\x4b\x03\x04': 'jar',  # JAR file
        b'\x89PNG': 'png',  # PNG image
        b'\xff\xd8\xff': 'jpeg',  # JPEG image
        b'GIF87a': 'gif',  # GIF image
        b'GIF89a': 'gif',  # GIF image
        b'%PDF': 'pdf',  # PDF document
    }
    
    def __init__(self, 
                 default_encoding: str = 'utf-8',
                 use_magic: bool = True,
                 cache_size: int = 1000):
        """
        Initialize FileUtils
        
        Args:
            default_encoding: Default encoding for text files
            use_magic: Use python-magic for MIME type detection
            cache_size: Size of LRU cache for file operations
        """
        self.default_encoding = default_encoding
        self.use_magic = use_magic and self._check_magic_available()
        self._lock = threading.RLock()
        
        # Initialize caches
        self._hash_cache = {}
        self._encoding_cache = {}
        
        # Configure LRU cache
        self.get_file_info = lru_cache(maxsize=cache_size)(self._get_file_info_impl)
    
    def _check_magic_available(self) -> bool:
        """Check if python-magic is available"""
        try:
            import magic
            return True
        except ImportError:
            logger.warning("python-magic not available, using basic MIME detection")
            return False
    
    def read_file(self, 
                  file_path: Union[str, Path],
                  encoding: Optional[str] = None,
                  errors: str = 'replace') -> str:
        """
        Read text file with automatic encoding detection
        
        Args:
            file_path: Path to file
            encoding: Force specific encoding
            errors: Error handling strategy
            
        Returns:
            File contents as string
        """
        file_path = Path(file_path)
        
        if not encoding:
            encoding = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding, errors=errors) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            # Try with default encoding
            with open(file_path, 'r', encoding=self.default_encoding, errors='replace') as f:
                return f.read()
    
    def write_file(self,
                   file_path: Union[str, Path],
                   content: str,
                   encoding: Optional[str] = None,
                   create_dirs: bool = True,
                   atomic: bool = False) -> bool:
        """
        Write text file with optional atomic write
        
        Args:
            file_path: Path to file
            content: Content to write
            encoding: Text encoding
            create_dirs: Create parent directories
            atomic: Use atomic write (write to temp then rename)
            
        Returns:
            Success status
        """
        file_path = Path(file_path)
        encoding = encoding or self.default_encoding
        
        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if atomic:
                # Atomic write using temporary file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    encoding=encoding,
                    dir=file_path.parent,
                    delete=False
                ) as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                # Atomic rename
                Path(tmp_path).replace(file_path)
            else:
                # Direct write
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False
    
    def read_binary(self, file_path: Union[str, Path]) -> bytes:
        """Read binary file"""
        with open(file_path, 'rb') as f:
            return f.read()
    
    def write_binary(self, 
                    file_path: Union[str, Path],
                    content: bytes,
                    atomic: bool = False) -> bool:
        """Write binary file"""
        file_path = Path(file_path)
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if atomic:
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    dir=file_path.parent,
                    delete=False
                ) as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                Path(tmp_path).replace(file_path)
            else:
                with open(file_path, 'wb') as f:
                    f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write binary file {file_path}: {e}")
            return False
    
    def detect_encoding(self, 
                       file_path: Union[str, Path],
                       sample_size: int = 65536) -> str:
        """
        Detect file encoding
        
        Args:
            file_path: Path to file
            sample_size: Bytes to sample for detection
            
        Returns:
            Detected encoding
        """
        file_path = Path(file_path)
        
        # Check cache
        if str(file_path) in self._encoding_cache:
            return self._encoding_cache[str(file_path)]
        
        try:
            # Read sample
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
            
            # Check for BOM
            if raw_data.startswith(codecs.BOM_UTF8):
                encoding = 'utf-8-sig'
            elif raw_data.startswith(codecs.BOM_UTF16_LE):
                encoding = 'utf-16-le'
            elif raw_data.startswith(codecs.BOM_UTF16_BE):
                encoding = 'utf-16-be'
            elif raw_data.startswith(codecs.BOM_UTF32_LE):
                encoding = 'utf-32-le'
            elif raw_data.startswith(codecs.BOM_UTF32_BE):
                encoding = 'utf-32-be'
            else:
                # Use chardet for detection
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', self.default_encoding)
                
                # Validate encoding
                if not encoding or result.get('confidence', 0) < 0.7:
                    encoding = self.default_encoding
            
            # Cache result
            self._encoding_cache[str(file_path)] = encoding
            
            return encoding
            
        except Exception as e:
            logger.warning(f"Failed to detect encoding for {file_path}: {e}")
            return self.default_encoding
    
    def get_mime_type(self, file_path: Union[str, Path]) -> Optional[str]:
        """Get MIME type of file"""
        file_path = Path(file_path)
        
        if self.use_magic:
            try:
                mime = magic.Magic(mime=True)
                return mime.from_file(str(file_path))
            except Exception as e:
                logger.warning(f"Magic MIME detection failed: {e}")
        
        # Fallback to mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type
    
    def is_binary(self, file_path: Union[str, Path]) -> bool:
        """Check if file is binary"""
        file_path = Path(file_path)
        
        try:
            # Check by extension first
            if file_path.suffix.lower() in {'.exe', '.dll', '.so', '.dylib', '.class', '.pyc'}:
                return True
            
            # Check magic numbers
            with open(file_path, 'rb') as f:
                header = f.read(512)
            
            # Check for known binary signatures
            for signature in self.BINARY_SIGNATURES:
                if header.startswith(signature):
                    return True
            
            # Check for null bytes
            if b'\x00' in header:
                return True
            
            # Try to decode as text
            try:
                header.decode('utf-8')
                return False
            except UnicodeDecodeError:
                return True
                
        except Exception as e:
            logger.warning(f"Failed to check if binary: {e}")
            return False
    
    def get_file_type(self, file_path: Union[str, Path]) -> FileType:
        """Determine file type based on extension and content"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension in self.SOURCE_EXTENSIONS:
            return FileType.SOURCE_CODE
        elif extension in self.CONFIG_EXTENSIONS:
            return FileType.CONFIGURATION
        elif extension in self.DOC_EXTENSIONS:
            return FileType.DOCUMENTATION
        elif extension in self.DATA_EXTENSIONS:
            return FileType.DATA
        elif extension in self.ARCHIVE_EXTENSIONS:
            return FileType.ARCHIVE
        elif extension in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico'}:
            return FileType.IMAGE
        elif extension in {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}:
            return FileType.VIDEO
        elif extension in {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}:
            return FileType.AUDIO
        elif self.is_binary(file_path):
            return FileType.BINARY
        elif extension in {'.txt', '.log', '.out'} or not extension:
            return FileType.TEXT
        else:
            return FileType.UNKNOWN
    
    def calculate_hash(self, 
                      file_path: Union[str, Path],
                      algorithm: str = 'sha256',
                      chunk_size: int = 8192) -> str:
        """
        Calculate file hash
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
            chunk_size: Chunk size for reading
            
        Returns:
            Hex digest of file hash
        """
        file_path = Path(file_path)
        cache_key = f"{file_path}:{algorithm}"
        
        # Check cache
        if cache_key in self._hash_cache:
            # Verify file hasn't changed
            stat = file_path.stat()
            cached_mtime, cached_hash = self._hash_cache[cache_key]
            if stat.st_mtime == cached_mtime:
                return cached_hash
        
        # Calculate hash
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hash_obj.update(chunk)
            
            file_hash = hash_obj.hexdigest()
            
            # Cache result
            stat = file_path.stat()
            self._hash_cache[cache_key] = (stat.st_mtime, file_hash)
            
            return file_hash
            
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _get_file_info_impl(self, file_path: Union[str, Path]) -> FileInfo:
        """Get comprehensive file information (implementation)"""
        file_path = Path(file_path)
        
        try:
            stat = file_path.stat()
            
            # Get file metadata
            info = FileInfo(
                path=file_path,
                name=file_path.name,
                extension=file_path.suffix,
                size_bytes=stat.st_size,
                mime_type=self.get_mime_type(file_path),
                encoding=self.detect_encoding(file_path) if not self.is_binary(file_path) else None,
                file_type=self.get_file_type(file_path),
                created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                accessed_at=datetime.fromtimestamp(stat.st_atime, tz=timezone.utc),
                is_binary=self.is_binary(file_path),
                is_hidden=file_path.name.startswith('.'),
                is_symlink=file_path.is_symlink(),
                permissions=oct(stat.st_mode)[-3:],
                owner=self._get_owner(stat),
                group=self._get_group(stat)
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise
    
    def _get_owner(self, stat_result) -> Optional[str]:
        """Get file owner"""
        try:
            import pwd
            return pwd.getpwuid(stat_result.st_uid).pw_name
        except:
            return None
    
    def _get_group(self, stat_result) -> Optional[str]:
        """Get file group"""
        try:
            import grp
            return grp.getgrgid(stat_result.st_gid).gr_name
        except:
            return None
    
    def copy_file(self,
                  src: Union[str, Path],
                  dst: Union[str, Path],
                  preserve_metadata: bool = True) -> bool:
        """Copy file with optional metadata preservation"""
        try:
            src = Path(src)
            dst = Path(dst)
            
            if preserve_metadata:
                shutil.copy2(src, dst)
            else:
                shutil.copy(src, dst)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy {src} to {dst}: {e}")
            return False
    
    def move_file(self,
                  src: Union[str, Path],
                  dst: Union[str, Path]) -> bool:
        """Move/rename file"""
        try:
            src = Path(src)
            dst = Path(dst)
            
            shutil.move(str(src), str(dst))
            return True
            
        except Exception as e:
            logger.error(f"Failed to move {src} to {dst}: {e}")
            return False
    
    def delete_file(self, 
                   file_path: Union[str, Path],
                   secure: bool = False) -> bool:
        """
        Delete file with optional secure deletion
        
        Args:
            file_path: Path to file
            secure: Overwrite with random data before deletion
        """
        try:
            file_path = Path(file_path)
            
            if secure and file_path.is_file():
                # Secure deletion - overwrite with random data
                size = file_path.stat().st_size
                with open(file_path, 'ba+', buffering=0) as f:
                    for _ in range(3):  # Triple overwrite
                        f.seek(0)
                        f.write(os.urandom(size))
                        f.flush()
                        os.fsync(f.fileno())
            
            file_path.unlink()
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False
    
    @contextmanager
    def temp_file(self, 
                  suffix: str = '',
                  prefix: str = 'tmp',
                  dir: Optional[Path] = None,
                  text: bool = True):
        """Context manager for temporary file"""
        temp = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            mode='w' if text else 'wb',
            delete=False
        )
        
        try:
            yield temp
        finally:
            temp.close()
            Path(temp.name).unlink(missing_ok=True)
    
    @contextmanager
    def temp_directory(self,
                      suffix: str = '',
                      prefix: str = 'tmp',
                      dir: Optional[Path] = None):
        """Context manager for temporary directory"""
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        
        try:
            yield Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @contextmanager
    def file_lock(self, 
                  file_path: Union[str, Path],
                  exclusive: bool = True,
                  timeout: float = 10.0):
        """Context manager for file locking"""
        file_path = Path(file_path)
        lock_file = file_path.with_suffix('.lock')
        
        fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY)
        
        try:
            # Acquire lock
            flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(fd, flag)
            
            yield
            
        finally:
            # Release lock
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            lock_file.unlink(missing_ok=True)
    
    def find_files(self,
                   directory: Union[str, Path],
                   pattern: str = '*',
                   recursive: bool = True,
                   file_type: Optional[FileType] = None,
                   min_size: Optional[int] = None,
                   max_size: Optional[int] = None) -> Iterator[Path]:
        """
        Find files matching criteria
        
        Args:
            directory: Directory to search
            pattern: Glob pattern
            recursive: Search recursively
            file_type: Filter by file type
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            
        Yields:
            Matching file paths
        """
        directory = Path(directory)
        
        # Use rglob for recursive, glob for non-recursive
        files = directory.rglob(pattern) if recursive else directory.glob(pattern)
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            # Apply filters
            if file_type and self.get_file_type(file_path) != file_type:
                continue
            
            if min_size or max_size:
                size = file_path.stat().st_size
                if min_size and size < min_size:
                    continue
                if max_size and size > max_size:
                    continue
            
            yield file_path
    
    def read_lines(self,
                   file_path: Union[str, Path],
                   encoding: Optional[str] = None,
                   skip_empty: bool = False,
                   strip: bool = True) -> Iterator[str]:
        """
        Read file line by line
        
        Args:
            file_path: Path to file
            encoding: Text encoding
            skip_empty: Skip empty lines
            strip: Strip whitespace from lines
            
        Yields:
            Lines from file
        """
        file_path = Path(file_path)
        encoding = encoding or self.detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for line in f:
                if strip:
                    line = line.strip()
                
                if skip_empty and not line:
                    continue
                
                yield line
    
    def tail(self,
             file_path: Union[str, Path],
             lines: int = 10,
             encoding: Optional[str] = None) -> List[str]:
        """
        Read last n lines from file (like Unix tail)
        
        Args:
            file_path: Path to file
            lines: Number of lines to read
            encoding: Text encoding
            
        Returns:
            Last n lines
        """
        file_path = Path(file_path)
        encoding = encoding or self.detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            # Efficient tail using deque
            from collections import deque
            return list(deque(f, lines))
    
    def head(self,
             file_path: Union[str, Path],
             lines: int = 10,
             encoding: Optional[str] = None) -> List[str]:
        """
        Read first n lines from file (like Unix head)
        
        Args:
            file_path: Path to file
            lines: Number of lines to read
            encoding: Text encoding
            
        Returns:
            First n lines
        """
        file_path = Path(file_path)
        encoding = encoding or self.detect_encoding(file_path)
        
        result = []
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for i, line in enumerate(f):
                if i >= lines:
                    break
                result.append(line.rstrip('\n'))
        
        return result
    
    def grep(self,
             file_path: Union[str, Path],
             pattern: str,
             case_sensitive: bool = True,
             line_numbers: bool = False,
             encoding: Optional[str] = None) -> List[Union[str, Tuple[int, str]]]:
        """
        Search for pattern in file (like Unix grep)
        
        Args:
            file_path: Path to file
            pattern: Regex pattern to search
            case_sensitive: Case sensitive search
            line_numbers: Include line numbers
            encoding: Text encoding
            
        Returns:
            Matching lines with optional line numbers
        """
        file_path = Path(file_path)
        encoding = encoding or self.detect_encoding(file_path)
        
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        results = []
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for i, line in enumerate(f, 1):
                if regex.search(line):
                    if line_numbers:
                        results.append((i, line.rstrip('\n')))
                    else:
                        results.append(line.rstrip('\n'))
        
        return results
    
    def load_json(self, file_path: Union[str, Path]) -> Any:
        """Load JSON file"""
        content = self.read_file(file_path)
        return json.loads(content)
    
    def save_json(self, 
                  file_path: Union[str, Path],
                  data: Any,
                  indent: int = 2,
                  ensure_ascii: bool = False) -> bool:
        """Save data as JSON"""
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str)
            return self.write_file(file_path, content)
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    def load_yaml(self, file_path: Union[str, Path]) -> Any:
        """Load YAML file"""
        content = self.read_file(file_path)
        return yaml.safe_load(content)
    
    def save_yaml(self,
                  file_path: Union[str, Path],
                  data: Any,
                  default_flow_style: bool = False) -> bool:
        """Save data as YAML"""
        try:
            content = yaml.dump(data, default_flow_style=default_flow_style, default=str)
            return self.write_file(file_path, content)
        except Exception as e:
            logger.error(f"Failed to save YAML to {file_path}: {e}")
            return False
    
    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """Load pickle file"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def save_pickle(self, 
                   file_path: Union[str, Path],
                   data: Any,
                   protocol: int = pickle.HIGHEST_PROTOCOL) -> bool:
        """Save data as pickle"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle to {file_path}: {e}")
            return False
    
    def extract_archive(self,
                       archive_path: Union[str, Path],
                       extract_to: Optional[Union[str, Path]] = None,
                       format: Optional[str] = None) -> Path:
        """
        Extract archive file
        
        Args:
            archive_path: Path to archive
            extract_to: Extraction directory (default: same as archive)
            format: Archive format (auto-detect if None)
            
        Returns:
            Path to extracted content
        """
        archive_path = Path(archive_path)
        
        if extract_to is None:
            extract_to = archive_path.parent / archive_path.stem
        else:
            extract_to = Path(extract_to)
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format
        if format is None:
            if archive_path.suffix in {'.zip'}:
                format = 'zip'
            elif archive_path.name.endswith('.tar.gz') or archive_path.suffix == '.tgz':
                format = 'tar.gz'
            elif archive_path.name.endswith('.tar.bz2'):
                format = 'tar.bz2'
            elif archive_path.suffix == '.tar':
                format = 'tar'
            else:
                format = 'zip'  # Default
        
        # Extract based on format
        if format == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif format.startswith('tar'):
            mode = 'r:gz' if 'gz' in format else 'r:bz2' if 'bz2' in format else 'r'
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(extract_to)
        
        return extract_to
    
    def create_archive(self,
                      source: Union[str, Path],
                      archive_path: Union[str, Path],
                      format: str = 'zip') -> bool:
        """
        Create archive from directory or file
        
        Args:
            source: Source directory or file
            archive_path: Output archive path
            format: Archive format (zip, tar, tar.gz)
            
        Returns:
            Success status
        """
        try:
            source = Path(source)
            archive_path = Path(archive_path)
            
            if format == 'zip':
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    if source.is_dir():
                        for file_path in source.rglob('*'):
                            if file_path.is_file():
                                arcname = file_path.relative_to(source.parent)
                                zipf.write(file_path, arcname)
                    else:
                        zipf.write(source, source.name)
            
            elif format.startswith('tar'):
                mode = 'w:gz' if 'gz' in format else 'w:bz2' if 'bz2' in format else 'w'
                with tarfile.open(archive_path, mode) as tarf:
                    tarf.add(source, arcname=source.name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            return False
    
    def get_directory_size(self, directory: Union[str, Path]) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        directory = Path(directory)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def clean_directory(self,
                       directory: Union[str, Path],
                       pattern: str = '*',
                       older_than_days: Optional[int] = None,
                       dry_run: bool = False) -> List[Path]:
        """
        Clean directory by removing files matching criteria
        
        Args:
            directory: Directory to clean
            pattern: File pattern to match
            older_than_days: Remove files older than N days
            dry_run: Don't actually delete, just return what would be deleted
            
        Returns:
            List of deleted (or would-be deleted) files
        """
        directory = Path(directory)
        deleted = []
        
        for file_path in directory.rglob(pattern):
            if not file_path.is_file():
                continue
            
            # Check age
            if older_than_days:
                age_days = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days
                if age_days < older_than_days:
                    continue
            
            deleted.append(file_path)
            
            if not dry_run:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
        
        return deleted

# Convenience functions
file_utils = FileUtils()

def read_file(path: Union[str, Path], encoding: Optional[str] = None) -> str:
    """Read text file"""
    return file_utils.read_file(path, encoding)

def write_file(path: Union[str, Path], content: str, encoding: Optional[str] = None) -> bool:
    """Write text file"""
    return file_utils.write_file(path, content, encoding)

def get_file_hash(path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Get file hash"""
    return file_utils.calculate_hash(path, algorithm)

def is_binary_file(path: Union[str, Path]) -> bool:
    """Check if file is binary"""
    return file_utils.is_binary(path)

def find_files(directory: Union[str, Path], pattern: str = '*', recursive: bool = True) -> List[Path]:
    """Find files matching pattern"""
    return list(file_utils.find_files(directory, pattern, recursive))