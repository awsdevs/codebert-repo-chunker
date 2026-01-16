"""
Logging configuration for CodeBERT Repository Chunker
Provides comprehensive logging with multiple handlers, formatters, and filters
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import traceback
from functools import wraps
import time

# Try to import colorlog for colored console output
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

# Configuration paths
LOG_DIR = Path(__file__).parent.parent.parent / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Environment detection
ENV = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_JSON_LOGS = os.getenv("ENABLE_JSON_LOGS", "false").lower() == "true"
ENABLE_PERFORMANCE_LOGS = os.getenv("ENABLE_PERFORMANCE_LOGS", "true").lower() == "true"

@dataclass
class LogContext:
    """Context information for structured logging"""
    timestamp: str
    environment: str
    hostname: str
    process_id: int
    thread_name: str
    user: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    @classmethod
    def create(cls, **kwargs):
        """Create log context with system information"""
        import socket
        import threading
        
        return cls(
            timestamp=datetime.now().isoformat(),
            environment=ENV,
            hostname=socket.gethostname(),
            process_id=os.getpid(),
            thread_name=threading.current_thread().name,
            **kwargs
        )

class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'api_key', 'secret', 'credential',
        'aws_access_key', 'aws_secret_key', 'private_key'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log messages"""
        if hasattr(record, 'msg'):
            msg_lower = str(record.msg).lower()
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in msg_lower:
                    # Redact the value after the sensitive key
                    import re
                    record.msg = re.sub(
                        rf'{pattern}["\']?:\s*["\']?[^"\',\s}}]+',
                        f'{pattern}: ***REDACTED***',
                        str(record.msg),
                        flags=re.IGNORECASE
                    )
        
        # Also check args
        if hasattr(record, 'args') and record.args:
            record.args = self._redact_args(record.args)
        
        return True
    
    def _redact_args(self, args):
        """Redact sensitive data from log arguments"""
        if isinstance(args, dict):
            return {
                k: '***REDACTED***' if any(p in k.lower() for p in self.SENSITIVE_PATTERNS) else v
                for k, v in args.items()
            }
        return args

class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to logs"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to log record"""
        if ENABLE_PERFORMANCE_LOGS:
            import psutil
            
            # Add memory usage
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
            
            # Add timing if available
            if hasattr(record, 'duration'):
                record.duration_ms = record.duration * 1000
        
        return True

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.threadName,
            'process': record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'stack_info', 'exc_text']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)

class ColoredFormatter(colorlog.ColoredFormatter if COLORLOG_AVAILABLE else logging.Formatter):
    """Colored formatter for console output"""
    
    def __init__(self):
        if COLORLOG_AVAILABLE:
            super().__init__(
                fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            super().__init__(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

class DetailedFormatter(logging.Formatter):
    """Detailed formatter for file logs"""
    
    FORMATS = {
        logging.DEBUG: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
        logging.INFO: '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        logging.WARNING: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        logging.ERROR: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s\n%(exc_info)s',
        logging.CRITICAL: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s\n%(exc_info)s'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format based on log level"""
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        
        # Add exception info for ERROR and CRITICAL
        if record.levelno >= logging.ERROR and record.exc_info:
            record.exc_info = self.formatException(record.exc_info)
        else:
            record.exc_info = ''
        
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

class LoggerManager:
    """Centralized logger management"""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.setup_root_logger()
        self.setup_handlers()
    
    def setup_root_logger(self):
        """Configure root logger"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
        
        # Remove default handlers
        root_logger.handlers = []
    
    def setup_handlers(self):
        """Setup all logging handlers"""
        root_logger = logging.getLogger()
        
        # Console Handler
        console_handler = self._create_console_handler()
        root_logger.addHandler(console_handler)
        
        # File Handlers
        file_handler = self._create_file_handler()
        root_logger.addHandler(file_handler)
        
        # Rotating File Handler
        rotating_handler = self._create_rotating_handler()
        root_logger.addHandler(rotating_handler)
        
        # Error File Handler
        error_handler = self._create_error_handler()
        root_logger.addHandler(error_handler)
        
        # Performance Log Handler (if enabled)
        if ENABLE_PERFORMANCE_LOGS:
            perf_handler = self._create_performance_handler()
            root_logger.addHandler(perf_handler)
        
        # JSON Handler (if enabled)
        if ENABLE_JSON_LOGS:
            json_handler = self._create_json_handler()
            root_logger.addHandler(json_handler)
    
    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with colored output"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, LOG_LEVEL))
        
        if COLORLOG_AVAILABLE and ENV == 'development':
            handler.setFormatter(ColoredFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            ))
        
        # Add filters
        handler.addFilter(SensitiveDataFilter())
        
        return handler
    
    def _create_file_handler(self) -> logging.Handler:
        """Create standard file handler"""
        log_file = LOG_DIR / f"chunker_{datetime.now():%Y%m%d}.log"
        
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(DetailedFormatter())
        
        # Add filters
        handler.addFilter(SensitiveDataFilter())
        handler.addFilter(PerformanceFilter())
        
        return handler
    
    def _create_rotating_handler(self) -> logging.Handler:
        """Create rotating file handler"""
        log_file = LOG_DIR / "chunker.log"
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(DetailedFormatter())
        
        # Add filters
        handler.addFilter(SensitiveDataFilter())
        
        return handler
    
    def _create_error_handler(self) -> logging.Handler:
        """Create error-only file handler"""
        error_log = LOG_DIR / "errors.log"
        
        handler = logging.FileHandler(error_log, encoding='utf-8')
        handler.setLevel(logging.ERROR)
        handler.setFormatter(DetailedFormatter())
        
        # Add filters
        handler.addFilter(SensitiveDataFilter())
        
        return handler
    
    def _create_performance_handler(self) -> logging.Handler:
        """Create performance metrics handler"""
        perf_log = LOG_DIR / "performance.log"
        
        handler = logging.handlers.RotatingFileHandler(
            perf_log,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(JsonFormatter())
        
        # Add performance filter
        handler.addFilter(PerformanceFilter())
        
        return handler
    
    def _create_json_handler(self) -> logging.Handler:
        """Create JSON formatted handler for structured logging"""
        json_log = LOG_DIR / "structured.json"
        
        handler = logging.handlers.RotatingFileHandler(
            json_log,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(JsonFormatter())
        
        # Add filters
        handler.addFilter(SensitiveDataFilter())
        handler.addFilter(PerformanceFilter())
        
        return handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        return self._loggers[name]

# Singleton instance
logger_manager = LoggerManager()

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logger_manager.get_logger(name)

# Decorators for logging

def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"Completed {func.__name__}",
                    extra={'duration': execution_time}
                )
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Failed {func.__name__} after {execution_time:.2f}s: {str(e)}",
                    exc_info=True,
                    extra={'duration': execution_time}
                )
                raise
        
        return wrapper
    return decorator

def log_errors(logger: Optional[logging.Logger] = None):
    """Decorator to log exceptions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'args': str(args)[:200],  # Truncate long args
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
        
        return wrapper
    return decorator

def log_entry_exit(logger: Optional[logging.Logger] = None):
    """Decorator to log function entry and exit"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            logger.debug(
                f"Entering {func.__name__}",
                extra={
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }
            )
            
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"Exiting {func.__name__}",
                    extra={'result_type': type(result).__name__}
                )
                return result
            except Exception as e:
                logger.debug(
                    f"Exiting {func.__name__} with exception: {str(e)}"
                )
                raise
        
        return wrapper
    return decorator

class LoggingContext:
    """Context manager for temporary logging configuration"""
    
    def __init__(self, level: str = None, handler: logging.Handler = None):
        self.level = level
        self.handler = handler
        self.old_level = None
        self.logger = logging.getLogger()
    
    def __enter__(self):
        if self.level:
            self.old_level = self.logger.level
            self.logger.setLevel(getattr(logging, self.level))
        
        if self.handler:
            self.logger.addHandler(self.handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_level is not None:
            self.logger.setLevel(self.old_level)
        
        if self.handler:
            self.logger.removeHandler(self.handler)

# Specialized loggers for different components

class ChunkerLogger:
    """Specialized logger for chunking operations"""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"chunker.{name}")
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'errors': 0
        }
    
    def log_file_processing(self, file_path: str, file_type: str):
        """Log file processing start"""
        self.logger.info(
            f"Processing file: {file_path}",
            extra={
                'file_path': file_path,
                'file_type': file_type,
                'event': 'file_processing_start'
            }
        )
        self.stats['files_processed'] += 1
    
    def log_chunk_created(self, chunk_type: str, token_count: int):
        """Log chunk creation"""
        self.logger.debug(
            f"Created chunk: {chunk_type} ({token_count} tokens)",
            extra={
                'chunk_type': chunk_type,
                'token_count': token_count,
                'event': 'chunk_created'
            }
        )
        self.stats['chunks_created'] += 1
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log processing error"""
        self.logger.error(
            f"Processing error: {str(error)}",
            exc_info=True,
            extra={
                'error_type': type(error).__name__,
                'context': context,
                'event': 'processing_error'
            }
        )
        self.stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return self.stats.copy()

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = get_logger("performance")
        self.metrics = []
    
    def log_metric(self, name: str, value: float, unit: str = "ms", **tags):
        """Log a performance metric"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'value': value,
            'unit': unit,
            'tags': tags
        }
        
        self.metrics.append(metric)
        
        self.logger.info(
            f"Performance metric: {name}={value}{unit}",
            extra=metric
        )
    
    def log_batch_metrics(self):
        """Log accumulated metrics"""
        if self.metrics:
            self.logger.info(
                "Batch performance metrics",
                extra={'metrics': self.metrics}
            )
            self.metrics = []

# Utility functions

def setup_logging(
    level: str = None,
    log_file: str = None,
    enable_json: bool = None,
    enable_performance: bool = None
):
    """Setup logging configuration"""
    global LOG_LEVEL, ENABLE_JSON_LOGS, ENABLE_PERFORMANCE_LOGS
    
    if level:
        LOG_LEVEL = level
    if enable_json is not None:
        ENABLE_JSON_LOGS = enable_json
    if enable_performance is not None:
        ENABLE_PERFORMANCE_LOGS = enable_performance
    
    # Reinitialize logger manager
    global logger_manager
    logger_manager = LoggerManager()

def get_log_file_path(name: str = "chunker") -> Path:
    """Get path for a log file"""
    return LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

def archive_old_logs(days: int = 30):
    """Archive logs older than specified days"""
    import shutil
    from datetime import timedelta
    
    archive_dir = LOG_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for log_file in LOG_DIR.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            archive_path = archive_dir / log_file.name
            shutil.move(str(log_file), str(archive_path))
            
            # Compress archived logs
            shutil.make_archive(
                str(archive_path.with_suffix('')),
                'gzip',
                root_dir=str(archive_dir),
                base_dir=log_file.name
            )
            archive_path.unlink()

# Initialize logging on import
if __name__ != "__main__":
    logger_manager = LoggerManager()