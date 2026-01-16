"""
Metrics collection and monitoring utilities
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from src.utils.logger import get_logger
import json

logger = get_logger(__name__)

@dataclass
class MetricValue:
    """Single metric value with timestamp"""
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float

class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter"""
        with self.lock:
            self.counters[name] += value
            
            # Also track as time series
            self.metrics[name].append(MetricValue(
                value=self.counters[name],
                timestamp=datetime.now(),
                tags=tags or {}
            ))
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        with self.lock:
            self.gauges[name] = value
            self.metrics[name].append(MetricValue(
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            ))
    
    def timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer value"""
        with self.lock:
            self.timers[name].append(duration)
            
            # Keep only recent values
            if len(self.timers[name]) > self.window_size:
                self.timers[name] = self.timers[name][-self.window_size:]
            
            self.metrics[name].append(MetricValue(
                value=duration,
                timestamp=datetime.now(),
                tags=tags or {}
            ))
    
    @contextmanager
    def measure_time(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager to measure execution time"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.timer(name, duration, tags)
    
    def get_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric"""
        with self.lock:
            if name in self.metrics:
                values = [m.value for m in self.metrics[name]]
                if values:
                    sorted_values = sorted(values)
                    count = len(values)
                    
                    return MetricSummary(
                        count=count,
                        sum=sum(values),
                        min=min(values),
                        max=max(values),
                        avg=sum(values) / count,
                        p50=sorted_values[count // 2],
                        p95=sorted_values[int(count * 0.95)],
                        p99=sorted_values[int(count * 0.99)]
                    )
            return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'timers': {k: self.get_summary(k) for k in self.timers},
                'summaries': {k: self.get_summary(k) for k in self.metrics}
            }
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.timers.clear()
    
    def export_json(self) -> str:
        """Export metrics as JSON"""
        metrics = self.get_all_metrics()
        
        # Convert summaries to dict
        for key in ['timers', 'summaries']:
            if key in metrics and metrics[key]:
                metrics[key] = {
                    k: v.__dict__ if v else None
                    for k, v in metrics[key].items()
                }
        
        return json.dumps(metrics, indent=2, default=str)

class SystemMetrics:
    """System resource metrics"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    @staticmethod
    def get_cpu_usage() -> Dict[str, float]:
        """Get CPU usage statistics"""
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
    
    @staticmethod
    def get_disk_usage(path: str = '/') -> Dict[str, float]:
        """Get disk usage statistics"""
        disk = psutil.disk_usage(path)
        return {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': disk.percent
        }