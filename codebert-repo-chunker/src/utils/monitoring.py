"""
Monitoring and health check utilities
"""

import psutil
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from src.utils.logger import get_logger
import json
import requests

logger = get_logger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()

class HealthChecker:
    """System health checker"""
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize health checker
        
        Args:
            check_interval: Interval between checks in seconds
        """
        self.check_interval = check_interval
        self.checks: List[Callable] = []
        self.last_results: Dict[str, HealthCheck] = {}
        self.running = False
        self.thread = None
    
    def add_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Add a health check"""
        self.checks.append((name, check_func))
    
    def start(self):
        """Start health checking"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_checks, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop health checking"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _run_checks(self):
        """Run health checks periodically"""
        while self.running:
            self.check()
            time.sleep(self.check_interval)
    
    def check(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        # System checks
        results['memory'] = self._check_memory()
        results['cpu'] = self._check_cpu()
        results['disk'] = self._check_disk()
        
        # Custom checks
        for name, check_func in self.checks:
            try:
                result = check_func()
                results[name] = result
                
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e)
                )
                overall_status = HealthStatus.UNHEALTHY
        
        self.last_results = results
        
        return {
            'status': overall_status.value,
            'checks': {k: v.__dict__ for k, v in results.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_memory(self) -> HealthCheck:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage critical: {memory.percent}%"
        elif memory.percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Memory usage high: {memory.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory.percent}%"
        
        return HealthCheck(
            name="memory",
            status=status,
            message=message,
            metadata={
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3)
            }
        )
    
    def _check_cpu(self) -> HealthCheck:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.DEGRADED
            message = f"CPU usage high: {cpu_percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent}%"
        
        return HealthCheck(
            name="cpu",
            status=status,
            message=message,
            metadata={
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            }
        )
    
    def _check_disk(self) -> HealthCheck:
        """Check disk usage"""
        disk = psutil.disk_usage('/')
        
        if disk.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Disk usage critical: {disk.percent}%"
        elif disk.percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Disk usage high: {disk.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {disk.percent}%"
        
        return HealthCheck(
            name="disk",
            status=status,
            message=message,
            metadata={
                'percent': disk.percent,
                'free_gb': disk.free / (1024**3),
                'total_gb': disk.total / (1024**3)
            }
        )

class MetricsExporter:
    """Export metrics to external systems"""
    
    def __init__(self):
        """Initialize metrics exporter"""
        self.exporters = []
    
    def add_prometheus_exporter(self, pushgateway_url: str, job_name: str):
        """Add Prometheus pushgateway exporter"""
        self.exporters.append(('prometheus', pushgateway_url, job_name))
    
    def export(self, metrics: Dict[str, Any]):
        """Export metrics to all configured exporters"""
        for exporter_type, *args in self.exporters:
            if exporter_type == 'prometheus':
                self._export_to_prometheus(metrics, args[0], args[1])
    
    def _export_to_prometheus(self, metrics: Dict[str, Any], 
                             pushgateway_url: str, job_name: str):
        """Export metrics to Prometheus pushgateway"""
        try:
            # Format metrics for Prometheus
            data = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    data.append(f"{key} {value}")
            
            # Send to pushgateway
            response = requests.post(
                f"{pushgateway_url}/metrics/job/{job_name}",
                data='\n'.join(data),
                headers={'Content-Type': 'text/plain'}
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to export metrics to Prometheus: {e}")