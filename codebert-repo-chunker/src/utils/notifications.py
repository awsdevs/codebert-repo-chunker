"""
Notification utilities for alerts and updates
"""

import smtplib
import requests
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.utils.logger import get_logger

logger = get_logger(__name__)

class NotificationLevel(Enum):
    """Notification severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class NotificationConfig:
    """Configuration for notifications"""
    email_enabled: bool = False
    email_smtp_host: str = "localhost"
    email_smtp_port: int = 587
    email_from: str = ""
    email_to: List[str] = None
    email_username: str = ""
    email_password: str = ""
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#general"
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = None

class NotificationManager:
    """Manages notifications across multiple channels"""
    
    def __init__(self, config: NotificationConfig = None, 
                 channels: List[str] = None):
        """
        Initialize notification manager
        
        Args:
            config: Notification configuration
            channels: List of channels to enable
        """
        self.config = config or NotificationConfig()
        self.channels = channels or []
    
    def send(self, message: str, level: NotificationLevel = NotificationLevel.INFO,
            title: Optional[str] = None, metadata: Optional[Dict] = None):
        """Send notification to all configured channels"""
        
        if 'email' in self.channels and self.config.email_enabled:
            self._send_email(message, level, title, metadata)
        
        if 'slack' in self.channels and self.config.slack_enabled:
            self._send_slack(message, level, title, metadata)
        
        if 'webhook' in self.channels and self.config.webhook_enabled:
            self._send_webhook(message, level, title, metadata)
    
    def _send_email(self, message: str, level: NotificationLevel,
                   title: Optional[str], metadata: Optional[Dict]):
        """Send email notification"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to or [])
            msg['Subject'] = title or f"[{level.value.upper()}] Pipeline Notification"
            
            # Build body
            body = f"{message}\n\n"
            if metadata:
                body += "Details:\n"
                body += json.dumps(metadata, indent=2)
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config.email_smtp_host, 
                             self.config.email_smtp_port) as server:
                if self.config.email_username:
                    server.starttls()
                    server.login(self.config.email_username, 
                               self.config.email_password)
                
                server.send_message(msg)
                
            logger.info(f"Email notification sent: {title}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack(self, message: str, level: NotificationLevel,
                   title: Optional[str], metadata: Optional[Dict]):
        """Send Slack notification"""
        try:
            # Build Slack message
            color = {
                NotificationLevel.INFO: "good",
                NotificationLevel.WARNING: "warning",
                NotificationLevel.ERROR: "danger",
                NotificationLevel.CRITICAL: "danger"
            }.get(level, "")
            
            payload = {
                'channel': self.config.slack_channel,
                'attachments': [{
                    'title': title or "Pipeline Notification",
                    'text': message,
                    'color': color,
                    'fields': [
                        {'title': k, 'value': str(v), 'short': True}
                        for k, v in (metadata or {}).items()
                    ]
                }]
            }
            
            # Send to Slack
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload
            )
            response.raise_for_status()
            
            logger.info(f"Slack notification sent: {title}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_webhook(self, message: str, level: NotificationLevel,
                     title: Optional[str], metadata: Optional[Dict]):
        """Send webhook notification"""
        try:
            # Build payload
            payload = {
                'level': level.value,
                'title': title,
                'message': message,
                'metadata': metadata
            }
            
            # Send webhook
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=self.config.webhook_headers or {}
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent: {title}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def send_error(self, error: str, exception: Optional[Exception] = None):
        """Send error notification"""
        metadata = {}
        if exception:
            metadata['exception_type'] = type(exception).__name__
            metadata['exception_message'] = str(exception)
        
        self.send(
            error,
            level=NotificationLevel.ERROR,
            title="Pipeline Error",
            metadata=metadata
        )
    
    def send_completion(self, stats: Dict[str, Any]):
        """Send completion notification"""
        message = f"Pipeline completed successfully\n"
        message += f"Files processed: {stats.get('files_processed', 0)}\n"
        message += f"Chunks created: {stats.get('chunks_created', 0)}\n"
        message += f"Time taken: {stats.get('duration', 0):.2f} seconds"
        
        self.send(
            message,
            level=NotificationLevel.INFO,
            title="Pipeline Completed",
            metadata=stats
        )