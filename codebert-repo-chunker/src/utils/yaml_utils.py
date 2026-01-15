
import yaml
from typing import Any, Dict, Optional

class YAMLProcessor:
    """Processor for YAML files"""
    
    def load(self, content: str) -> Dict[str, Any]:
        """Load YAML content"""
        try:
            return yaml.safe_load(content)
        except Exception:
            return {}
