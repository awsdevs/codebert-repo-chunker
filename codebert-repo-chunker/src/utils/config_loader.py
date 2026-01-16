import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ConfigLoader:
    """Loads configuration from JSON file"""
    
    @staticmethod
    def load_config(config_path: str = "config.json") -> Dict[str, Any]:
        """
        Load config from JSON file. 
        Returns nested dict of values (stripping descriptions).
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
            
        try:
            text = path.read_text()
            data = json.loads(text)
            
            # Extract values from the verbose structure {key: {value: x, description: y}}
            clean_config = {}
            
            def extract_values(node: Dict[str, Any]) -> Dict[str, Any]:
                result = {}
                for key, item in node.items():
                    if key == 'description': continue
                    
                    if isinstance(item, dict):
                        if 'value' in item and 'description' in item:
                            # It's a leaf node with description
                            result[key] = item['value']
                        else:
                            # It's a nested section
                            nested = extract_values(item)
                            if nested:
                                result[key] = nested
                return result

            clean_config = extract_values(data)
            logger.info(f"Loaded configuration from {config_path}")
            return clean_config
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
