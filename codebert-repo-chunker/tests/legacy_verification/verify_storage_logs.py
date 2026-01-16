
from src.storage.storage_manager import StorageConfig, StorageManager
from src.utils.logger import get_logger
import logging

# Configure logger to ensure we see INFO
# (Assuming get_logger handles config, but let's basicConfig just in case invalid env)
logging.basicConfig(level=logging.INFO)

print("--- Initializing Storage ---")
config = StorageConfig(base_path="data_test_verify")
manager = StorageManager(config)
print("--- Storage Initialized ---")

print("--- Closing Storage ---")
manager.close()
print("--- Done ---")
