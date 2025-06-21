from typing import Dict

class ConfigManager:
    # Parses config.cfg using confection and validates structure
    
    def load_config(config_path: str) -> Dict:
        # Loads and validates configuration from file
        ...
    
    def validate_config(config: Dict) -> bool:
        # Ensures all required fields are present and valid
        ...
    
    def get_run_name(config: Dict) -> str:
        # Extracts run_name from configuration
        ...