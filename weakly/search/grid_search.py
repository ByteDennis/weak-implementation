from typing import Dict, List

class GridSearchManager:
    # Manages grid search operations using wrench.search
    
    def __init__(self, search_space: Dict):
        # Initializes with parameter search space definition
        ...
    
    def generate_configurations(self) -> List[Dict]:
        # Generates all parameter combinations for grid search
        ...
    
    def execute_search(self, base_config: Dict) -> List[Dict]:
        # Runs grid search and returns all results
        ...
    
    def find_best_configuration(self, results: List[Dict]) -> Dict:
        # Identifies best performing configuration from results
        ...