from typing import Dict, List

class ExperimentRunner:
    # Handles the actual execution of experiments with different strategies
    
    def run_single_experiment(self, config: Dict) -> Dict:
        # Runs a single experiment configuration
        ...
    
    def run_grid_search(self, search_config: Dict) -> List[Dict]:
        # Executes grid search over parameter space
        ...
    
    def run_with_composer(self, composer_config: Dict):
        # Integrates with Composer for distributed training if needed
        ...