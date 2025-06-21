from typing import Dict

class WMRCConfigSchema:
    # Configuration validation schemas for WMRC experiments
    
    def validate_config(self, config: Dict) -> bool:
        # Validates WMRC-specific configuration parameters
        ...
    
    def _validate_solver_config(self, solver_config: Dict) -> bool:
        # Validates optimization solver configuration (MOSEK, GUROBI)
        ...
    
    def _validate_confidence_config(self, conf_config: Dict) -> bool:
        # Validates confidence interval computation configuration
        ...
    
    def get_default_config(self) -> Dict:
        # Returns default WMRC configuration for quick setup
        ...