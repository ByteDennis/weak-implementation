import os

from typing import Dict, List
from .output_manager import OutputManager
from .logging import ExperimentLogger

class Experiment:
    # Main experiment class that coordinates all components
    
    def __init__(config: Dict):
        # Initializes experiment with configuration and output handler
        # Expect Logging Manager
        # Expect ExperimentLogger
        ...
    
    def setup(self):
        # Prepares datasets, models, and search strategies
        ...
    
    def run(self):
        # Executes the complete experiment pipeline
        ...
    
    def save_results(self):
        # Persists all results, configs, and logs to output directory
        ...
