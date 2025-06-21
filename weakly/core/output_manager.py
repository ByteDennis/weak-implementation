from typing import Dict, List
from datetime import datetime
from upath import UPath
import whenever
import json
import pickle

class Base:
    # Manages all output artifacts in structured directories

    def create_run_directory(self, run_name: str):
        # Creates output/run_name/ directory structure
        ...

    def save_config(self, config: Dict):
        # Saves configuration to run directory
        ...

    def save_results(self, results: Dict):
        # Saves experiment results with timestamps
        ...

    def save_logs(self, logs: List[str]):
        # Saves execution logs and error traces
        ...

    def save_models(self, models: Dict):
        # Serializes and saves trained models
        ...


class OutputManager(Base):
    # Simple output management with structured directories

    def __init__(self, base_output_dir: str = "./output", remove_old = False):
        # Initialize output manager with base directory
        self.__base_output_dir = UPath(base_output_dir)
        self.__current_run_dir = None
        self.run_name = None
        self.__remove_old = remove_old
        self.start_time = self.end_time = None
        self.create_run_directory()

    def create_run_directory(self, run_name: str):
        # Creates output/run_name/ directory structure with subdirectories
        self.start_time = whenever.now().format("YYYYMMDD_HHMMSS")
        self.run_name = run_name
        self.__current_run_dir = self.__base_output_dir / run_name

        # Create main run directory
        self.__current_run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        subdirs = ['results', 'models', 'logs', 'configs', 'cache']
        for subdir in subdirs:
            (self.__current_run_dir / subdir).mkdir(exist_ok=True)


    @property
    def run_directory(self) -> UPath:
        # Returns current run directory path
        if self.__current_run_dir is None:
            raise RuntimeError("No run directory created. Call create_run_directory() first.")
        return self.__current_run_dir

    def save_config(self, config: Dict):
        # Saves configuration as JSON file
        config_dir = self.run_directory / 'configs'
        timestamp = whenever.now().format("YYYYMMDD_HHMMSS")

        json_path = config_dir / f"config_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)

        return str(json_path)

    def save_results(self, results: Dict):
        # Saves experiment results with timestamps
        if self.current_run_dir is None:
            raise RuntimeError("No run directory created. Call create_run_directory() first.")

        results_dir = self.current_run_dir / 'results'
        timestamp = whenever.now().format("YYYYMMDD_HHMMSS")

        # Add metadata to results
        enhanced_results = results.copy()
        enhanced_results['_metadata'] = {
            'timestamp': timestamp,
            'run_name': self.run_name,
            'save_time': whenever.now().format_rfc3339()
        }

        # Save as JSON
        json_path = results_dir / f"results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)

        return str(json_path)

    def save_logs(self, logs: List[str]):
        # Saves execution logs and error traces
        if self.current_run_dir is None:
            raise RuntimeError("No run directory created. Call create_run_directory() first.")

        logs_dir = self.current_run_dir / 'logs'
        timestamp = whenever.now().format("YYYYMMDD_HHMMSS")

        log_path = logs_dir / f"execution_log_{timestamp}.txt"
        with open(log_path, 'w') as f:
            f.write(f"Execution Log - {whenever.now().format_rfc3339()}\n")
            f.write("=" * 50 + "\n\n")
            for log_entry in logs:
                f.write(f"{log_entry}\n")

        return str(log_path)

    def save_models(self, models: Dict):
        # Serializes and saves trained models as pickle files
        if self.current_run_dir is None:
            raise RuntimeError("No run directory created. Call create_run_directory() first.")

        models_dir = self.current_run_dir / 'models'
        timestamp = whenever.now().format("YYYYMMDD_HHMMSS")
        saved_paths = {}

        for model_name, model_obj in models.items():
            pickle_path = models_dir / f"{model_name}_{timestamp}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(model_obj, f)
            saved_paths[model_name] = str(pickle_path)

        return saved_paths

    def _json_serializer(self, obj):
        # Custom JSON serializer for numpy arrays and other objects
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)
