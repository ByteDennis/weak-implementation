from typing import Dict, List
from upath import UPath

from ..core import Experiment, OutputManager
from ..utils.metrics import WMRCMetrics
from ..utils.logging import ExperimentLogger
from ..models.wmrc_model import WMRCModel, WMRCConfidenceComputer
from ..data.matlab_loader import MatlabDatasetLoader


class AdversarialExperiment(Experiment):
    # WMRC-specific experiment orchestrator extending base experiment

    def __init__(self, config: Dict, output_manager):
        # Initialize WMRC experiment with specific configuration
        super().__init__(config, output_manager)
        self.exp_logger = ExperimentLogger()
        self._setup_experiment_logging()

    def _setup_experiment_logging(self):
        # Configure experiment-specific logging
        run_name = f"wmrc_{self.config.get('dataset_name', 'unknown')}_{self.config.get('constraint_form', 'accuracy')}"
        log_level = self.config.get("log_level", "INFO")
        self.exp_logger.setup_logging(run_name, log_level)

    def setup(self) -> None:
        # Sets up WMRC label model and confidence computer
        self.exp_logger.log_experiment_start(self.config)

        try:
            self.exp_logger.log_progress("Initializing WMRC model")
            self.model = WMRCModel(self.config)

            self.exp_logger.log_progress("Setting up confidence computer")
            self.confidence_computer = WMRCConfidenceComputer(self.model, self.config)

            self.exp_logger.log_progress("Initializing data loader")
            self.data_loader = MatlabDatasetLoader()

            self.exp_logger.log_progress("Loading and preparing dataset")
            self._load_and_prepare_data()

            self.exp_logger.log_progress("Setup completed successfully")

        except Exception as e:
            self.exp_logger.log_error(e, "experiment setup")
            raise

    def run(self) -> Dict:
        # Executes single WMRC trial with given configuration
        n_runs = self.config.get("n_runs", 1)

        self.exp_logger.log_progress(f"Starting experiment with {n_runs} run(s)")

        try:
            if n_runs > 1:
                return self._run_multiple_trials(n_runs)
            return self._run_single_trial()
        except Exception as e:
            self.exp_logger.log_error(e, "experiment execution")
            self.exp_logger.log_experiment_end("failed")
            raise

    def run_grid_search(self) -> List[Dict]:
        # Runs grid search over WMRC parameter combinations
        self.exp_logger.log_progress("Starting grid search")

        try:
            from ..search.grid_search import GridSearchManager

            search_manager = GridSearchManager(self.config.get("search_space", {}))

            search_space = self.config.get("search_space", {})
            self.exp_logger.log_progress(f"Grid search space: {search_space}")

            results = search_manager.execute_search(self.config)
            self.exp_logger.log_progress(
                f"Grid search completed with {len(results)} configurations"
            )

            return results

        except Exception as e:
            self.exp_logger.log_error(e, "grid search execution")
            raise

    def _run_multiple_trials(self, n_runs: int) -> Dict:
        # Runs multiple WMRC trials and aggregates statistics
        self.exp_logger.log_progress(f"Running {n_runs} trials")
        results = []

        for run_no in range(n_runs):
            self.exp_logger.log_progress(
                f"Starting trial {run_no + 1}", run_no + 1, n_runs
            )

            try:
                trial_result = self._run_single_trial()
                results.append(trial_result)

                # Log key metrics from this trial
                if "accuracy" in trial_result:
                    self.exp_logger.log_progress(
                        f"Trial {run_no + 1} accuracy: {trial_result['accuracy']:.4f}"
                    )

            except Exception as e:
                self.exp_logger.log_error(e, f"trial {run_no + 1}")
                continue

        if not results:
            raise RuntimeError("All trials failed")

        aggregated_results = self._aggregate_trial_results(results)
        self.exp_logger.log_progress(
            f"Completed {len(results)}/{n_runs} successful trials"
        )

        return aggregated_results

    def save_results(self, results: Dict) -> None:
        # Saves WMRC results in .mat format for compatibility
        try:
            from scipy.io import savemat

            metrics = WMRCMetrics()
            filename = metrics.get_result_filename(
                self.config["dataset_name"],
                self.config.get("constraint_form", "accuracy"),
                **self.config,
            )

            save_path = self.output_manager.get_run_directory()
            full_path = UPath(save_path, filename)

            self.exp_logger.log_progress(f"Saving results to: {full_path}")
            savemat(full_path, results)

            self.exp_logger.log_checkpoint(
                full_path, 0, {"result_keys": list(results.keys())}
            )

        except Exception as e:
            self.exp_logger.log_error(e, "result saving")
            raise

    def _load_and_prepare_data(self) -> None:
        # Internal method to load and prepare dataset
        try:
            dataset_name = self.config["dataset_name"]
            dataset_prefix = self.config.get("dataset_prefix", "./datasets/")

            self.exp_logger.log_progress(
                f"Loading dataset: {dataset_name} from {dataset_prefix}"
            )

            data = self.data_loader.load_dataset(dataset_name, dataset_prefix)
            self.train_data, self.labeled_data, self.test_data = (
                self.data_loader.split_dataset(data, self.config)
            )

            # Log dataset statistics
            data_stats = {
                "train_size": (
                    len(self.train_data) if self.train_data is not None else 0
                ),
                "labeled_size": (
                    len(self.labeled_data) if self.labeled_data is not None else 0
                ),
                "test_size": len(self.test_data) if self.test_data is not None else 0,
            }

            self.exp_logger.log_progress("Dataset loaded and split successfully")
            for stat_name, stat_value in data_stats.items():
                self.exp_logger.log_progress(f"  {stat_name}: {stat_value}")

        except Exception as e:
            self.exp_logger.log_error(e, "data loading and preparation")
            raise

    def _run_single_trial(self) -> Dict:
        # Internal method to run single experimental trial
        try:
            self.exp_logger.log_progress("Starting model training")
            self.model.train((self.train_data, self.labeled_data))
            self.exp_logger.log_progress("Model training completed")

            results = {}

            # Evaluate on training data
            self.exp_logger.log_progress("Evaluating on training data")
            train_results = self.model.evaluate(self.train_data)
            results.update(train_results)

            # Log training performance
            if train_results:
                self.exp_logger.log_model_performance("WMRC_Train", train_results)

            # Evaluate on test data if available
            if self.test_data and self.config.get("use_test", True):
                self.exp_logger.log_progress("Evaluating on test data")
                test_results = self._evaluate_test_data()
                results.update(test_results)

                if test_results:
                    self.exp_logger.log_model_performance("WMRC_Test", test_results)

            # Compute confidence intervals if requested
            if self.config.get("get_confidences", False):
                self.exp_logger.log_progress("Computing confidence intervals")
                confidence_results = self._compute_confidence_intervals()
                results.update(confidence_results)

                if confidence_results:
                    self.exp_logger.log_model_performance(
                        "WMRC_Confidence", confidence_results
                    )

            self.exp_logger.log_progress("Trial completed successfully")
            self.exp_logger.log_experiment_end("completed")

            return results

        except Exception as e:
            self.exp_logger.log_error(e, "single trial execution")
            raise
